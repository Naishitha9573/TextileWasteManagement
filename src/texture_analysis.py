"""Standalone handcrafted texture analysis for fabric images.

This module is intentionally independent of the EfficientNet classifier and its
training data. It produces interpretable GLCM, LBP, and Gabor descriptors for
one image at a time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_GLCM_LEVELS = 16
DEFAULT_LBP_POINTS = 8
DEFAULT_GABOR_FREQUENCIES = (0.10, 0.20, 0.30)
DEFAULT_GABOR_ORIENTATIONS = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
MAX_TEXTURE_DIMENSION = 1024
MAX_SOURCE_PIXELS = 100_000_000


def load_grayscale_image(path: str | Path, max_size: int = MAX_TEXTURE_DIMENSION) -> np.ndarray:
    """Load a safely bounded grayscale image as an efficient uint8 array."""
    if max_size < 1:
        raise ValueError("max_size must be positive")
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Texture image not found: {image_path}")
    with Image.open(image_path) as image:
        source_width, source_height = image.size
        if source_width * source_height > MAX_SOURCE_PIXELS:
            raise ValueError("texture image dimensions are too large to process safely")
        image = image.convert("L")
        if max(image.size) > max_size:
            scale = max_size / max(image.size)
            size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            image = image.resize(size, Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


def _quantize(image: np.ndarray, levels: int) -> np.ndarray:
    if image.ndim != 2 or image.size == 0:
        raise ValueError("image must be a non-empty two-dimensional array")
    if levels < 2:
        raise ValueError("levels must be at least 2")
    clipped = np.clip(image, 0, 255).astype(np.float32) if image.dtype == np.uint8 else np.clip(image, 0.0, 1.0) * 255
    return np.minimum((clipped * levels / 256).astype(np.int32), levels - 1)


def compute_glcm_features(image: np.ndarray, levels: int = DEFAULT_GLCM_LEVELS) -> dict[str, float]:
    """Compute averaged Haralick-style features over four one-pixel directions."""
    quantized = _quantize(image, levels)
    matrices: list[np.ndarray] = []
    for dy, dx in ((0, 1), (1, 1), (1, 0), (1, -1)):
        source_y = slice(max(0, -dy), min(quantized.shape[0], quantized.shape[0] - dy))
        source_x = slice(max(0, -dx), min(quantized.shape[1], quantized.shape[1] - dx))
        target_y = slice(max(0, dy), min(quantized.shape[0], quantized.shape[0] + dy))
        target_x = slice(max(0, dx), min(quantized.shape[1], quantized.shape[1] + dx))
        source = quantized[source_y, source_x].ravel()
        target = quantized[target_y, target_x].ravel()
        matrix = np.zeros((levels, levels), dtype=np.float32)
        np.add.at(matrix, (source, target), 1)
        matrix += matrix.T
        total = matrix.sum()
        matrices.append(matrix / total if total else matrix)

    values = {name: [] for name in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "asm")}
    row_indices, column_indices = np.indices((levels, levels))
    for matrix in matrices:
        row_mean = (matrix.sum(axis=1) * np.arange(levels)).sum()
        column_mean = (matrix.sum(axis=0) * np.arange(levels)).sum()
        row_std = np.sqrt((matrix.sum(axis=1) * (np.arange(levels) - row_mean) ** 2).sum())
        column_std = np.sqrt((matrix.sum(axis=0) * (np.arange(levels) - column_mean) ** 2).sum())
        values["contrast"].append(float(((row_indices - column_indices) ** 2 * matrix).sum()))
        values["dissimilarity"].append(float((np.abs(row_indices - column_indices) * matrix).sum()))
        values["homogeneity"].append(float((matrix / (1 + (row_indices - column_indices) ** 2)).sum()))
        values["asm"].append(float((matrix**2).sum()))
        values["energy"].append(float(np.sqrt(values["asm"][-1])))
        correlation = 0.0 if row_std == 0 or column_std == 0 else float(
            (((row_indices - row_mean) * (column_indices - column_mean) * matrix).sum())
            / (row_std * column_std)
        )
        values["correlation"].append(correlation)
    return {name: float(np.mean(feature_values)) for name, feature_values in values.items()}


def compute_lbp_features(image: np.ndarray, points: int = DEFAULT_LBP_POINTS, radius: float = 1.0) -> dict[str, Any]:
    """Compute a basic circular LBP histogram and its first two moments."""
    if points < 1 or radius <= 0:
        raise ValueError("points must be positive and radius must be greater than zero")
    if points > 31:
        raise ValueError("points must be 31 or fewer to fit in an integer descriptor")
    descriptor = _compute_lbp_map(image, points, radius)
    bins = 2**points
    histogram = np.bincount(descriptor.ravel(), minlength=bins).astype(np.float32)
    histogram /= histogram.sum()
    return {
        "mean": float(descriptor.mean()),
        "variance": float(descriptor.var()),
        "histogram": histogram.tolist(),
        "points": points,
        "radius": float(radius),
    }


def _compute_lbp_map(image: np.ndarray, points: int, radius: float) -> np.ndarray:
    padding = int(np.ceil(radius))
    padded = np.pad(image, padding, mode="edge")
    center = padded[padding : padding + image.shape[0], padding : padding + image.shape[1]]
    descriptor = np.zeros(image.shape, dtype=np.uint32)
    for point in range(points):
        angle = 2 * np.pi * point / points
        offset_y = int(round(np.sin(angle) * radius))
        offset_x = int(round(np.cos(angle) * radius))
        neighbor = padded[
            padding + offset_y : padding + offset_y + image.shape[0],
            padding + offset_x : padding + offset_x + image.shape[1],
        ]
        descriptor |= (neighbor >= center).astype(np.uint32) << point
    return descriptor


def _gabor_kernel(size: int, frequency: float, orientation: float, sigma: float) -> np.ndarray:
    coordinates = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
    x, y = np.meshgrid(coordinates, coordinates)
    rotated_x = x * np.cos(orientation) + y * np.sin(orientation)
    rotated_y = -x * np.sin(orientation) + y * np.cos(orientation)
    envelope = np.exp(-(rotated_x**2 + rotated_y**2) / (2 * sigma**2))
    kernel = envelope * np.cos(2 * np.pi * frequency * rotated_x)
    kernel -= kernel.mean()
    norm = np.sqrt((kernel**2).sum())
    return kernel / norm if norm else kernel


def _convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    padding = kernel.shape[0] // 2
    padded = np.pad(image.astype(np.float32, copy=False), padding, mode="reflect")
    output = np.empty(image.shape, dtype=np.float32)
    kernel = kernel.astype(np.float32, copy=False)
    chunk_rows = max(1, 1_048_576 // (image.shape[1] * kernel.size))
    for start in range(0, image.shape[0], chunk_rows):
        stop = min(image.shape[0], start + chunk_rows)
        windows = np.lib.stride_tricks.sliding_window_view(
            padded[start : stop + 2 * padding], kernel.shape
        )
        output[start:stop] = np.einsum("ijkl,kl->ij", windows, kernel, optimize=True)
    return output


def compute_gabor_features(
    image: np.ndarray,
    frequencies: tuple[float, ...] = DEFAULT_GABOR_FREQUENCIES,
    orientations: tuple[float, ...] = DEFAULT_GABOR_ORIENTATIONS,
) -> dict[str, Any]:
    """Compute response energy by frequency and orientation using Gabor filters."""
    if not frequencies or not orientations or any(value <= 0 for value in frequencies):
        raise ValueError("frequencies and orientations must be non-empty; frequencies must be positive")
    responses: list[tuple[float, float, float]] = []
    for frequency in frequencies:
        size = max(7, int(round(6 / frequency)) | 1)
        sigma = max(1.5, 0.56 / frequency)
        for orientation in orientations:
            response = _convolve(image, _gabor_kernel(size, frequency, orientation, sigma))
            responses.append((frequency, orientation, float(np.mean(np.abs(response)))))
    dominant_frequency, dominant_orientation, _ = max(responses, key=lambda item: item[2])
    response_values = np.asarray([item[2] for item in responses], dtype=np.float32)
    return {
        "mean_response": float(response_values.mean()),
        "variance": float(response_values.var()),
        "dominant_frequency": float(dominant_frequency),
        "dominant_orientation": float(np.degrees(dominant_orientation) % 180),
        "responses": [
            {"frequency": float(frequency), "orientation": float(np.degrees(orientation) % 180), "mean_abs_response": value}
            for frequency, orientation, value in responses
        ],
    }


def summarize_texture(features: dict[str, Any]) -> dict[str, Any]:
    """Turn descriptor values into cautious, human-readable observations."""
    glcm = features["glcm"]
    lbp = features["lbp"]
    gabor = features["gabor"]
    observations = []
    observations.append("Higher local contrast" if glcm["contrast"] >= 8 else "Lower local contrast")
    observations.append("More regular local patterns" if glcm["homogeneity"] >= 0.65 else "More varied local patterns")
    observations.append(f"Strongest directional response near {gabor['dominant_orientation']:.0f} degrees")
    texture_type = "structured / directional" if gabor["variance"] > 0.0001 else "uniform / low-frequency"
    return {"texture_type": texture_type, "observations": observations, "interpretation_note": "Descriptors are analytical signals, not fabric-class predictions."}


def analyze_texture_image(path: str | Path) -> dict[str, Any]:
    """Analyze one image and return JSON-serializable texture descriptors."""
    image = load_grayscale_image(path)
    features: dict[str, Any] = {
        "image": str(Path(path)),
        "glcm": compute_glcm_features(image),
        "lbp": compute_lbp_features(image),
        "gabor": compute_gabor_features(image),
    }
    features["summary"] = summarize_texture(features)
    return features


def save_texture_visualization(path: str | Path, output_path: str | Path) -> None:
    """Save a compact visual diagnostic of the image and its texture maps."""
    import matplotlib.pyplot as plt

    image = load_grayscale_image(path)
    with Image.open(path) as original:
        original = original.convert("RGB")
    lbp_map = _compute_lbp_map(image, DEFAULT_LBP_POINTS, 1.0)
    gabor_response = np.abs(_convolve(image, _gabor_kernel(21, 0.2, 0.0, 2.8)))
    figure, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[1].imshow(image, cmap="gray")
    axes[1].set_title("Grayscale")
    axes[2].imshow(lbp_map, cmap="viridis")
    axes[2].set_title("LBP map")
    axes[3].imshow(gabor_response, cmap="magma")
    axes[3].set_title("Gabor response")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze fabric texture independently of the classifier")
    parser.add_argument("image", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--visualize", type=Path, help="Optional PNG path for a diagnostic visualization")
    arguments = parser.parse_args()
    result = analyze_texture_image(arguments.image)
    serialized = json.dumps(result, indent=2)
    if arguments.json_out:
        arguments.json_out.write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)
    if arguments.visualize:
        save_texture_visualization(arguments.image, arguments.visualize)


if __name__ == "__main__":
    main()
