from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.texture_analysis import (
    analyze_texture_image,
    compute_gabor_features,
    compute_glcm_features,
    compute_lbp_features,
    load_grayscale_image,
    summarize_texture,
)


def test_texture_analysis_outputs_on_valid_image(tmp_path: Path) -> None:
    image = np.zeros((64, 64), dtype=np.uint8)
    for y in range(64):
        for x in range(64):
            image[y, x] = (x * 3 + y * 2) % 256
    path = tmp_path / "fabric_texture.png"
    Image.fromarray(image).save(path)

    gray = load_grayscale_image(path)
    assert gray.shape[0] > 0 and gray.shape[1] > 0

    glcm = compute_glcm_features(gray)
    assert set(glcm) >= {"contrast", "dissimilarity", "homogeneity", "energy", "correlation", "asm"}
    assert all(np.isfinite(value) for value in glcm.values())
    assert glcm["contrast"] > 0

    lbp = compute_lbp_features(gray)
    assert "mean" in lbp and "variance" in lbp and "histogram" in lbp
    assert len(lbp["histogram"]) > 0
    assert np.isfinite(lbp["mean"]) and np.isfinite(lbp["variance"])

    gabor = compute_gabor_features(gray)
    assert set(gabor) >= {"mean_response", "variance", "dominant_orientation", "dominant_frequency"}
    assert np.isfinite(gabor["mean_response"]) and np.isfinite(gabor["variance"])

    summary = summarize_texture({"glcm": glcm, "lbp": lbp, "gabor": gabor})
    assert isinstance(summary, dict)
    assert "texture_type" in summary
    assert "observations" in summary

    analysis = analyze_texture_image(path)
    assert isinstance(analysis["glcm"], dict)
    assert isinstance(analysis["lbp"], dict)
    assert isinstance(analysis["gabor"], dict)
    assert isinstance(analysis["summary"], dict)
    assert "texture_type" in analysis["summary"]


def test_invalid_image_path_raises_clear_error() -> None:
    try:
        analyze_texture_image("not_a_real_path.png")
        raise AssertionError("Expected FileNotFoundError to be raised")
    except FileNotFoundError as exc:
        assert "not_a_real_path.png" in str(exc)
