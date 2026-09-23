"""Deterministic dominant-color analysis for uploaded fabric images."""
from __future__ import annotations

import colorsys
import io
from typing import Any

from PIL import Image, ImageOps


ANALYSIS_SIZE = (160, 160)
PALETTE_SIZE = 3


def _color_name(rgb: tuple[int, int, int]) -> str:
    red, green, blue = (channel / 255 for channel in rgb)
    hue, saturation, value = colorsys.rgb_to_hsv(red, green, blue)
    if value < 0.16:
        return "Black"
    if saturation < 0.12:
        if value > 0.88:
            return "White"
        if value > 0.55:
            return "Gray"
        return "Charcoal"
    if saturation < 0.32 and value > 0.78:
        return "Cream" if red >= blue and green >= blue else "Gray"
    degrees = hue * 360
    if degrees < 15 or degrees >= 345:
        return "Red" if value < 0.65 else "Pink" if saturation < 0.55 else "Red"
    if degrees < 45:
        return "Brown" if value < 0.62 else "Orange"
    if degrees < 70:
        return "Yellow"
    if degrees < 165:
        return "Green"
    if degrees < 195:
        return "Cyan"
    if degrees < 255:
        return "Navy" if value < 0.55 else "Blue"
    if degrees < 315:
        return "Purple"
    return "Pink"


def _color_details(rgb: tuple[int, int, int], percentage: float) -> dict[str, Any]:
    return {
        "name": _color_name(rgb),
        "percentage": round(percentage, 1),
        "rgb": list(rgb),
        "hex": "#{:02X}{:02X}{:02X}".format(*rgb),
    }


def analyze_image_color(image: Image.Image) -> dict[str, Any]:
    """Return a quantized three-color palette derived from actual image pixels."""
    normalized = ImageOps.exif_transpose(image).convert("RGB")
    if normalized.width == 0 or normalized.height == 0:
        raise ValueError("Image has no pixels")
    sampled = normalized.resize(ANALYSIS_SIZE, Image.Resampling.LANCZOS)
    quantized = sampled.quantize(colors=PALETTE_SIZE, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    counts = quantized.getcolors(maxcolors=ANALYSIS_SIZE[0] * ANALYSIS_SIZE[1]) or []
    palette = quantized.getpalette()
    total = sum(count for count, _ in counts)
    colors = []
    for count, palette_index in sorted(counts, reverse=True)[:PALETTE_SIZE]:
        offset = palette_index * 3
        rgb = tuple(int(channel) for channel in palette[offset:offset + 3])
        colors.append((count, rgb))
    dominant_colors = [_color_details(rgb, count * 100 / total) for count, rgb in colors]
    print(
        f"[COLOR ANALYSIS] image={normalized.width}x{normalized.height} "
        f"dominant={dominant_colors[0]} top_colors={dominant_colors}"
    )
    return {
        "color": dominant_colors[0],
        "dominant_colors": dominant_colors,
    }


def analyze_image_color_bytes(image_bytes: bytes) -> dict[str, Any]:
    if not image_bytes:
        raise ValueError("Uploaded image is empty")
    with Image.open(io.BytesIO(image_bytes)) as image:
        return analyze_image_color(image)
