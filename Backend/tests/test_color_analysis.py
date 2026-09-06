import io

from PIL import Image

from app.services.color_analysis import analyze_image_color_bytes


def _image_bytes(color):
    image = Image.new("RGB", (40, 40), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_color_analysis_uses_actual_pixels():
    red = analyze_image_color_bytes(_image_bytes((220, 30, 25)))
    blue = analyze_image_color_bytes(_image_bytes((25, 60, 220)))

    assert red["color"]["rgb"] != blue["color"]["rgb"]
    assert red["color"]["name"] == "Red"
    assert blue["color"]["name"] == "Blue"


def test_color_analysis_returns_valid_palette():
    result = analyze_image_color_bytes(_image_bytes((238, 235, 228)))

    assert 0 < result["color"]["percentage"] <= 100
    assert len(result["color"]["rgb"]) == 3
    assert all(0 <= channel <= 255 for channel in result["color"]["rgb"])
    assert result["color"]["hex"].startswith("#")
    assert len(result["color"]["hex"]) == 7
    assert result["dominant_colors"] == sorted(
        result["dominant_colors"], key=lambda color: color["percentage"], reverse=True
    )
    assert abs(sum(color["percentage"] for color in result["dominant_colors"]) - 100) <= 0.2
