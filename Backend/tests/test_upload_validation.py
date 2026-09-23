"""Unit tests for image upload security validation."""
import io
import pytest
from PIL import Image
from app.core.upload_validator import validate_uploaded_image


def _create_valid_image_bytes(fmt="JPEG", size=(100, 100)) -> bytes:
    img = Image.new("RGB", size, color="blue")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_valid_image_upload():
    jpeg_bytes = _create_valid_image_bytes("JPEG")
    valid, err = validate_uploaded_image(jpeg_bytes, "test.jpg", "image/jpeg")
    assert valid is True
    assert err == ""

    png_bytes = _create_valid_image_bytes("PNG")
    valid, err = validate_uploaded_image(png_bytes, "sample.png", "image/png")
    assert valid is True
    assert err == ""


def test_empty_file():
    valid, err = validate_uploaded_image(b"", "test.jpg", "image/jpeg")
    assert valid is False
    assert "empty" in err.lower()


def test_oversized_file():
    # Mock small max limit
    jpeg_bytes = _create_valid_image_bytes("JPEG")
    valid, err = validate_uploaded_image(jpeg_bytes, "test.jpg", "image/jpeg", max_size=10)
    assert valid is False
    assert "exceeds" in err.lower()


def test_invalid_extension():
    jpeg_bytes = _create_valid_image_bytes("JPEG")
    valid, err = validate_uploaded_image(jpeg_bytes, "script.exe", "image/jpeg")
    assert valid is False
    assert "extension" in err.lower()


def test_corrupted_bytes():
    corrupt_bytes = b"NOT_AN_IMAGE_PAYLOAD_BYTES"
    valid, err = validate_uploaded_image(corrupt_bytes, "photo.jpg", "image/jpeg")
    assert valid is False
    assert "corrupted" in err.lower() or "invalid" in err.lower()
