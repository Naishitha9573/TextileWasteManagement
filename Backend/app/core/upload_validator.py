"""Upload validation helper — MIME, extension, size, and corruption checks."""
from __future__ import annotations

import io
from typing import Tuple
from PIL import Image, ImageOps
 
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit


def validate_uploaded_image(
    file_bytes: bytes,
    filename: str,
    content_type: str,
    max_size: int = MAX_UPLOAD_SIZE_BYTES,
) -> Tuple[bool, str]:
    """
    Validates uploaded image file:
    - File size check
    - Extension check
    - Content-type MIME check
    - PIL readability & non-zero dimension check
    Returns (is_valid, error_message)
    """
    if not file_bytes:
        return False, "Uploaded file is empty"

    if len(file_bytes) > max_size:
        max_mb = max_size / (1024 * 1024)
        return False, f"File size exceeds maximum limit of {max_mb:.1f} MB"

    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file extension '{ext}'. Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}"

    if content_type and content_type.lower() not in ALLOWED_MIME_TYPES and content_type != "application/octet-stream":
        return False, f"Unsupported Content-Type '{content_type}'. Allowed MIME types: {', '.join(sorted(ALLOWED_MIME_TYPES))}"

    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.verify()  # Check for file corruption
    except Exception as exc:
        return False, f"Corrupted or invalid image file: {exc}"

    try:
        # Re-open after verify() to inspect dimensions
        image = Image.open(io.BytesIO(file_bytes))
        image = ImageOps.exif_transpose(image)
        width, height = image.size
        if width <= 0 or height <= 0:
            return False, f"Invalid image dimensions: {width}x{height}"
    except Exception as exc:
        return False, f"Failed to parse image headers: {exc}"

    return True, ""
