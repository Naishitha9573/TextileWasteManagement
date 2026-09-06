from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from PIL import Image, UnidentifiedImageError

from material_classes import MATERIAL_CLASSES, VALID_MATERIAL_CLASSES

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tif", ".tiff"}
MIN_IMAGE_SIZE = (32, 32)


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_image_files(root: Path, recursive: bool = True) -> List[Path]:
    if not root.exists():
        return []
    if recursive:
        return sorted(p for p in root.rglob("*") if is_image_file(p))
    return sorted(p for p in root.iterdir() if is_image_file(p))


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_image_file(path: Path) -> Dict[str, Any]:
    result = {"path": str(path), "valid": False, "width": None, "height": None, "format": None, "error": None}
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            result["width"] = image.width
            result["height"] = image.height
            result["format"] = image.format
            result["valid"] = image.width >= MIN_IMAGE_SIZE[0] and image.height >= MIN_IMAGE_SIZE[1]
            if not result["valid"]:
                result["error"] = f"Image too small: {image.width}x{image.height}"
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        result["error"] = str(exc)
    return result


def find_duplicate_images(paths: Iterable[Path]) -> List[List[str]]:
    seen: Dict[str, List[str]] = {}
    for path in paths:
        try:
            digest = sha256sum(path)
        except OSError:
            continue
        seen.setdefault(digest, []).append(str(path))
    return [group for group in seen.values() if len(group) > 1]


def validate_material_dataset(dataset_root: str | Path) -> Dict[str, Any]:
    root = Path(dataset_root)
    image_paths = collect_image_files(root)
    class_counts: Counter[str] = Counter()
    valid_images = []
    corrupted = []
    duplicates = 0
    invalid_ext = []

    for path in image_paths:
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            invalid_ext.append(str(path))
            continue
        result = validate_image_file(path)
        if not result["valid"]:
            corrupted.append({"path": str(path), "error": result["error"]})
            continue
        valid_images.append(path)
        class_name = path.parent.name
        class_counts[class_name] += 1

    duplicate_groups = find_duplicate_images(valid_images)
    duplicates = sum(len(group) - 1 for group in duplicate_groups)

    valid_class_names = sorted(class_counts)
    missing = [cls for cls in MATERIAL_CLASSES if cls not in valid_class_names]
    available_valid = [cls for cls in VALID_MATERIAL_CLASSES if cls in valid_class_names]

    report = {
        "dataset_root": str(root),
        "total_images": len(image_paths),
        "valid_images": len(valid_images),
        "corrupted_images": len(corrupted),
        "duplicate_images": duplicates,
        "invalid_extensions": len(invalid_ext),
        "class_counts": dict(class_counts),
        "valid_classes": available_valid,
        "missing_classes": missing,
        "required_classes": MATERIAL_CLASSES,
        "status": "OK" if len(available_valid) >= 3 else "BLOCKED",
        "corrupted_details": corrupted[:20],
        "duplicate_groups": duplicate_groups[:10],
    }
    return report


def print_dataset_validation_report(report: Dict[str, Any]) -> str:
    lines = [
        "DATASET VALIDATION",
        "==================",
        f"Total images: {report['total_images']}",
        f"Valid: {report['valid_images']}",
        f"Corrupted: {report['corrupted_images']}",
        f"Duplicates: {report['duplicate_images']}",
        f"Classes: {len(report['valid_classes'])}",
        "",
        "Class distribution:",
    ]
    for label, count in sorted(report["class_counts"].items()):
        lines.append(f"- {label}: {count}")
    lines.extend([
        "",
        f"Required classes: {len(report['required_classes'])}",
        f"Available valid classes: {len(report['valid_classes'])}",
        f"Missing classes: {len(report['missing_classes'])}",
    ])
    if report["missing_classes"]:
        lines.append(f"Missing: {', '.join(report['missing_classes'])}")
    return "\n".join(lines)
