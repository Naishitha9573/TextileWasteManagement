"""Lightweight validator for material-classification image datasets.

Scans a processed dataset directory (one subfolder per class containing
images) and reports which classes are present, how many images each holds,
and which expected classes are missing or empty.
"""

from pathlib import Path

from material_classes import MATERIAL_CLASSES

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_material_dataset(dataset_path):
    """Return an honest report about a dataset directory's class coverage.

    The report contains:
    - total_images: number of readable image files found
    - class_counts: {class_name: count} for every non-empty class folder
    - valid_classes: classes with at least one image
    - missing_classes: expected MATERIAL_CLASSES with no folder/images
    - unreadable_images: files with non-image extensions inside class folders
    """
    dataset_path = Path(dataset_path)
    report = {
        "dataset_path": str(dataset_path),
        "exists": dataset_path.is_dir(),
        "total_images": 0,
        "class_counts": {},
        "valid_classes": [],
        "missing_classes": [],
        "unreadable_images": 0,
    }
    if not report["exists"]:
        report["missing_classes"] = list(MATERIAL_CLASSES)
        return report

    for entry in sorted(dataset_path.iterdir()):
        if not entry.is_dir():
            continue
        image_count = 0
        for file in entry.iterdir():
            if file.suffix.lower() in IMAGE_EXTENSIONS:
                image_count += 1
            elif not file.name.startswith("."):
                report["unreadable_images"] += 1
        if image_count > 0:
            report["class_counts"][entry.name] = image_count
            report["valid_classes"].append(entry.name)

    report["total_images"] = sum(report["class_counts"].values())
    present = set(report["class_counts"])
    report["missing_classes"] = [c for c in MATERIAL_CLASSES if c not in present]
    return report
