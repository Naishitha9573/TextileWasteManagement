"""Read-only analysis of the original iBUG Fabrics image dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = ROOT / "Backend" / "datasets" / "ibug_fabrics"
REPORTS_PATH = ROOT / "reports"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze an iBUG Fabrics dataset without modifying it.")
    parser.add_argument(
        "dataset_path",
        nargs="?",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Original dataset directory (default: {DEFAULT_DATASET_PATH})",
    )
    return parser.parse_args()


def collect_class_folders(dataset_path: Path) -> list[Path]:
    """Treat each immediate directory under the dataset root as an original class folder."""
    try:
        return sorted(path for path in dataset_path.iterdir() if path.is_dir())
    except OSError as exc:
        raise RuntimeError(f"Unable to read dataset directory '{dataset_path}': {exc}") from exc


def analyze_image(path: Path) -> dict[str, Any]:
    """Read dimensions while leaving the source file untouched."""
    result: dict[str, Any] = {
        "path": str(path),
        "extension": path.suffix.lower(),
        "readable": False,
        "width": None,
        "height": None,
        "error": None,
    }
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            result.update({"readable": True, "width": image.width, "height": image.height})
    except (OSError, SyntaxError, UnidentifiedImageError, ValueError) as exc:
        result["error"] = str(exc)
    return result


def build_class_record(class_folder: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    image_paths = sorted(
        path
        for path in class_folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    image_records = [analyze_image(path) for path in image_paths]
    readable = [record for record in image_records if record["readable"]]
    widths = [record["width"] for record in readable]
    heights = [record["height"] for record in readable]
    extension_counts = Counter(record["extension"] for record in image_records)

    record = {
        "class_name": class_folder.name,
        "image_count": len(image_records),
        "readable_image_count": len(readable),
        "corrupted_image_count": len(image_records) - len(readable),
        "file_extensions": dict(sorted(extension_counts.items())),
        "min_width": min(widths) if widths else None,
        "max_width": max(widths) if widths else None,
        "average_width": round(sum(widths) / len(widths), 2) if widths else None,
        "min_height": min(heights) if heights else None,
        "max_height": max(heights) if heights else None,
        "average_height": round(sum(heights) / len(heights), 2) if heights else None,
        "empty": not image_records,
    }
    return record, image_records


def create_visualization(class_records: list[dict[str, Any]], output_path: Path) -> None:
    labels = [record["class_name"] for record in class_records]
    counts = [record["image_count"] for record in class_records]
    figure_width = max(12, len(labels) * 0.55)
    figure, axis = plt.subplots(figsize=(figure_width, 7))
    axis.bar(labels, counts, color="#267a78")
    axis.set_title("iBUG Fabrics Image Distribution")
    axis.set_xlabel("Original class folder")
    axis.set_ylabel("Supported image files")
    axis.tick_params(axis="x", labelrotation=65)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def print_summary(summary: dict[str, Any]) -> None:
    print("iBUG FABRICS DATASET ANALYSIS")
    print("==============================")
    print(f"Dataset: {summary['dataset_path']}")
    print(f"Class folders: {summary['total_classes']}")
    print(f"Supported images: {summary['total_images']}")
    print(f"Readable images: {summary['readable_images']}")
    print(f"Corrupted/unreadable images: {summary['corrupted_images']}")
    print(f"Empty class folders: {len(summary['empty_class_folders'])}")
    print("\nImages per class:")
    for record in summary["class_records"]:
        print(f"  {record['class_name']}: {record['image_count']}")
    print("\nReports:")
    print(f"  {REPORTS_PATH / 'dataset_analysis.csv'}")
    print(f"  {REPORTS_PATH / 'dataset_summary.json'}")
    print(f"  {REPORTS_PATH / 'class_distribution.png'}")


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path.expanduser().resolve()
    if not dataset_path.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {dataset_path}")

    class_records: list[dict[str, Any]] = []
    image_records: list[dict[str, Any]] = []
    for class_folder in collect_class_folders(dataset_path):
        class_record, records = build_class_record(class_folder)
        class_records.append(class_record)
        for image_record in records:
            image_record["class_name"] = class_folder.name
        image_records.extend(records)

    readable = [record for record in image_records if record["readable"]]
    widths = [record["width"] for record in readable]
    heights = [record["height"] for record in readable]
    extension_counts = Counter(record["extension"] for record in image_records)
    corrupted = [record for record in image_records if not record["readable"]]
    empty_classes = [record["class_name"] for record in class_records if record["empty"]]

    summary: dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "total_classes": len(class_records),
        "total_images": len(image_records),
        "readable_images": len(readable),
        "corrupted_images": len(corrupted),
        "file_extensions": dict(sorted(extension_counts.items())),
        "dimensions": {
            "min_width": min(widths) if widths else None,
            "max_width": max(widths) if widths else None,
            "average_width": round(sum(widths) / len(widths), 2) if widths else None,
            "min_height": min(heights) if heights else None,
            "max_height": max(heights) if heights else None,
            "average_height": round(sum(heights) / len(heights), 2) if heights else None,
        },
        "empty_class_folders": empty_classes,
        "missing_class_folders": [],
        "missing_class_folders_note": "No expected final class list was assumed; missing folders require an external expected-class list.",
        "class_records": class_records,
        "corrupted_files": corrupted,
        "read_only": True,
    }

    REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    csv_records = []
    for record in class_records:
        csv_records.append(record | {"file_extensions": json.dumps(record["file_extensions"], sort_keys=True)})
    pd.DataFrame(csv_records).to_csv(REPORTS_PATH / "dataset_analysis.csv", index=False)
    (REPORTS_PATH / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    create_visualization(class_records, REPORTS_PATH / "class_distribution.png")
    print_summary(summary)


if __name__ == "__main__":
    main()