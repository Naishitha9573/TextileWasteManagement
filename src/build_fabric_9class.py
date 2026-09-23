"""Create and verify the focused 9-class dataset from the cleaned 16-class dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "dataset" / "processed" / "fabric_16class" / "all"
OUTPUT_DATASET = ROOT / "dataset" / "processed" / "fabric_9class" / "all"
REPORTS = ROOT / "reports"
FINAL_CLASSES = ["Cotton", "Polyester", "Denim", "Wool", "Nylon", "Viscose", "Silk", "Fleece", "Terrycloth"]
EXCLUDED_CLASSES = ["Acrylic", "Chenille", "Corduroy", "Crepe", "Linen", "Satin", "Velvet"]
EXPECTED_COUNTS = {
    "Cotton": 2320, "Polyester": 852, "Denim": 644, "Wool": 344, "Nylon": 228,
    "Viscose": 148, "Silk": 144, "Fleece": 132, "Terrycloth": 120,
}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUMMARY_FIELDS = ["class_name", "source_count", "copied_count", "missing_count"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and verify the focused 9-class dataset.")
    parser.add_argument("source_path", nargs="?", type=Path, default=DEFAULT_SOURCE)
    return parser.parse_args()


def image_paths(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_image(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, SyntaxError, UnidentifiedImageError, ValueError) as exc:
        raise RuntimeError(f"Unreadable copied image: {path}: {exc}") from exc


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def main() -> int:
    args = parse_args()
    source = args.source_path.expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"16-class source dataset does not exist: {source}")
    missing_source_classes = [class_name for class_name in FINAL_CLASSES if not (source / class_name).is_dir()]
    if missing_source_classes:
        raise SystemExit(f"Missing source class folders: {', '.join(missing_source_classes)}")

    source_paths = {class_name: image_paths(source / class_name) for class_name in FINAL_CLASSES}
    source_counts = {class_name: len(paths) for class_name, paths in source_paths.items()}
    OUTPUT_DATASET.mkdir(parents=True, exist_ok=True)
    for class_name in FINAL_CLASSES:
        (OUTPUT_DATASET / class_name).mkdir(exist_ok=True)
        for source_path in source_paths[class_name]:
            relative = source_path.relative_to(source / class_name)
            destination = OUTPUT_DATASET / class_name / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)
            verify_image(destination)
            if sha256(source_path) != sha256(destination):
                raise RuntimeError(f"Copied file hash mismatch: {source_path}")

    copied_paths = {class_name: image_paths(OUTPUT_DATASET / class_name) for class_name in FINAL_CLASSES}
    copied_counts = {class_name: len(paths) for class_name, paths in copied_paths.items()}
    missing_counts = {class_name: source_counts[class_name] - copied_counts[class_name] for class_name in FINAL_CLASSES}
    mismatches = {
        class_name: {"expected": EXPECTED_COUNTS[class_name], "source": source_counts[class_name], "copied": copied_counts[class_name]}
        for class_name in FINAL_CLASSES
        if source_counts[class_name] != EXPECTED_COUNTS[class_name] or copied_counts[class_name] != EXPECTED_COUNTS[class_name]
    }
    unexpected_folders = sorted(path.name for path in OUTPUT_DATASET.iterdir() if path.is_dir() and path.name not in FINAL_CLASSES)
    rows = [{"class_name": class_name, "source_count": source_counts[class_name], "copied_count": copied_counts[class_name], "missing_count": missing_counts[class_name]} for class_name in FINAL_CLASSES]
    summary = {
        "final_classes": FINAL_CLASSES,
        "source_dataset": str(source),
        "output_dataset": str(OUTPUT_DATASET),
        "total_images": sum(copied_counts.values()),
        "count_per_class": copied_counts,
        "expected_counts": EXPECTED_COUNTS,
        "source_counts": source_counts,
        "excluded_classes": EXCLUDED_CLASSES,
        "selection_reason": "Focused first experiment using the nine strongest remaining classes by final usable image count; smaller classes were excluded, and Satin has zero usable images after conflict handling.",
        "count_mismatches": mismatches,
        "unexpected_output_folders": unexpected_folders,
        "all_expected_counts_match": not mismatches and not unexpected_folders and sum(copied_counts.values()) == 4932,
        "verification": {"readability_checked": True, "sha256_source_destination_checked": True},
        "additional_duplicate_removal": False,
        "original_dataset_modified": False,
        "sixteen_class_dataset_modified": False,
        "splits_created": False,
        "training_performed": False,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    write_csv(REPORTS / "fabric_9class_summary.csv", SUMMARY_FIELDS, rows)
    (REPORTS / "fabric_9class_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    markdown = [
        "# Fabric 9-Class Dataset", "", "The original iBUG dataset remains untouched. The cleaned 16-class dataset also remains untouched; this dataset was copied from it into a separate derived directory.",
        "", "## Selection", "", "This is a focused first experiment using the nine classes with the strongest remaining final counts: Cotton, Polyester, Denim, Wool, Nylon, Viscose, Silk, Fleece, and Terrycloth. Acrylic, Chenille, Corduroy, Crepe, Linen, Satin, and Velvet were excluded because their final usable counts were too small for the first robust experiment. Satin was specifically excluded because it has zero usable images after cross-class conflict handling.",
        "", "## Copy and Verification", "", "Images were copied without additional duplicate removal. Every copied file was checked for readability and compared with its source using SHA-256. No train/validation/test folders were created and no model was trained.",
        "", "## Final Counts", "", "| Class | Source | Copied | Missing |", "|---|---:|---:|---:|",
    ]
    markdown.extend(f"| {row['class_name']} | {row['source_count']} | {row['copied_count']} | {row['missing_count']} |" for row in rows)
    markdown.extend(["", f"- Total images: **{summary['total_images']}**", f"- Expected total: **4932**", f"- Count validation: **{'PASS' if summary['all_expected_counts_match'] else 'MISMATCH'}**"])
    if mismatches:
        markdown.extend(["", "### Count Mismatches", "", *[f"- {class_name}: expected {values['expected']}, source {values['source']}, copied {values['copied']}" for class_name, values in mismatches.items()]])
    (REPORTS / "fabric_9class_cleaning.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print("# FINAL 9-CLASS DATASET")
    print("Class Source Copied")
    for row in rows:
        print(f"{row['class_name']} {row['source_count']} {row['copied_count']}")
    print(f"\nTotal images: {summary['total_images']}")
    print(f"Expected total: 4932")
    print(f"Count validation: {'PASS' if summary['all_expected_counts_match'] else 'MISMATCH'}")
    return 0 if summary["all_expected_counts_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())