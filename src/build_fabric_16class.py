"""Build the provisional 16-class dataset without modifying the source dataset."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "Backend" / "datasets" / "ibug_fabrics"
OUTPUT_ROOT = ROOT / "dataset" / "processed" / "fabric_16class"
OUTPUT_ALL = OUTPUT_ROOT / "all"
REPORTS = ROOT / "reports"
EXACT_REPORT = REPORTS / "exact_duplicates.csv"
NEAR_REPORT = REPORTS / "near_duplicates.csv"
APPROVED_CLASSES = [
    "Acrylic", "Chenille", "Corduroy", "Cotton", "Crepe", "Denim", "Fleece", "Linen",
    "Nylon", "Polyester", "Satin", "Silk", "Terrycloth", "Velvet", "Viscose", "Wool",
]
EXCLUDED_CLASSES = [
    "Artificial_fur", "Artificial_leather", "Blended", "Felt", "Leather", "Lut", "Suede", "Unclassified", "Utilities",
]
CONFLICT_SAMPLE_IDS = {
    "998", "999", "1000", "1001", "1002", "1003", "1028", "1029", "1030", "1031", "1032", "1033",
    "770", "771", "772", "773", "1004", "1005", "1006", "1007", "1008", "1009", "1010", "1011", "1012", "1013", "1014", "1015", "1016", "1017",
}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXCLUDED_FIELDS = ["file_path", "class_name", "reason", "duplicate_group_id", "sample_id"]
CLEANING_FIELDS = ["class_name", "original_images", "same_class_exact_duplicates", "cross_class_conflict_images", "near_duplicate_images_flagged", "final_usable_images"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the provisional read-only 16-class processed dataset.")
    parser.add_argument("source_path", nargs="?", type=Path, default=DEFAULT_SOURCE)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sample_id_from_path(file_path: str) -> str | None:
    parts = Path(file_path.replace("\\", "/")).parts
    return parts[1] if len(parts) >= 3 else None


def source_images(source_path: Path) -> dict[str, list[Path]]:
    return {
        class_name: sorted(
            path for path in (source_path / class_name).rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        for class_name in APPROVED_CLASSES
    }


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def is_readable(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, SyntaxError, UnidentifiedImageError, ValueError):
        return False


def main() -> int:
    args = parse_args()
    source_path = args.source_path.expanduser().resolve()
    if not source_path.is_dir():
        raise SystemExit(f"Source dataset directory does not exist: {source_path}")
    exact_rows = read_csv(EXACT_REPORT)
    near_rows = read_csv(NEAR_REPORT)
    images_by_class = source_images(source_path)
    source_by_relative = {
        path.relative_to(source_path).as_posix(): (class_name, path)
        for class_name, paths in images_by_class.items() for path in paths
    }

    exact_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in exact_rows:
        if row.get("file_path") in source_by_relative:
            exact_groups[row["duplicate_group_id"]].append(row)
    same_class_representatives: set[str] = set()
    same_class_excluded: dict[str, tuple[str, str]] = {}
    for group_id, rows in exact_groups.items():
        classes = {row["class_name"] for row in rows}
        for class_name in classes & set(APPROVED_CLASSES):
            class_paths = sorted(row["file_path"] for row in rows if row["class_name"] == class_name)
            if len(classes) == 1 and class_paths:
                same_class_representatives.add(class_paths[0])
                for duplicate_path in class_paths[1:]:
                    same_class_excluded[duplicate_path] = ("SAME_CLASS_EXACT_DUPLICATE", group_id)

    near_by_class: Counter[str] = Counter()
    for row in near_rows:
        for file_path, class_name in ((row.get("file_path", ""), row.get("class_name", "")), (row.get("paired_file_path", ""), row.get("paired_class_name", ""))):
            if file_path in source_by_relative and class_name in APPROVED_CLASSES:
                near_by_class[file_path] += 1
    near_flagged_paths = set(near_by_class)
    excluded_rows: list[dict[str, str]] = []
    cleaning_rows: list[dict[str, Any]] = []
    final_counts: dict[str, int] = {}
    for class_name in APPROVED_CLASSES:
        original_paths = images_by_class[class_name]
        final_count = 0
        same_count = 0
        conflict_count = 0
        for path in original_paths:
            relative = path.relative_to(source_path).as_posix()
            sample_id = sample_id_from_path(relative)
            if not is_readable(path):
                continue
            exclusion: tuple[str, str] | None = same_class_excluded.get(relative)
            if sample_id in CONFLICT_SAMPLE_IDS:
                exclusion = ("CROSS_CLASS_LABEL_CONFLICT", "cross-class sample ID policy")
            if exclusion:
                reason, group_id = exclusion
                excluded_rows.append({"file_path": relative, "class_name": class_name, "reason": reason, "duplicate_group_id": group_id, "sample_id": sample_id or ""})
                if reason == "SAME_CLASS_EXACT_DUPLICATE":
                    same_count += 1
                else:
                    conflict_count += 1
                continue
            destination = OUTPUT_ALL / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            final_count += 1
        final_counts[class_name] = final_count
        cleaning_rows.append({
            "class_name": class_name,
            "original_images": len(original_paths),
            "same_class_exact_duplicates": same_count,
            "cross_class_conflict_images": conflict_count,
            "near_duplicate_images_flagged": len({path for path in near_flagged_paths if path.startswith(f"{class_name}/")}),
            "final_usable_images": final_count,
        })

    original_total = sum(row["original_images"] for row in cleaning_rows)
    summary = {
        "approved_classes": APPROVED_CLASSES,
        "excluded_classes": EXCLUDED_CLASSES,
        "source_dataset": str(source_path),
        "output_dataset": str(OUTPUT_ALL),
        "original_counts": {row["class_name"]: row["original_images"] for row in cleaning_rows},
        "removed_counts": {row["class_name"]: row["same_class_exact_duplicates"] + row["cross_class_conflict_images"] for row in cleaning_rows},
        "final_counts": final_counts,
        "total_original_images": original_total,
        "total_final_images": sum(final_counts.values()),
        "total_excluded_images": len(excluded_rows),
        "near_duplicate_images_flagged": len(near_flagged_paths),
        "cross_class_conflict_policy": "Exclude all images whose sample ID is in the confirmed conflict list; preserve originals and make no label decision.",
        "same_class_duplicate_policy": "Keep the lexicographically first file in each same-class SHA-256 group and record redundant copies; originals remain untouched.",
        "near_duplicate_policy": "Do not exclude near duplicates; flag them for later review.",
        "read_only_source": True,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    write_csv(REPORTS / "final_dataset_cleaning.csv", CLEANING_FIELDS, cleaning_rows)
    write_csv(REPORTS / "excluded_images.csv", EXCLUDED_FIELDS, excluded_rows)
    (REPORTS / "fabric_16class_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    markdown = [
        "# Fabric 16-Class Dataset Cleaning", "", "The original iBUG dataset was preserved as read-only. Images were copied into a derived dataset; no source image was deleted, moved, renamed, or modified.",
        "", "## Class Policy", "", "The provisional 16 classes focus on clearly defined fabric/textile categories for the first experiment. Artificial fur, artificial leather, blended, felt, leather, Lut, suede, unclassified, and utilities were not used because they are small, ambiguous, general-material, or unclassified categories. These classes remain in the original dataset.",
        "", "## Duplicate and Conflict Policy", "", "- Same-class exact SHA-256 duplicates keep one deterministic representative; every redundant copy is recorded in `excluded_images.csv`.",
        "- All confirmed cross-class conflict sample IDs are excluded from every approved class, including all available views. This is a provisional conflict policy, not a decision about the correct label.",
        "- Near duplicates are not removed. They are counted as review flags from `near_duplicates.csv`.",
        "", "## Final Counts", "", "| Class | Original | Same-class exact excluded | Conflict excluded | Near flagged | Final |", "|---|---:|---:|---:|---:|---:|",
    ]
    markdown.extend(f"| {row['class_name']} | {row['original_images']} | {row['same_class_exact_duplicates']} | {row['cross_class_conflict_images']} | {row['near_duplicate_images_flagged']} | {row['final_usable_images']} |" for row in cleaning_rows)
    markdown.extend(["", f"- Total original approved-class images: **{original_total}**", f"- Total final images: **{sum(final_counts.values())}**", f"- Total excluded images: **{len(excluded_rows)}**", "", "No train/validation/test folders were created. No model was trained."])
    (REPORTS / "fabric_16class_cleaning.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print("# FINAL CLEAN DATASET")
    print("Class Original Final")
    for row in cleaning_rows:
        print(f"{row['class_name']} {row['original_images']} {row['final_usable_images']}")
    print(f"\nTotal original: {original_total}")
    print(f"Total final: {sum(final_counts.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())