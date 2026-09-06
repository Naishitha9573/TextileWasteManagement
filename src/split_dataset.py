"""Create and verify a leakage-safe split of the cleaned 9-class dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "dataset" / "processed" / "fabric_9class" / "all"
OUTPUT_ROOT = ROOT / "dataset" / "processed" / "fabric_9class"
REPORTS = ROOT / "reports"
SPLITS = ("train", "validation", "test")
RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
SEED = 42
CLASSES = ["Cotton", "Polyester", "Denim", "Wool", "Nylon", "Viscose", "Silk", "Fleece", "Terrycloth"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GROUP_FIELDS = ["sample_id", "class_name", "image_count", "file_paths", "cross_class_conflict"]
SPLIT_FIELDS = ["sample_id", "class_name", "split", "image_count"]
SUMMARY_FIELDS = ["class_name", "total", "train", "validation", "test", "train_percentage", "validation_percentage", "test_percentage"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a deterministic, sample-grouped dataset split.")
    parser.add_argument("source_path", nargs="?", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def image_paths(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in EXTENSIONS)


def sample_id_from_path(relative_path: str) -> str | None:
    parts = Path(relative_path.replace("\\", "/")).parts
    return parts[1] if len(parts) >= 3 else None


def verify_readable(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, SyntaxError, UnidentifiedImageError, ValueError) as exc:
        raise RuntimeError(f"Unreadable image: {path}: {exc}") from exc


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def main() -> int:
    args = parse_args()
    source = args.source_path.expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"9-class source dataset does not exist: {source}")
    missing_source_classes = [class_name for class_name in CLASSES if not (source / class_name).is_dir()]
    if missing_source_classes:
        raise SystemExit(f"Missing source class folders: {', '.join(missing_source_classes)}")

    grouped: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    source_records: list[tuple[str, str, Path]] = []
    for class_name in CLASSES:
        for path in image_paths(source / class_name):
            relative = path.relative_to(source).as_posix()
            sample_id = sample_id_from_path(relative)
            if sample_id is None:
                raise SystemExit(f"Unable to extract sample ID from path: {relative}")
            grouped[sample_id][class_name].append(path)
            source_records.append((sample_id, class_name, path))

    group_rows = []
    for sample_id in sorted(grouped):
        classes = sorted(grouped[sample_id])
        conflict = len(classes) > 1
        for class_name in classes:
            paths = sorted(grouped[sample_id][class_name])
            group_rows.append({"sample_id": sample_id, "class_name": class_name, "image_count": len(paths), "file_paths": json.dumps([path.relative_to(source).as_posix() for path in paths]), "cross_class_conflict": conflict})

    rng = random.Random(args.seed)
    sample_groups = list(grouped.items())
    rng.shuffle(sample_groups)
    sample_groups.sort(key=lambda item: -sum(len(paths) for paths in item[1].values()))
    targets = {class_name: {split: sum(len(paths) for paths in grouped_sample.values()) * RATIOS[split] for split in SPLITS} for class_name, grouped_sample in ((class_name, {sample_id: grouped[sample_id].get(class_name, []) for sample_id in grouped}) for class_name in CLASSES)}
    current = {class_name: {split: 0 for split in SPLITS} for class_name in CLASSES}
    assignments: dict[str, str] = {}
    for sample_id, class_groups in sample_groups:
        choices = []
        for split in SPLITS:
            projected_score = 0.0
            for class_name in CLASSES:
                for candidate_split in SPLITS:
                    projected = current[class_name][candidate_split]
                    if class_name in class_groups and candidate_split == split:
                        projected += len(class_groups[class_name])
                    projected_score += abs(projected - targets[class_name][candidate_split])
            choices.append((projected_score, sum(current[class_name][split] for class_name in CLASSES), SPLITS.index(split), split))
        assignments[sample_id] = min(choices)[3]
        for class_name, paths in class_groups.items():
            current[class_name][assignments[sample_id]] += len(paths)

    for split in SPLITS:
        split_root = OUTPUT_ROOT / split
        if split_root.exists():
            shutil.rmtree(split_root)
        for class_name in CLASSES:
            (split_root / class_name).mkdir(parents=True, exist_ok=True)

    split_rows = []
    for sample_id in sorted(grouped):
        split = assignments[sample_id]
        for class_name in sorted(grouped[sample_id]):
            paths = grouped[sample_id][class_name]
            split_rows.append({"sample_id": sample_id, "class_name": class_name, "split": split, "image_count": len(paths)})
            for path in paths:
                destination = OUTPUT_ROOT / split / class_name / path.relative_to(source / class_name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                verify_readable(destination)
                if file_hash(path) != file_hash(destination):
                    raise RuntimeError(f"Copied image hash mismatch: {path}")

    split_counts = {split: {class_name: len(image_paths(OUTPUT_ROOT / split / class_name)) for class_name in CLASSES} for split in SPLITS}
    all_source_hashes: dict[str, set[str]] = {split: set() for split in SPLITS}
    duplicate_hashes: set[str] = set()
    for split in SPLITS:
        for class_name in CLASSES:
            for path in image_paths(OUTPUT_ROOT / split / class_name):
                digest = file_hash(path)
                if digest in set().union(*(all_source_hashes[other] for other in SPLITS if other != split)):
                    duplicate_hashes.add(digest)
                all_source_hashes[split].add(digest)
                verify_readable(path)
    sample_splits: dict[str, set[str]] = defaultdict(set)
    for row in split_rows:
        sample_splits[row["sample_id"]].add(row["split"])
    leaked_samples = {sample_id: sorted(splits) for sample_id, splits in sample_splits.items() if len(splits) > 1}
    conflict_samples = sorted(sample_id for sample_id, classes in grouped.items() if len(classes) > 1)
    conflict_leaks = {sample_id: leaked_samples[sample_id] for sample_id in conflict_samples if sample_id in leaked_samples}
    missing_classes = {split: [class_name for class_name in CLASSES if not (OUTPUT_ROOT / split / class_name).is_dir()] for split in SPLITS}
    source_total = len(source_records)
    copied_total = sum(sum(counts.values()) for counts in split_counts.values())
    total_check = copied_total == source_total
    image_check = not duplicate_hashes and not any(missing_classes.values()) and not leaked_samples
    summary_rows = []
    for class_name in CLASSES:
        total = sum(split_counts[split][class_name] for split in SPLITS)
        summary_rows.append({"class_name": class_name, "total": total, **{split: split_counts[split][class_name] for split in SPLITS}, "train_percentage": round(split_counts["train"][class_name] / total * 100, 2) if total else 0, "validation_percentage": round(split_counts["validation"][class_name] / total * 100, 2) if total else 0, "test_percentage": round(split_counts["test"][class_name] / total * 100, 2) if total else 0})
    summary = {
        "random_seed": args.seed, "grouping_method": "global sample_id; all views remain together", "split_ratios": RATIOS,
        "source_dataset": str(source), "output_dataset": str(OUTPUT_ROOT), "total_images": source_total,
        "total_train_images": sum(split_counts["train"].values()), "total_validation_images": sum(split_counts["validation"].values()), "total_test_images": sum(split_counts["test"].values()),
        "per_class_counts": {row["class_name"]: {split: row[split] for split in SPLITS} for row in summary_rows}, "excluded_conflicting_samples": [],
        "detected_cross_class_sample_conflicts": conflict_samples, "leakage_checks": {"duplicate_across_splits": not duplicate_hashes, "sample_groups_across_splits": not leaked_samples, "missing_classes": not any(missing_classes.values()), "total_accounted_for": total_check, "image_verification": image_check, "cross_class_conflicts_not_split": not conflict_leaks},
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    write_csv(REPORTS / "fabric_9class_sample_groups.csv", GROUP_FIELDS, group_rows)
    write_csv(REPORTS / "fabric_9class_split.csv", SPLIT_FIELDS, split_rows)
    write_csv(REPORTS / "fabric_9class_split_summary.csv", SUMMARY_FIELDS, summary_rows)
    (REPORTS / "fabric_9class_split_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    checks = summary["leakage_checks"]
    markdown = ["# Data Leakage Check", "", f"- Source images: **{source_total}**", f"- Train: **{summary['total_train_images']}**", f"- Validation: **{summary['total_validation_images']}**", f"- Test: **{summary['total_test_images']}**", f"- Cross-class sample conflicts detected: **{len(conflict_samples)}**", "", "## Automated Checks", "", f"- Duplicate across splits: **{'PASS' if checks['duplicate_across_splits'] else 'FAIL'}**", f"- Sample groups across splits: **{'PASS' if checks['sample_groups_across_splits'] else 'FAIL'}**", f"- Missing classes: **{'PASS' if checks['missing_classes'] else 'FAIL'}**", f"- Total accounting: **{'PASS' if checks['total_accounted_for'] else 'FAIL'}**", f"- Image verification: **{'PASS' if checks['image_verification'] else 'FAIL'}**", f"- Cross-class conflicts kept together: **{'PASS' if checks['cross_class_conflicts_not_split'] else 'FAIL'}**", "", "All source images were assigned to exactly one split. No source image was modified, and no train/validation/test split was made from the original iBUG tree."]
    (REPORTS / "data_leakage_check.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print("# DATASET SPLIT COMPLETE")
    print(f"Total images: {source_total}")
    for split in SPLITS:
        print(f"{split.title()}: {sum(split_counts[split].values())} ({sum(split_counts[split].values()) / source_total * 100:.2f}%)")
    print("\nPer-class distribution:\nClass Train Val Test")
    for row in summary_rows:
        print(f"{row['class_name']} {row['train']} {row['validation']} {row['test']}")
    print("\nLeakage checks:")
    print(f"Duplicate across splits: {'PASS' if checks['duplicate_across_splits'] else 'FAIL'}")
    print(f"Sample groups across splits: {'PASS' if checks['sample_groups_across_splits'] else 'FAIL'}")
    print(f"Missing classes: {'PASS' if checks['missing_classes'] else 'FAIL'}")
    print(f"Image verification: {'PASS' if checks['image_verification'] else 'FAIL'}")
    if not all(checks.values()):
        raise SystemExit("Leakage validation failed; inspect reports/data_leakage_check.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())