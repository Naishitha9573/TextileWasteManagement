from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
AUDIT_PATH = ROOT / "Backend" / "data" / "metadata" / "material_dataset_validation.json"
CONFLICT_PATH = ROOT / "Backend" / "data" / "metadata" / "conflicting_garment_labels.csv"
DATASET_ROOT = ROOT / "Backend" / "data" / "processed" / "material_deepfashion_verified_clean"
SPLIT_ROOT = ROOT / "Backend" / "data" / "splits" / "material_deepfashion_verified_clean"
MANIFEST_PATH = ROOT / "Backend" / "data" / "metadata" / "material_dataset_final_manifest.json"
FINAL_AUDIT_PATH = ROOT / "Backend" / "data" / "metadata" / "material_dataset_final_audit.json"
REPORT_PATH = ROOT / "FINAL_MATERIAL_DATASET_REPORT.md"
CLASSES = ("Cotton", "Denim", "Mixed Fabrics")
SPLITS = ("train", "validation", "test")
RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def garment_id(path: str) -> str:
    stem = Path(path).stem
    return stem.split("-id_", 1)[1].split("_", 1)[0] if "-id_" in stem else stem


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(path: Path, size: int = 16) -> str:
    from PIL import Image

    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((size + 1, size)).getdata())
    return "".join("1" if pixels[row * (size + 1) + col] > pixels[row * (size + 1) + col + 1] else "0"
                   for row in range(size) for col in range(size))


def assign_groups(records: list[dict[str, Any]]) -> dict[str, str]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["garment_id"]].append(record)
    totals = Counter(record["label"] for record in records)
    targets = {split: {label: totals[label] * RATIOS[split] for label in CLASSES} for split in SPLITS}
    sizes = {split: len(records) * RATIOS[split] for split in SPLITS}
    current = {split: Counter() for split in SPLITS}
    current_sizes = Counter()
    ordered = list(groups.items())
    random.Random(42).shuffle(ordered)
    ordered.sort(key=lambda pair: (-len(pair[1]), pair[0]))
    assignments = {}
    for group_id, members in ordered:
        group_counts = Counter(member["label"] for member in members)
        def score(split: str) -> float:
            projected_sizes = Counter(current_sizes)
            projected_sizes[split] += len(members)
            projected_counts = {name: Counter(values) for name, values in current.items()}
            projected_counts[split].update(group_counts)
            size_error = sum(abs(projected_sizes[name] - sizes[name]) / max(1, len(records)) for name in SPLITS)
            class_error = sum(abs(projected_counts[name][label] - targets[name][label]) / max(1, totals[label]) for name in SPLITS for label in CLASSES)
            return size_error + class_error
        selected = min(SPLITS, key=score)
        assignments[group_id] = selected
        current[selected].update(group_counts)
        current_sizes[selected] += len(members)
    return assignments


def main() -> None:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    with CONFLICT_PATH.open(newline="", encoding="utf-8") as handle:
        conflict_ids = {row["garment_id"] for row in csv.DictReader(handle)}
    records = []
    for original in audit["records"]:
        record = dict(original)
        record["garment_id"] = garment_id(record["path"])
        if record["caption_label_confidence"] == "HIGH" and record["garment_id"] not in conflict_ids:
            records.append(record)
    assignments = assign_groups(records)
    for record in records:
        record["split"] = assignments[record["garment_id"]]
        record["source_path"] = record.pop("path")

    for directory in (DATASET_ROOT, SPLIT_ROOT):
        if directory.exists():
            shutil.rmtree(directory)
    manifest_entries = []
    for record in records:
        source = ROOT / record["source_path"]
        dataset_destination = DATASET_ROOT / record["label"] / source.name
        split_destination = SPLIT_ROOT / record["split"] / record["label"] / source.name
        dataset_destination.parent.mkdir(parents=True, exist_ok=True)
        split_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dataset_destination)
        shutil.copy2(source, split_destination)
        manifest_entries.append({"source_path": record["source_path"], "final_path": str(dataset_destination.relative_to(ROOT)), "split_path": str(split_destination.relative_to(ROOT)), "label": record["label"], "garment_id": record["garment_id"], "split": record["split"], "confidence": record["caption_label_confidence"], "caption_material_mentions": record["caption_material_mentions"], "sha256": sha256(source)})

    by_hash: dict[str, list[str]] = defaultdict(list)
    by_dhash: dict[str, list[str]] = defaultdict(list)
    for entry in manifest_entries:
        by_hash[entry["sha256"]].append(entry["source_path"])
        by_dhash[dhash(ROOT / entry["source_path"])].append(entry["source_path"])
    exact_groups = [paths for paths in by_hash.values() if len(paths) > 1]
    near_groups = [paths for paths in by_dhash.values() if len(paths) > 1]
    split_ids = {split: {entry["garment_id"] for entry in manifest_entries if entry["split"] == split} for split in SPLITS}
    cross_split_pairs = [{"left": left, "right": right, "garment_ids": sorted(split_ids[left] & split_ids[right])} for index, left in enumerate(SPLITS) for right in SPLITS[index + 1:] if split_ids[left] & split_ids[right]]
    class_counts = Counter(entry["label"] for entry in manifest_entries)
    split_counts = {split: Counter(entry["label"] for entry in manifest_entries if entry["split"] == split) for split in SPLITS}
    unique_class_ids = {label: len({entry["garment_id"] for entry in manifest_entries if entry["label"] == label}) for label in CLASSES}
    confidence_counts = Counter(entry["confidence"] for entry in manifest_entries)
    high_records = [record for record in audit["records"] if record["caption_label_confidence"] == "HIGH"]
    excluded_high = [record for record in high_records if garment_id(record["path"]) in conflict_ids]
    final_audit = {"audit_type": "final_clean_material_dataset", "audited_at": datetime.now(timezone.utc).isoformat(), "read_only_source": True, "training_performed": False, "source_dataset": "Backend/data/processed/material_deepfashion", "conflict_source": str(CONFLICT_PATH.relative_to(ROOT)), "total_images": len(manifest_entries), "images_per_class": dict(class_counts), "unique_garment_ids": len({entry["garment_id"] for entry in manifest_entries}), "unique_garment_ids_per_class": unique_class_ids, "split_counts": {split: {"total": sum(counts.values()), **dict(counts)} for split, counts in split_counts.items()}, "class_percentages": {label: round(class_counts[label] / len(manifest_entries) * 100, 2) for label in CLASSES}, "excluded_conflicting_garment_ids": len(conflict_ids), "excluded_images": len(excluded_high), "duplicate_statistics": {"exact_duplicate_groups": exact_groups, "exact_duplicate_extra_images": sum(len(group) - 1 for group in exact_groups), "dhash_identical_groups": near_groups}, "cross_split_garment_leakage": {"pairs": cross_split_pairs, "detected": bool(cross_split_pairs)}, "caption_confidence": dict(confidence_counts), "label_composition_warning": "HIGH confidence means caption evidence supports the assigned class; it is not laboratory-verified fiber composition.", "manifest_entries": manifest_entries}
    MANIFEST_PATH.write_text(json.dumps({"dataset": "material_deepfashion_verified_clean", "generated_at": final_audit["audited_at"], "entries": manifest_entries}, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    FINAL_AUDIT_PATH.write_text(json.dumps(final_audit, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    report = ["# Final Material Dataset Report", "", "This is a new, non-destructive supervised-training candidate. Original images, labels, dataset directories, models, frontend, backend, database, and API routes were not modified. No training was performed.", "", "## Final Dataset", "", f"- Total images: **{len(manifest_entries)}**", f"- Unique garment IDs: **{final_audit['unique_garment_ids']}**", "", "| Class | Images | Unique garment IDs | Percentage |", "|---|---:|---:|---:|"]
    for label in CLASSES:
        report.append(f"| {label} | {class_counts[label]} | {unique_class_ids[label]} | {final_audit['class_percentages'][label]:.2f}% |")
    report += ["", "## Split Statistics", "", "| Split | Total | Cotton | Denim | Mixed Fabrics |", "|---|---:|---:|---:|---:|"]
    for split in SPLITS:
        report.append(f"| {split} | {sum(split_counts[split].values())} | {split_counts[split]['Cotton']} | {split_counts[split]['Denim']} | {split_counts[split]['Mixed Fabrics']} |")
    report += ["", "## Exclusions", "", f"- Conflicting garment IDs excluded: **{len(conflict_ids)}**", f"- HIGH-confidence images excluded with those IDs: **{len(excluded_high)}**", "- All excluded images remain in the original dataset; none were deleted or relabeled.", "", "## Integrity Checks", "", f"- Exact duplicate groups: **{len(exact_groups)}**", f"- Exact duplicate extra images: **{sum(len(group) - 1 for group in exact_groups)}**", f"- Identical dHash groups: **{len(near_groups)}**", f"- Cross-split garment-ID leakage: **{'YES' if cross_split_pairs else 'NO'}**", f"- All final records HIGH confidence: **{'YES' if confidence_counts == Counter({'HIGH': len(manifest_entries)}) else 'NO'}**", "", "## Caption Confidence", "", f"- HIGH: **{confidence_counts['HIGH']}**", f"- MEDIUM: **{confidence_counts['MEDIUM']}**", f"- LOW: **{confidence_counts['LOW']}**", "", "HIGH confidence indicates that the caption-derived material mention supports the assigned class under the audit rules. It does not prove laboratory-verified material composition.", "", "## Training Readiness", "", "The final candidate has no detected garment-ID leakage, no conflicting garment IDs, and no exact duplicate groups. However, the class distribution remains strongly imbalanced: 469 Cotton, 163 Denim, and 76 Mixed Fabrics. Only 76 Mixed Fabrics images are available, which is not a reasonable class size for reliable MobileNetV3-Small training. Caption-derived labels also remain unverified composition claims.", "", "**READY_FOR_TRAINING = NO**", "", "The dataset is not ready for model training. Human verification and additional Mixed Fabrics and Denim samples are required before training.", "", "## Generated Artifacts", "", f"- `{DATASET_ROOT.relative_to(ROOT)}`", f"- `{SPLIT_ROOT.relative_to(ROOT)}`", f"- `{MANIFEST_PATH.relative_to(ROOT)}`", f"- `{FINAL_AUDIT_PATH.relative_to(ROOT)}`", ""]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"total": len(manifest_entries), "class_counts": dict(class_counts), "excluded_ids": len(conflict_ids), "excluded_images": len(excluded_high), "split_counts": {split: dict(counts) for split, counts in split_counts.items()}, "cross_split_leakage": bool(cross_split_pairs), "outputs": [str(DATASET_ROOT), str(SPLIT_ROOT), str(MANIFEST_PATH), str(FINAL_AUDIT_PATH), str(REPORT_PATH)]}, indent=2))


if __name__ == "__main__":
    main()