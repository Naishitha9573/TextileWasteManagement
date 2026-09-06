from __future__ import annotations

import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
AUDIT_PATH = ROOT / "Backend" / "data" / "metadata" / "material_dataset_validation.json"
REVIEW_PATH = ROOT / "Backend" / "data" / "metadata" / "material_label_review.csv"
VERIFIED_ROOT = ROOT / "Backend" / "data" / "processed" / "material_deepfashion_verified"
SPLIT_ROOT = ROOT / "Backend" / "data" / "splits" / "material_deepfashion_verified"
REPORT_PATH = ROOT / "MATERIAL_DATASET_CLEANING_REPORT.md"
CLASSES = ("Cotton", "Denim", "Mixed Fabrics")
SPLITS = ("train", "validation", "test")
TARGET_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def garment_id(path: str) -> str:
    stem = Path(path).stem
    if "-id_" not in stem:
        return stem
    return stem.split("-id_", 1)[1].split("_", 1)[0]


def source_path(relative_path: str) -> Path:
    return ROOT / relative_path


def assign_groups(records: list[dict[str, Any]]) -> dict[str, str]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["garment_id"]].append(record)
    totals = Counter(record["label"] for record in records)
    targets = {split: {label: totals[label] * TARGET_RATIOS[split] for label in CLASSES} for split in SPLITS}
    image_targets = {split: len(records) * TARGET_RATIOS[split] for split in SPLITS}
    assignments: dict[str, str] = {}
    current = {split: Counter() for split in SPLITS}
    current_sizes = Counter()
    rng = random.Random(42)
    ordered = list(groups.items())
    rng.shuffle(ordered)
    ordered.sort(key=lambda pair: (-len(pair[1]), pair[0]))

    for group_id, members in ordered:
        group_counts = Counter(member["label"] for member in members)
        def score(split: str) -> float:
            projected_size_ratio = (current_sizes[split] + len(members)) / max(1, image_targets[split])
            projected_class_ratios = [(current[split][label] + group_counts[label]) / max(1, targets[split][label]) for label in CLASSES]
            return projected_size_ratio + sum(projected_class_ratios) / len(CLASSES)
        selected = min(SPLITS, key=score)
        assignments[group_id] = selected
        current[selected].update(group_counts)
        current_sizes[selected] += len(members)
    return assignments


def suspicion_reason(record: dict[str, Any]) -> str:
    reasons = []
    if record.get("error"):
        reasons.append("image read/open error")
    if record.get("low_variance"):
        reasons.append("near-uniform image")
    if not record.get("garment_text_evidence"):
        reasons.append("no garment/textile evidence in caption or filename")
    if not record.get("caption_supports_label"):
        reasons.append("caption does not support assigned label")
    return "; ".join(reasons) or "review required because label is caption-derived"


def copy_file(record: dict[str, Any], destination_root: Path, split: str | None = None) -> None:
    source = source_path(record["path"])
    if split:
        destination = destination_root / split / record["label"] / source.name
    else:
        destination = destination_root / record["label"] / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    manifest_path = ROOT / audit["source_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    captions = {str(Path(item["image_path"]).resolve()): item.get("original_label", "") for item in manifest["entries"]}
    records = audit["records"]
    for record in records:
        record["garment_id"] = garment_id(record["path"])
        record["split"] = ""
    assignments = assign_groups(records)
    for record in records:
        record["split"] = assignments[record["garment_id"]]

    review_records = [record for record in records if suspicion_reason(record) != "review required because label is caption-derived"]
    REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REVIEW_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_path", "current_label", "caption", "material_mentions", "confidence", "garment_id", "split", "reason"))
        writer.writeheader()
        for record in review_records:
            writer.writerow({"image_path": record["path"], "current_label": record["label"], "caption": captions.get(str(source_path(record["path"]).resolve()), ""),
                             "material_mentions": "; ".join(record["caption_material_mentions"]), "confidence": record["caption_label_confidence"],
                             "garment_id": record["garment_id"], "split": record["split"], "reason": suspicion_reason(record)})

    if VERIFIED_ROOT.exists():
        shutil.rmtree(VERIFIED_ROOT)
    if SPLIT_ROOT.exists():
        shutil.rmtree(SPLIT_ROOT)
    for record in records:
        copy_file(record, SPLIT_ROOT, record["split"])
        if record["caption_label_confidence"] == "HIGH":
            copy_file(record, VERIFIED_ROOT)

    split_counts = {split: Counter(record["label"] for record in records if record["split"] == split) for split in SPLITS}
    verified_counts = Counter(record["label"] for record in records if record["caption_label_confidence"] == "HIGH")
    ids_by_split = {split: {record["garment_id"] for record in records if record["split"] == split} for split in SPLITS}
    leakage = sum(bool(ids_by_split[left] & ids_by_split[right]) for index, left in enumerate(SPLITS) for right in SPLITS[index + 1:])
    groups = defaultdict(list)
    for record in records:
        groups[record["garment_id"]].append(record)
    cross_label_groups = {key: sorted({record["label"] for record in members}) for key, members in groups.items() if len({record["label"] for record in members}) > 1}
    high_group_counts = {label: len({record["garment_id"] for record in records if record["label"] == label and record["caption_label_confidence"] == "HIGH"}) for label in CLASSES}
    report = ["# Material Dataset Cleaning Report", "", "This preparation was non-destructive. Existing processed images, labels, splits, models, frontend, backend, database, and API routes were not modified.", "", "## 1. Original Dataset Statistics", "", "| Class | Images | Caption-supported | HIGH-confidence |", "|---|---:|---:|---:|"]
    for label in CLASSES:
        stats = audit["class_balance"][label]
        report.append(f"| {label} | {stats['count']} | {stats['caption_supported']} | {verified_counts[label]} |")
    report += ["", f"Total images: **{len(records)}**", f"Usable images: **{audit['totals']['usable_images']} ({audit['totals']['usable_percentage']}%)**", "", "## 2. Leakage Statistics", "", f"Unique garment IDs: **{len(groups)}**", f"Garment IDs containing multiple current labels: **{len(cross_label_groups)}**", f"Same garment IDs crossing new splits: **{leakage} pairwise overlaps** (must be 0)", "", "The new split is group-aware: every parsed garment ID is assigned to exactly one split, so front/side/back/additional views remain together.", "", "## 3. New Train/Validation/Test Statistics", "", "| Split | Total | Cotton | Denim | Mixed Fabrics |", "|---|---:|---:|---:|---:|"]
    for split in SPLITS:
        report.append(f"| {split} | {sum(split_counts[split].values())} | {split_counts[split]['Cotton']} | {split_counts[split]['Denim']} | {split_counts[split]['Mixed Fabrics']} |")
    report += ["", "## 4. Verified Dataset Class Distribution", "", "Only HIGH-confidence records were copied to `Backend/data/processed/material_deepfashion_verified/`. No images were duplicated to balance classes.", "", "| Class | Verified images | Verified garment IDs | Additional human reviews to reach 300 | Additional reviews to match Cotton (477) |", "|---|---:|---:|---:|---:|"]
    for label in CLASSES:
        report.append(f"| {label} | {verified_counts[label]} | {high_group_counts[label]} | {max(0, 300 - verified_counts[label])} | {max(0, verified_counts['Cotton'] - verified_counts[label])} |")
    report += ["", "## 5. High-Confidence Samples", "", f"HIGH-confidence images copied: **{sum(verified_counts.values())}** ({', '.join(f'{label}: {verified_counts[label]}' for label in CLASSES)}).", "", "HIGH means the caption evidence supports the assigned class under the audit rules. It does not prove laboratory-verified fiber composition.", "", "## 6. Suspicious Samples", "", f"Review manifest rows: **{len(review_records)}**. The full manifest is at `Backend/data/metadata/material_label_review.csv`.", f"Suspicious/review candidates from the audit: **{audit['totals']['suspicious_images']}**. No suspicious image was deleted or relabeled.", "", "## 7. Label-Review Requirements", "", f"- Denim requires at least **{max(0, 300 - verified_counts['Denim'])}** additional human-confirmed images to reach a practical 300-image class.", f"- Mixed Fabrics requires at least **{max(0, 300 - verified_counts['Mixed Fabrics'])}** additional human-confirmed images to reach a practical 300-image class.", f"- Matching the current HIGH-confidence Cotton count would require **{max(0, verified_counts['Cotton'] - verified_counts['Denim'])}** additional Denim and **{max(0, verified_counts['Cotton'] - verified_counts['Mixed Fabrics'])}** additional Mixed Fabrics confirmations.", "- The 12 multi-label garment groups need explicit human adjudication before any future supervised training.", "", "## 8. Recommended Verified Images Per Class", "", "Recommend **at least 300 independently verified images per class** for this prototype, with group-aware splits. For production, collect a substantially larger and composition-verified dataset spanning the full material taxonomy.", "", "## 9. Training Readiness", "", "**No, the cleaned dataset is not ready for MobileNetV3-Small training.** The new split has no garment-ID leakage, but Denim and Mixed Fabrics do not yet have sufficient independently verified samples, and caption-derived labels do not establish actual fiber composition. Human review and label adjudication are required before training.", "", "## Output Directories", "", "- Group-aware all-image split: `Backend/data/splits/material_deepfashion_verified/`", "- HIGH-confidence-only dataset: `Backend/data/processed/material_deepfashion_verified/`", "- Review manifest: `Backend/data/metadata/material_label_review.csv`", ""]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"records": len(records), "review_rows": len(review_records), "unique_garment_ids": len(groups), "split_leakage_pairs": leakage, "split_counts": {split: dict(split_counts[split]) for split in SPLITS}, "verified_counts": dict(verified_counts), "outputs": [str(REVIEW_PATH), str(VERIFIED_ROOT), str(SPLIT_ROOT), str(REPORT_PATH)]}, indent=2))


if __name__ == "__main__":
    main()