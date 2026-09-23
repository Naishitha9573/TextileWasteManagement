from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat, UnidentifiedImageError


ROOT = Path(__file__).resolve().parent
PROCESSED_ROOT = ROOT / "Backend" / "data" / "processed" / "material_deepfashion"
SPLIT_ROOT = ROOT / "Backend" / "data" / "splits" / "material_deepfashion"
MANIFEST_PATH = ROOT / "Backend" / "data" / "manifests" / "material_deepfashion_manifest.json"
OUTPUT_JSON = ROOT / "Backend" / "data" / "metadata" / "material_dataset_validation.json"
OUTPUT_REPORT = ROOT / "MATERIAL_DATASET_VALIDATION_REPORT.md"

CLASSES = ("Cotton", "Denim", "Mixed Fabrics")
MATERIAL_KEYWORDS = {
    "cotton": "Cotton",
    "denim": "Denim",
    "polyester": "Polyester",
    "wool": "Wool",
    "silk": "Silk",
    "linen": "Linen",
    "nylon": "Nylon",
    "polyamide": "Nylon",
    "rayon": "Rayon",
    "viscose": "Rayon",
    "acrylic": "Acrylic",
}
GARMENT_WORDS = re.compile(
    r"\b(shirt|tee|t-shirt|tank|sweater|dress|pants|shorts|skirt|jacket|coat|hoodie|"
    r"cardigan|blouse|romper|jumpsuit|vest|trouser|top|clothing|garment|fabric)\b",
    re.IGNORECASE,
)
ITEM_RE = re.compile(r"^(.*-id_\d+-\d+)(?:_\d+)?(?:_[^.]*)?$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(path: Path, size: int = 16) -> str:
    with Image.open(path) as image:
        gray = image.convert("L").resize((size + 1, size))
        pixels = list(gray.getdata())
    return "".join("1" if pixels[row * (size + 1) + col] > pixels[row * (size + 1) + col + 1] else "0"
                   for row in range(size) for col in range(size))


def hamming(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right))


def item_key(path: Path) -> str:
    match = ITEM_RE.match(path.stem)
    return match.group(1) if match else path.stem


def material_mentions(caption: str) -> set[str]:
    lowered = caption.lower()
    return {target for keyword, target in MATERIAL_KEYWORDS.items() if re.search(r"\b" + re.escape(keyword) + r"\b", lowered)}


def confidence(label: str, mentions: set[str], usable: bool) -> str:
    if not usable or label not in mentions and label != "Mixed Fabrics":
        return "LOW"
    if label == "Mixed Fabrics":
        return "HIGH" if len(mentions) >= 2 else "LOW"
    return "HIGH" if mentions == {label} else "MEDIUM"


def group_paths(paths: list[Path]) -> list[list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        groups[sha256(path)].append(str(path.relative_to(ROOT)))
    return [group for group in groups.values() if len(group) > 1]


def near_groups(records: list[dict[str, Any]], threshold: int = 8) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    assigned: set[int] = set()
    for index, record in enumerate(records):
        if index in assigned or not record.get("usable"):
            continue
        members = [index]
        for other in range(index + 1, len(records)):
            if other not in assigned and records[other].get("usable") and hamming(record["dhash"], records[other]["dhash"]) <= threshold:
                members.append(other)
        if len(members) > 1:
            assigned.update(members)
            groups.append({"threshold_bits": threshold, "images": [records[i]["path"] for i in members]})
    return groups


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest_by_path = {Path(entry["image_path"]).resolve(): entry for entry in manifest["entries"]}
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    dimensions: Counter[str] = Counter()
    channels: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    caption_support: Counter[str] = Counter()
    confidence_counts: dict[str, Counter[str]] = {label: Counter() for label in CLASSES}
    split_records: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for path in sorted(PROCESSED_ROOT.glob("*/*")):
        if not path.is_file():
            continue
        label = path.parent.name
        entry = manifest_by_path.get(path.resolve(), {})
        caption = entry.get("original_label", "")
        mentions = material_mentions(caption)
        record: dict[str, Any] = {
            "path": str(path.relative_to(ROOT)),
            "label": label,
            "caption_material_mentions": sorted(mentions),
            "caption_supports_label": label in mentions if label != "Mixed Fabrics" else len(mentions) >= 2,
            "usable": False,
        }
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
                mode = image.mode
                channels_count = len(image.getbands())
                stat = ImageStat.Stat(image.convert("RGB"))
                low_variance = max(stat.var) < 2.0
                record.update({"width": width, "height": height, "mode": mode, "channels": channels_count,
                               "format": image.format, "low_variance": low_variance,
                               "usable": width >= 32 and height >= 32 and not low_variance,
                               "sha256": sha256(path), "dhash": dhash(path)})
                dimensions[f"{width}x{height}"] += 1
                channels[str(channels_count)] += 1
        except (OSError, SyntaxError, ValueError, UnidentifiedImageError) as exc:
            record["error"] = str(exc)
            errors.append({"path": record["path"], "error": str(exc)})
        record["garment_text_evidence"] = bool(GARMENT_WORDS.search(caption) or GARMENT_WORDS.search(path.name))
        record["likely_textile_or_garment"] = bool(record["usable"] and record["garment_text_evidence"])
        record["caption_label_confidence"] = confidence(label, mentions, bool(record["usable"]))
        records.append(record)
        class_counts[label] += 1
        if record["caption_supports_label"]:
            caption_support[label] += 1
        confidence_counts[label][record["caption_label_confidence"]] += 1

    usable = [record for record in records if record["usable"]]
    exact_groups = []
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in usable:
        by_hash[record["sha256"]].append(record)
    for members in by_hash.values():
        if len(members) > 1:
            exact_groups.append([record["path"] for record in members])
    near = near_groups(records)

    for split in ("train", "validation", "test"):
        for path in sorted((SPLIT_ROOT / split).glob("*/*")):
            if path.is_file():
                split_records[split].append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "item_key": item_key(path)})
    split_hashes = defaultdict(list)
    split_items = defaultdict(list)
    for split, entries in split_records.items():
        for entry in entries:
            split_hashes[entry["sha256"]].append({"split": split, "path": entry["path"]})
            split_items[entry["item_key"]].append({"split": split, "path": entry["path"]})
    cross_split_exact = [members for members in split_hashes.values() if len({member["split"] for member in members}) > 1]
    cross_split_items = [members for members in split_items.values() if len({member["split"] for member in members}) > 1]

    likely_mislabeled = [record for record in records if not record["caption_supports_label"] or not record["likely_textile_or_garment"]]
    suspicious = [record for record in records if record.get("error") or record.get("low_variance") or not record["garment_text_evidence"] or not record["caption_supports_label"]]
    result = {
        "audit_type": "second_stage_prepared_dataset_validation",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(PROCESSED_ROOT.relative_to(ROOT)),
        "source_manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        "policy": {"read_only": True, "labels_changed": False, "images_deleted_or_moved": False, "model_changed": False,
                   "near_duplicate_method": "16x16 dHash; Hamming distance <= 8 is flagged", "caption_confidence_is_not_fabric_composition": True},
        "totals": {"images": len(records), "usable_images": len(usable), "usable_percentage": round(len(usable) / len(records) * 100, 2) if records else 0,
                   "errors": len(errors), "suspicious_images": len(suspicious), "likely_mislabeled": len(likely_mislabeled)},
        "class_balance": {label: {"count": class_counts[label], "percentage": round(class_counts[label] / len(records) * 100, 2),
                                   "usable": sum(r["usable"] for r in records if r["label"] == label),
                                   "caption_supported": caption_support[label], "caption_confidence": dict(confidence_counts[label])} for label in CLASSES},
        "image_properties": {"dimensions": dict(dimensions), "channels": dict(channels)},
        "duplicates": {"exact_groups": exact_groups, "exact_duplicate_extra_images": sum(len(group) - 1 for group in exact_groups),
                        "near_duplicate_groups": near, "near_duplicate_images_in_groups": sum(len(group["images"]) - 1 for group in near)},
        "split_leakage": {"split_counts": {split: len(entries) for split, entries in split_records.items()},
                          "exact_hash_cross_split_groups": cross_split_exact, "exact_hash_cross_split": bool(cross_split_exact),
                          "same_garment_item_cross_split_groups": cross_split_items, "same_garment_item_cross_split": bool(cross_split_items)},
        "records": records,
        "suspicious_images": suspicious,
        "errors": errors,
        "limitations": ["Caption support is not proof of fiber composition.", "Garment presence is assessed from caption/filename evidence plus image sanity checks; no human or pretrained object detector was used.", "Near-duplicate results depend on the stated dHash threshold and are review flags, not automatic removals."],
    }
    OUTPUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    leakage = "YES" if result["split_leakage"]["same_garment_item_cross_split"] or result["split_leakage"]["exact_hash_cross_split"] else "NO detected"
    lines = ["# Material Dataset Second-Stage Validation", "", f"Audit timestamp: `{result['audited_at']}`", "", "## Executive Answers", "",
             f"1. **Usable images:** {len(usable)} of {len(records)} ({result['totals']['usable_percentage']}%).", f"2. **Likely mislabeled:** {len(likely_mislabeled)} review candidates; this is an evidence flag, not an automatic relabel.",
             f"3. **Duplicates/near-duplicates:** {result['duplicates']['exact_duplicate_extra_images']} exact duplicate extras in {len(exact_groups)} groups; {result['duplicates']['near_duplicate_images_in_groups']} near-duplicate extras in {len(near)} groups.",
             f"4. **Train/validation/test leakage:** {leakage} ({len(cross_split_exact)} exact-hash groups; {len(cross_split_items)} same-garment identifier groups across splits).",
             f"5. **High-confidence Cotton:** {confidence_counts['Cotton']['HIGH']}.", f"6. **High-confidence Denim:** {confidence_counts['Denim']['HIGH']}.", f"7. **High-confidence Mixed Fabrics:** {confidence_counts['Mixed Fabrics']['HIGH']}.",
             "8. **Enough for a prototype?** Quantity is adequate for a 3-class prototype, but not for production or the 10-class taxonomy because labels are caption-derived and only three classes are present.",
             "9. **Expected noisy-label impact:** label noise can reduce validation accuracy, blur class boundaries, and cause the model to learn garment/category or caption-selection artifacts instead of material properties.",
             "10. **Retrain MobileNetV3-Small now?** **No.** Review leakage and suspicious/mislabeled samples and obtain stronger labels before retraining.", "", "## Audit Results", "",
             f"- Processed files inspected: **{len(records)}**", f"- Read/open failures: **{len(errors)}**", f"- Suspicious image records: **{len(suspicious)}**", f"- Caption-supported assigned labels: **{sum(caption_support.values())}**", "",
             "### Class Balance", "", "| Class | Count | Share | Usable | Caption-supported | HIGH | MEDIUM | LOW |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label in CLASSES:
        counts = confidence_counts[label]
        lines.append(f"| {label} | {class_counts[label]} | {class_counts[label] / len(records) * 100:.2f}% | {sum(r['usable'] for r in records if r['label'] == label)} | {caption_support[label]} | {counts['HIGH']} | {counts['MEDIUM']} | {counts['LOW']} |")
    lines += ["", "### Methods and Interpretation", "", "- Every canonical processed image was opened, verified, dimensioned, channel-counted, hashed, and checked for near-uniform content.", "- Exact duplicates use SHA-256. Near-duplicates use 16x16 dHash with Hamming distance <= 8.", "- Split leakage is checked by exact hashes and the DeepFashion item identifier parsed from filenames. The split directory contains copies of the processed images; those source-copy matches are expected and are not counted as leakage by themselves.", "- Caption confidence is evidence confidence only: HIGH means the assigned class is supported by unambiguous caption material evidence (or at least two material mentions for Mixed Fabrics). It does not verify fiber composition.", "- Garment/textile presence is a conservative automated screen based on readable, non-uniform images and garment terms in captions/filenames. It is not a substitute for human visual review or an object detector.", "", "### Suspicious Image Count by Reason", ""]
    reasons = Counter()
    for record in suspicious:
        if record.get("error"): reasons["read/open error"] += 1
        if record.get("low_variance"): reasons["near-uniform image"] += 1
        if not record.get("garment_text_evidence"): reasons["no garment term in metadata/filename"] += 1
        if not record.get("caption_supports_label"): reasons["caption does not support assigned label"] += 1
    lines.extend(f"- {reason}: **{count}**" for reason, count in sorted(reasons.items()))
    lines += ["", "Generated from actual files and the prepared manifest. No labels, images, splits, model, frontend, backend, or database were modified.", ""]
    OUTPUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"images": len(records), "usable": len(usable), "mislabeled": len(likely_mislabeled), "exact_groups": len(exact_groups), "near_groups": len(near), "cross_split_item_groups": len(cross_split_items), "outputs": [str(OUTPUT_JSON), str(OUTPUT_REPORT)]}, indent=2))


if __name__ == "__main__":
    main()