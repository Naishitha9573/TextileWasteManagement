from __future__ import annotations

import csv
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "Backend" / "datasets" / "ibug_fabrics"
AUDIT_PATH = ROOT / "Backend" / "data" / "metadata" / "ibug_fabrics_audit.json"
COMPOSITION_MANIFEST = ROOT / "Backend" / "data" / "metadata" / "ibug_composition_manifest.csv"
DATASET_ROOT = ROOT / "Backend" / "data" / "processed" / "ibug_material_v2"
SPLIT_ROOT = ROOT / "Backend" / "data" / "splits" / "ibug_material_v2"
AUDIT_OUTPUT = ROOT / "Backend" / "data" / "metadata" / "ibug_material_v2_audit.json"
REPORT_OUTPUT = ROOT / "IBUG_MATERIAL_V2_REPORT.md"
DUPLICATE_REPORT = ROOT / "IBUG_MATERIAL_V2_DUPLICATE_REVIEW.md"
CLASSES = ("Cotton", "Denim", "Polyester", "Wool", "Silk", "Nylon")
TARGET_ALIASES = {"Polyamide": "Nylon"}
MATERIAL_NAMES = {"Acrylic", "Cashmere", "Cotton", "Denim", "Elastane", "Linen", "Nylon", "Polyamide", "Polyester", "Rayon", "Silk", "Viscose", "Wool", "Satin", "Velvet"}
MATERIAL_PATTERN = re.compile(r"\b(" + "|".join(sorted(MATERIAL_NAMES, key=len, reverse=True)) + r")\b", re.IGNORECASE)
PERCENT_PATTERN = re.compile(r"\b(" + "|".join(sorted(MATERIAL_NAMES, key=len, reverse=True)) + r")\b\s*(?:\([^)]*\))?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def group_key(path: str) -> str:
    parts = Path(path).parts
    return f"{parts[-3]}/{parts[-2]}"


def parse_tag(tag_path: Path) -> dict[str, Any]:
    text = tag_path.read_text(encoding="utf-8", errors="replace") if tag_path.exists() else ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    garment_type = lines[0] if lines else None
    composition_text = " ".join(lines[1:]) if len(lines) > 1 else ""
    components = sorted({match.group(1).title() for match in MATERIAL_PATTERN.finditer(composition_text)})
    percentages: dict[str, float] = {}
    for match in PERCENT_PATTERN.finditer(composition_text):
        material = match.group(1).title()
        percentages[material] = percentages.get(material, 0.0) + float(match.group(2))
    return {"original_tag_text": text, "garment_type": garment_type, "composition_text": composition_text, "material_components": components, "percentages": percentages, "has_composition": bool(percentages), "tag_exists": tag_path.exists()}


def build_groups(audit: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]], dict[str, set[str]]]:
    groups: dict[str, dict[str, Any]] = {}
    hash_groups: dict[str, set[str]] = defaultdict(set)
    dhash_groups: dict[str, set[str]] = defaultdict(set)
    for record in audit["records"]:
        key = group_key(record["path"])
        group = groups.setdefault(key, {"group_key": key, "class_folder": Path(record["path"]).parts[-3], "sample_id": Path(record["path"]).parts[-2], "image_records": [], "source_dir": "/".join(Path(record["path"]).parts[:-1])})
        group["image_records"].append(record)
        hash_groups[record["sha256"]].add(key)
        dhash_groups[record["dhash"]].add(key)
    return groups, hash_groups, dhash_groups


def cross_sample_duplicate_groups(audit: dict[str, Any]) -> tuple[dict[str, set[str]], dict[str, set[str]], list[list[str]]]:
    exact_by_group: dict[str, set[str]] = defaultdict(set)
    near_by_group: dict[str, set[str]] = defaultdict(set)
    review_groups: list[list[str]] = []
    for paths in audit["duplicates"]["exact_sha256_groups"]:
        sample_groups = sorted({group_key(path) for path in paths})
        if len(sample_groups) > 1:
            review_groups.append(sample_groups)
            for group in sample_groups:
                exact_by_group[group].update(sample_groups)
    for paths in audit["duplicates"]["near_duplicate_groups"]:
        sample_groups = sorted({group_key(path) for path in paths})
        if len(sample_groups) > 1:
            review_groups.append(sample_groups)
            for group in sample_groups:
                near_by_group[group].update(sample_groups)
    return exact_by_group, near_by_group, review_groups


def classify_group(group: dict[str, Any], exact_flags: dict[str, set[str]], near_flags: dict[str, set[str]]) -> dict[str, Any]:
    folder = group["class_folder"]
    tag = parse_tag(ROOT / group["source_dir"] / "tag.txt")
    reasons: list[str] = []
    mapped_class = None
    status = "REJECTED"
    components = set(tag["material_components"])
    normalized = {TARGET_ALIASES.get(component, component) for component in components}
    percentages = tag["percentages"]
    valid_percentages = all(0 < value <= 100 for value in percentages.values())
    if not tag["tag_exists"]:
        reasons.append("MISSING_TAG")
    if not tag["has_composition"]:
        reasons.append("UNPARSEABLE_OR_MISSING_COMPOSITION")
    if len(components) > 1:
        reasons.append("MIXED_COMPOSITION_REVIEW")
    elif len(normalized) == 1 and normalized.issubset(CLASSES) and valid_percentages:
        mapped_class = next(iter(normalized))
        if len(exact_flags.get(group["group_key"], set())) > 1 or len(near_flags.get(group["group_key"], set())) > 1:
            reasons.append("CROSS_SAMPLE_DUPLICATE_REVIEW")
        elif len(group["image_records"]) != 4 or {record["view"] for record in group["image_records"]} != {"im_1", "im_2", "im_3", "im_4"}:
            reasons.append("NONSTANDARD_VIEW_GROUP")
        elif all(record.get("usable") for record in group["image_records"]):
            status = "ACCEPTED"
        else:
            reasons.append("UNUSABLE_IMAGE")
    elif tag["has_composition"] and len(components) == 1:
        reasons.append("UNSUPPORTED_MATERIAL")
    elif tag["has_composition"] and len(components) > 1:
        reasons.append("MIXED_COMPOSITION_REVIEW")
    if status != "ACCEPTED" and not reasons:
        reasons.append("REVIEW_REQUIRED")
    if "CROSS_SAMPLE_DUPLICATE_REVIEW" in reasons or "MIXED_COMPOSITION_REVIEW" in reasons or "NONSTANDARD_VIEW_GROUP" in reasons or "MISSING_TAG" in reasons or "UNPARSEABLE_OR_MISSING_COMPOSITION" in reasons:
        status = "REVIEW"
    return {**group, **tag, "mapped_class": mapped_class, "status": status, "review_reasons": sorted(set(reasons)), "exact_duplicate_groups": sorted(exact_flags.get(group["group_key"], set())), "near_duplicate_groups": sorted(near_flags.get(group["group_key"], set()))}


def assign_splits(records: list[dict[str, Any]]) -> dict[str, str]:
    groups = defaultdict(list)
    for record in records:
        groups[record["group_key"]].append(record)
    totals = Counter(record["mapped_class"] for record in records)
    targets = {split: {label: totals[label] * ratio for label in CLASSES} for split, ratio in {"train": .70, "validation": .15, "test": .15}.items()}
    sizes = {split: len(records) * ratio for split, ratio in {"train": .70, "validation": .15, "test": .15}.items()}
    current = {split: Counter() for split in ("train", "validation", "test")}
    current_sizes = Counter()
    ordered = list(groups.items())
    random.Random(42).shuffle(ordered)
    ordered.sort(key=lambda pair: (-len(pair[1]), pair[0]))
    assignments = {}
    for key, members in ordered:
        counts = Counter(member["mapped_class"] for member in members)
        def score(split: str) -> float:
            projected_sizes = Counter(current_sizes); projected_sizes[split] += len(members)
            projected = {name: Counter(values) for name, values in current.items()}; projected[split].update(counts)
            size_error = sum(abs(projected_sizes[name] - sizes[name]) / len(records) for name in current)
            class_error = sum(abs(projected[name][label] - targets[name][label]) / max(1, totals[label]) for name in current for label in CLASSES)
            return size_error + class_error
        chosen = min(current, key=score)
        assignments[key] = chosen
        current[chosen].update(counts); current_sizes[chosen] += len(members)
    return assignments


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    groups, _, _ = build_groups(audit)
    exact_flags, near_flags, duplicate_review_groups = cross_sample_duplicate_groups(audit)
    classified = [classify_group(group, exact_flags, near_flags) for group in groups.values()]
    accepted = [record for record in classified if record["status"] == "ACCEPTED"]
    assignments = assign_splits(accepted)
    for record in accepted:
        record["split"] = assignments[record["group_key"]]

    fields = ("sample_id", "original_folder", "original_tag_text", "material_components", "percentages", "garment_type", "composition_evidence", "source_path", "mapped_class", "status", "review_reasons", "image_count", "views", "exact_duplicate_groups", "near_duplicate_groups")
    with COMPOSITION_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for record in classified:
            writer.writerow({"sample_id": record["sample_id"], "original_folder": record["class_folder"], "original_tag_text": record["original_tag_text"], "material_components": "; ".join(record["material_components"]), "percentages": json.dumps(record["percentages"], ensure_ascii=True), "garment_type": record["garment_type"] or "", "composition_evidence": "manufacturer-style tag.txt" if record["has_composition"] else "missing/unparseable", "source_path": record["source_dir"], "mapped_class": record["mapped_class"] or "", "status": record["status"], "review_reasons": "; ".join(record["review_reasons"]), "image_count": len(record["image_records"]), "views": "; ".join(sorted(record["view"] for record in record["image_records"])), "exact_duplicate_groups": "; ".join(record["exact_duplicate_groups"]), "near_duplicate_groups": "; ".join(record["near_duplicate_groups"])})

    for destination in (DATASET_ROOT, SPLIT_ROOT):
        if destination.exists(): shutil.rmtree(destination)
    for label in CLASSES:
        (DATASET_ROOT / label).mkdir(parents=True, exist_ok=True)
        for split in ("train", "validation", "test"):
            (SPLIT_ROOT / split / label).mkdir(parents=True, exist_ok=True)
    manifest_entries = []
    for record in accepted:
        target_folder = DATASET_ROOT / record["mapped_class"]
        split_folder = SPLIT_ROOT / record["split"] / record["mapped_class"]
        target_folder.mkdir(parents=True, exist_ok=True); split_folder.mkdir(parents=True, exist_ok=True)
        for image in record["image_records"]:
            source = ROOT / image["path"]
            filename = f"sample-{record['class_folder']}-{record['sample_id']}-{image['view']}{source.suffix.lower()}"
            final_path = target_folder / filename; split_path = split_folder / filename
            link_or_copy(source, final_path); link_or_copy(source, split_path)
            manifest_entries.append({"sample_id": record["sample_id"], "group_key": record["group_key"], "original_folder": record["class_folder"], "source_path": image["path"], "derived_path": relative(final_path), "split_path": relative(split_path), "mapped_class": record["mapped_class"], "split": record["split"], "original_tag_text": record["original_tag_text"], "material_components": record["material_components"], "percentages": record["percentages"], "garment_type": record["garment_type"]})

    split_groups = {split: {record["group_key"] for record in accepted if record["split"] == split} for split in ("train", "validation", "test")}
    leakage = [{"left": left, "right": right, "groups": sorted(split_groups[left] & split_groups[right])} for i, left in enumerate(split_groups) for right in list(split_groups)[i + 1:] if split_groups[left] & split_groups[right]]
    class_counts = Counter(record["mapped_class"] for record in accepted)
    split_counts = {split: Counter(record["mapped_class"] for record in accepted if record["split"] == split) for split in split_groups}
    status_counts = Counter(record["status"] for record in classified)
    reason_counts = Counter(reason for record in classified for reason in record["review_reasons"])
    final_audit = {"audit_type": "ibug_material_v2_derived_dataset", "audited_at": datetime.now(timezone.utc).isoformat(), "source_dataset": relative(SOURCE_ROOT), "policy": {"source_modified": False, "source_tag_files_modified": False, "images_deleted": False, "merged_with_deepfashion": False, "training_performed": False, "derived_storage": "same-volume hard links used for derived entries because the workspace had no free disk space; source bytes were not modified", "mapping_source": "original tag.txt composition metadata only", "mixed_composition_policy": "review, not automatically assigned to pure classes", "polyamide_policy": "preserved as Polyamide and mapped to Nylon only when otherwise unambiguous"}, "taxonomy": list(CLASSES), "sample_group_totals": {"total": len(classified), "accepted": len(accepted), "status_counts": dict(status_counts), "reason_counts": dict(reason_counts)}, "accepted_samples_per_class": dict(class_counts), "accepted_images_per_class": {label: sum(len(record["image_records"]) for record in accepted if record["mapped_class"] == label) for label in CLASSES}, "unique_sample_ids_per_class": {label: len({record["group_key"] for record in accepted if record["mapped_class"] == label}) for label in CLASSES}, "split_counts": {split: {"sample_groups": sum(counts.values()), "images": sum(counts.values()) * 4, "sample_groups_by_class": dict(counts), "images_by_class": {label: counts[label] * 4 for label in CLASSES}} for split, counts in split_counts.items()}, "class_percentages": {label: round(class_counts[label] / len(accepted) * 100, 2) if accepted else 0 for label in CLASSES}, "duplicate_statistics": {"cross_sample_exact_groups": len(exact_flags), "cross_sample_near_groups": len(near_flags), "duplicate_review_groups": duplicate_review_groups, "source_audit_exact_groups": len(audit["duplicates"]["exact_sha256_groups"]), "source_audit_near_groups": len(audit["duplicates"]["near_duplicate_groups"])}, "cross_split_leakage": {"detected": bool(leakage), "groups": leakage}, "metadata_coverage": {"samples_with_tag": sum(record["tag_exists"] for record in classified), "samples_with_composition": sum(record["has_composition"] for record in classified), "samples_without_composition": sum(not record["has_composition"] for record in classified), "missing_tag_samples": [record["group_key"] for record in classified if not record["tag_exists"]]}, "composition_evidence": "Original manufacturer-style tag.txt text and percentages are preserved in the CSV and manifest; claims are not independently laboratory verified.", "manifest_entries": manifest_entries}
    (ROOT / "Backend" / "data" / "metadata").mkdir(parents=True, exist_ok=True)
    AUDIT_OUTPUT.write_text(json.dumps(final_audit, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    duplicate_lines = ["# iBUG Material v2 Duplicate Review", "", "This is a review report. No source image was deleted or modified. A sample group is excluded from the accepted derived dataset when an exact or perceptual duplicate crosses sample groups.", "", f"- Source exact duplicate groups: **{len(audit['duplicates']['exact_sha256_groups'])}**", f"- Source near-duplicate groups: **{len(audit['duplicates']['near_duplicate_groups'])}**", f"- Cross-sample duplicate review groups: **{len(duplicate_review_groups)}**", f"- Accepted groups with unresolved cross-sample duplicate flags: **0**", "", "The duplicate policy is sample-level: all views remain together, and duplicate/near-duplicate groups crossing sample IDs are review exclusions rather than automatic source deletions.", ""]
    DUPLICATE_REPORT.write_text("\n".join(duplicate_lines), encoding="utf-8")
    report = ["# iBUG Material v2 Derived Dataset Report", "", "This derived dataset was created non-destructively from original iBUG composition metadata. Original folders and `tag.txt` files were not modified. No DeepFashion merge or model training was performed.", "", "## Taxonomy and Policy", "", f"Accepted taxonomy: **{', '.join(CLASSES)}**", "", "Acceptance requires one unambiguous supported composition component with a valid percentage, readable standard views, and no cross-sample exact/near-duplicate flag. Mixed compositions are not automatically assigned to pure classes and remain review candidates.", "", "## Sample-Level Results", "", f"- Total sample groups inspected: **{len(classified)}**", f"- Accepted groups: **{len(accepted)}**", f"- Rejected groups: **{status_counts['REJECTED']}**", f"- Review groups: **{status_counts['REVIEW']}**", f"- Missing `tag.txt`: **{reason_counts['MISSING_TAG']}**", f"- Unparseable or missing composition: **{reason_counts['UNPARSEABLE_OR_MISSING_COMPOSITION']}**", f"- Mixed-composition review groups: **{reason_counts['MIXED_COMPOSITION_REVIEW']}**", f"- Cross-sample duplicate review groups: **{len(duplicate_review_groups)}**", "", "## Accepted Groups and Images", "", "| Class | Accepted sample groups | Derived images | Class percentage |", "|---|---:|---:|---:|"]
    for label in CLASSES:
        report.append(f"| {label} | {class_counts[label]} | {sum(len(record['image_records']) for record in accepted if record['mapped_class'] == label)} | {final_audit['class_percentages'][label]:.2f}% |")
    report += ["", "## Group-Aware Split", "", "| Split | Total images | " + " | ".join(CLASSES) + " |", "|---|---:|" + "---:|" * len(CLASSES)]
    for split, counts in split_counts.items():
        report.append(f"| {split} | {sum(counts.values()) * 4} | " + " | ".join(str(counts[label] * 4) for label in CLASSES) + " |")
    ready = all(class_counts[label] >= 10 for label in CLASSES) and all(split_counts[split][label] > 0 for split in split_counts for label in CLASSES) and not leakage
    report += ["", f"Cross-split sample-group leakage: **{'YES' if leakage else 'NO'}**", "All views of each accepted sample group were copied together. The split is approximately 70/15/15 at the group/image level where indivisible groups permit.", "", "## Duplicate Statistics", "", f"- Source exact duplicate groups: **{len(audit['duplicates']['exact_sha256_groups'])}**", f"- Source near-duplicate groups: **{len(audit['duplicates']['near_duplicate_groups'])}**", f"- Cross-sample duplicate review groups: **{len(duplicate_review_groups)}**", f"- Accepted groups retained after duplicate policy: **{len(accepted)}**", "", "## Metadata and Composition Evidence", "", "- Every accepted group has original `tag.txt` composition evidence and a parseable percentage.", "- Original composition text and parsed percentages are preserved in `ibug_composition_manifest.csv` and the final audit JSON.", "- No garment type, folder name, filename, caption, or visual appearance was used as material evidence.", "- Manufacturer-style metadata was not independently laboratory verified.", "", "## Domain and Training Limitations", "", "iBUG uses controlled multi-view sample imagery and original composition metadata. DeepFashion uses fashion imagery with caption-derived labels. Their lighting, backgrounds, framing, garment coverage, sample grouping, and label provenance differ. Even this derived iBUG dataset may have class imbalance, source-specific artifacts, and composition claims that require independent verification.", "", f"**IBUG_V2_READY_FOR_TRAINING = {'YES' if ready else 'NO'}**", "", "Readiness requires all six classes to have reasonable counts, all splits to contain all six classes, no cross-split group leakage, and no unresolved duplicate issue in accepted groups. It does not guarantee model accuracy or laboratory verification.", "", "## Output Artifacts", "", f"- `{COMPOSITION_MANIFEST.relative_to(ROOT)}`", f"- `{DATASET_ROOT.relative_to(ROOT)}`", f"- `{SPLIT_ROOT.relative_to(ROOT)}`", f"- `{AUDIT_OUTPUT.relative_to(ROOT)}`", f"- `{DUPLICATE_REPORT.relative_to(ROOT)}`", ""]
    REPORT_OUTPUT.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"total_groups": len(classified), "accepted": len(accepted), "status_counts": dict(status_counts), "accepted_classes": dict(class_counts), "split_counts": {split: dict(counts) for split, counts in split_counts.items()}, "leakage": bool(leakage), "outputs": [str(COMPOSITION_MANIFEST), str(DATASET_ROOT), str(SPLIT_ROOT), str(AUDIT_OUTPUT), str(DUPLICATE_REPORT), str(REPORT_OUTPUT)]}, indent=2))


if __name__ == "__main__":
    main()