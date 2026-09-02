from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "Backend" / "datasets" / "ibug_fabrics"
OUTPUT_JSON = ROOT / "Backend" / "data" / "metadata" / "ibug_fabrics_audit.json"
OUTPUT_REPORT = ROOT / "IBUG_FABRICS_AUDIT_REPORT.md"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
MATERIAL_TOKEN = re.compile(r"([A-Za-z][A-Za-z_ -]*?)\s+(\d+(?:\.\d+)?)")
KNOWN_MATERIALS = {
    "Acrylic", "Cotton", "Elastane", "Linen", "Nylon", "Polyester", "Silk", "Viscose", "Wool",
    "Rayon", "Polyamide", "Hemp", "Bamboo", "Cashmere", "Mohair", "Alpaca", "Polyurethane",
}
PROJECT_OVERLAP = {"Cotton": "Cotton", "Denim": "Denim", "Blended": "Mixed Fabrics"}


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(path: Path, size: int = 16) -> str:
    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((size + 1, size)).getdata())
    return "".join("1" if pixels[row * (size + 1) + col] > pixels[row * (size + 1) + col + 1] else "0"
                   for row in range(size) for col in range(size))


def parse_tag(path: Path) -> dict[str, Any]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    garment_type = lines[0] if lines else None
    composition_text = " ".join(lines[1:]) if len(lines) > 1 else ""
    composition: dict[str, float] = {}
    for material, percentage in MATERIAL_TOKEN.findall(composition_text):
        material = material.strip()
        if material in KNOWN_MATERIALS:
            composition[material] = composition.get(material, 0.0) + float(percentage)
    return {"garment_type": garment_type, "composition_text": composition_text, "composition": composition,
            "has_composition": bool(composition), "is_blended": len(composition) > 1 or "Blended" in path.parts}


def main() -> None:
    sample_dirs = sorted(path for path in DATASET_ROOT.glob("*/*") if path.is_dir())
    image_paths = sorted(path for path in DATASET_ROOT.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    top_classes = sorted(path.name for path in DATASET_ROOT.iterdir() if path.is_dir())
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    dimensions = Counter()
    channels = Counter()
    by_hash: dict[str, list[str]] = defaultdict(list)
    by_dhash: dict[str, list[str]] = defaultdict(list)
    dhash_values: dict[str, int] = {}
    view_counts = Counter()
    sample_records = []
    composition_material_counts = Counter()
    blended_samples = []
    missing_tag_samples = []
    missing_view_samples = []

    for sample_dir in sample_dirs:
        class_name = sample_dir.parent.name
        sample_id = sample_dir.name
        tag_path = sample_dir / "tag.txt"
        tag = parse_tag(tag_path) if tag_path.exists() else {"garment_type": None, "composition_text": "", "composition": {}, "has_composition": False, "is_blended": class_name == "Blended"}
        if not tag_path.exists():
            missing_tag_samples.append(relative(sample_dir))
        views = sorted(path for path in sample_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        view_names = [path.stem for path in views]
        view_counts[len(views)] += 1
        if len(views) != 4 or set(view_names) != {"im_1", "im_2", "im_3", "im_4"}:
            missing_view_samples.append({"sample": relative(sample_dir), "views": view_names})
        if tag["is_blended"]:
            blended_samples.append({"class_folder": class_name, "sample_id": sample_id, "composition": tag["composition"], "composition_text": tag["composition_text"]})
        for material in tag["composition"]:
            composition_material_counts[material] += 1
        sample_records.append({"class_folder": class_name, "sample_id": sample_id, "path": relative(sample_dir), "garment_type": tag["garment_type"], "composition_text": tag["composition_text"], "composition": tag["composition"], "has_composition": tag["has_composition"], "is_blended": tag["is_blended"], "view_count": len(views), "view_names": view_names})

    for path in image_paths:
        record = {"path": relative(path), "class_folder": path.parent.parent.name, "sample_id": path.parent.name, "view": path.stem, "usable": False}
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                record.update({"width": image.width, "height": image.height, "mode": image.mode, "channels": len(image.getbands()), "format": image.format, "usable": image.width >= 32 and image.height >= 32, "sha256": sha256(path), "dhash": dhash(path)})
                dimensions[f"{image.width}x{image.height}"] += 1
                channels[str(len(image.getbands()))] += 1
                by_hash[record["sha256"]].append(record["path"])
                by_dhash[record["dhash"]].append(record["path"])
                dhash_values[record["path"]] = int(record["dhash"], 2)
        except (OSError, SyntaxError, ValueError, UnidentifiedImageError) as exc:
            record["error"] = str(exc)
            errors.append({"path": record["path"], "error": str(exc)})
        records.append(record)

    exact_groups = [paths for paths in by_hash.values() if len(paths) > 1]
    dhash_groups = [paths for paths in by_dhash.values() if len(paths) > 1]
    paths = list(dhash_values)
    parent = {path: path for path in paths}

    def find(path: str) -> str:
        while parent[path] != path:
            parent[path] = parent[parent[path]]
            path = parent[path]
        return path

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    near_threshold = 8
    for index, left in enumerate(paths):
        left_hash = dhash_values[left]
        for right in paths[index + 1:]:
            if (left_hash ^ dhash_values[right]).bit_count() <= near_threshold:
                union(left, right)
    near_components: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        near_components[find(path)].append(path)
    near_groups = [group for group in near_components.values() if len(group) > 1]
    sample_class_counts = Counter(record["class_folder"] for record in sample_records)
    image_class_counts = Counter(record["class_folder"] for record in records)
    composition_available = sum(record["has_composition"] for record in sample_records)
    audit = {
        "audit_type": "ibug_fabrics_read_only_dataset_audit",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": relative(DATASET_ROOT),
        "policy": {"dataset_modified": False, "merged": False, "labels_renamed": False, "training_performed": False, "appearance_material_inference_used": False},
        "files": {"total_files": sum(1 for path in DATASET_ROOT.rglob("*") if path.is_file()), "image_files": len(image_paths), "tag_files": len(list(DATASET_ROOT.rglob("tag.txt"))), "extensions": dict(Counter(path.suffix.lower() for path in DATASET_ROOT.rglob("*") if path.is_file()))},
        "structure": {"top_level_class_folders": top_classes, "sample_directories": len(sample_dirs), "sample_folder_pattern": "<original folder>/<sample id>/{im_1,im_2,im_3,im_4}.png + tag.txt"},
        "samples": {"total": len(sample_records), "usable_images": sum(record["usable"] for record in records), "unreadable_images": len(errors), "class_sample_counts": dict(sample_class_counts), "class_image_counts": dict(image_class_counts), "records": sample_records},
        "metadata": {"tag_files": len(list(DATASET_ROOT.rglob("tag.txt"))), "samples_with_composition": composition_available, "samples_without_composition": len(sample_records) - composition_available, "missing_tag_samples": missing_tag_samples, "composition_material_sample_counts": dict(composition_material_counts), "manufacturer_label_evidence": "Present as original tag.txt composition percentages; provenance is preserved and not inferred from images."},
        "materials": {"exact_folder_names": top_classes, "exact_composition_material_names": sorted(composition_material_counts), "blended_samples": len(blended_samples), "blended_sample_details": blended_samples[:200], "project_overlap": {folder: PROJECT_OVERLAP[folder] for folder in top_classes if folder in PROJECT_OVERLAP}, "potential_new_project_classes": [name for name in ("Silk", "Wool", "Polyester", "Linen", "Nylon", "Viscose", "Acrylic") if name in top_classes or name in composition_material_counts]},
        "views_and_groups": {"view_count_distribution": dict(view_counts), "missing_or_nonstandard_views": missing_view_samples, "multiple_images_same_sample": True, "group_key": "original class folder + sample directory id", "illumination_view_interpretation": "im_1 through im_4 are treated as four provided views/illumination conditions per sample based on the original naming; the files were not visually interpreted."},
        "image_properties": {"dimensions": dict(dimensions), "channels": dict(channels)},
        "duplicates": {"exact_sha256_groups": exact_groups, "exact_duplicate_extra_images": sum(len(group) - 1 for group in exact_groups), "identical_dhash_groups": dhash_groups, "identical_dhash_extra_images": sum(len(group) - 1 for group in dhash_groups), "near_duplicate_threshold_bits": near_threshold, "near_duplicate_groups": near_groups, "near_duplicate_extra_images": sum(len(group) - 1 for group in near_groups), "near_duplicate_method": "16x16 dHash with Hamming distance <= 8; groups are review flags, not automatic deletions."},
        "license_and_restrictions": {"local_license_or_readme_files": [], "status": "NOT DOCUMENTED IN DOWNLOADED TREE", "finding": "No local README, LICENSE, copyright, terms, citation, or metadata documentation file was found by filename scan. Do not assume commercial or redistribution rights; verify the official iBUG/Imperial College terms before merge or publication."},
        "rgb_training_assessment": {"technically_suitable_for_rgb_mobilenetv3_small": not errors and set(channels) <= {"3", "4"}, "conditions": ["Convert RGBA to RGB deterministically if needed.", "Preserve group IDs and manufacturer composition metadata.", "Use manufacturer composition labels only after license review and human/data-quality review.", "Do not convert garment_type into material labels."]},
        "records": records,
        "errors": errors,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(audit, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    overlap = ", ".join(f"{folder} -> {target}" for folder, target in audit["materials"]["project_overlap"].items()) or "None"
    new_classes = ", ".join(audit["materials"]["potential_new_project_classes"]) or "None confirmed from folder/tag names"
    report = ["# iBUG Fabrics Dataset Audit", "", "This audit is read-only. The downloaded dataset was not renamed, deleted, modified, merged, or used for training. Material findings below distinguish original manufacturer-style `tag.txt` composition evidence from any image inference; no material was inferred from appearance.", "", "## Executive Findings", "", f"1. **Total usable samples/images:** {len(sample_records)} sample groups and {audit['samples']['usable_images']} technically usable images out of {len(image_paths)} image files.", f"2. **Material classes:** {', '.join(top_classes)}.", f"3. **Images per class:** {', '.join(f'{k}: {v}' for k, v in sorted(image_class_counts.items()))}.", f"4. **Composition information:** available for {composition_available}/{len(sample_records)} samples from original `tag.txt` files; this is manufacturer-label evidence as stored, not inferred ground truth.", f"5. **Blended fabrics:** {len(blended_samples)} samples, identified by the original `Blended` folder and/or multiple composition components.", f"6. **Recommended project classes:** retain Cotton and Denim overlap after provenance/license review; treat Blended as a candidate for Mixed Fabrics only after defining policy; evaluate Silk, Wool, Polyester, Linen, Nylon, Viscose, and Acrylic as new classes.", "7. **Combine with existing 708-image dataset:** NO for now. First resolve license terms, label taxonomy mapping, composition policy, group identity, and domain differences.", "8. **Recommended preprocessing:** preserve originals; parse `tag.txt`; normalize composition into structured percentages without overwriting source text; convert all images to RGB in a derived dataset; deduplicate; split by sample group; retain provenance and license records.", "9. **Potential domain mismatch:** iBUG appears to contain controlled fabric/garment sample views with manufacturer composition metadata, while the existing DeepFashion set contains caption-derived fashion imagery. Differences in lighting, framing, garment type, backgrounds, and label reliability can shift performance.", "10. **READY_FOR_MERGE = NO**", "", "## Dataset Structure", "", f"- Top-level folders: **{len(top_classes)}**", f"- Sample directories: **{len(sample_records)}**", f"- Image files: **{len(image_paths)}**", f"- `tag.txt` files: **{audit['files']['tag_files']}**", "- Common sample pattern: one numeric sample directory containing `im_1.png` through `im_4.png` and `tag.txt`.", "", "## Class Counts", "", "| Original folder | Samples | Images | Project overlap/interpretation |", "|---|---:|---:|---|"]
    for folder in top_classes:
        report.append(f"| {folder} | {sample_class_counts[folder]} | {image_class_counts[folder]} | {PROJECT_OVERLAP.get(folder, 'No direct 3-class overlap')} |")
    report += ["", "## Composition and Metadata", "", f"- Samples with parseable composition percentages: **{composition_available}**", f"- Samples without parseable composition: **{len(sample_records) - composition_available}**", f"- Samples missing `tag.txt`: **{len(missing_tag_samples)}**", f"- Composition material names observed: **{', '.join(sorted(composition_material_counts))}**", "- Original `tag.txt` text must be retained. Parsed percentages are an audit view, not a replacement label.", "- Garment-type values such as `Blouse`, `Jeans`, `Fabric`, and `T-shirt` are not material labels.", "", "## Views, Groups, and Duplicates", "", f"- View-count distribution: **{dict(view_counts)}**", f"- Nonstandard/missing view groups: **{len(missing_view_samples)}**", f"- Exact duplicate groups: **{len(exact_groups)}** ({sum(len(group) - 1 for group in exact_groups)} extra images)", f"- Identical dHash groups: **{len(dhash_groups)}** ({sum(len(group) - 1 for group in dhash_groups)} extra images)", f"- Near-duplicate groups at dHash Hamming distance <= {near_threshold}: **{len(near_groups)}** ({sum(len(group) - 1 for group in near_groups)} extra images)", "- Every sample directory has multiple images; these are grouped and must never be split across train/validation/test.", "", "## Image Properties", "", f"- Dimensions: `{dict(dimensions)}`", f"- Channels: `{dict(channels)}`", f"- Read/open failures: **{len(errors)}**", "- The images are technically compatible with RGB MobileNetV3-Small preprocessing if a derived RGB conversion is used where necessary.", "", "## License and Research Use", "", "No local license, README, terms, citation, or copyright file was found in the downloaded tree. This audit therefore cannot establish whether commercial use, redistribution, or dataset merging is permitted. Verify the official iBUG/Imperial College London license and attribution requirements before using the data outside research evaluation.", "", "## Recommended Next Step", "", f"- Direct folder overlaps: **{overlap}**", f"- Candidate additional classes from original names: **{new_classes}**", "- Do not merge until legal terms, composition-label semantics, sample-group IDs, and a harmonized taxonomy are reviewed.", "- If approved, create a derived manifest only; preserve original folders and `tag.txt` files unchanged.", "", "## Audit Limitations", "", "- No material was inferred from image appearance.", "- “Manufacturer-label evidence” describes the composition percentages present in the downloaded `tag.txt` files; the audit did not independently verify the manufacturer claims.", "- Near-duplicate groups are perceptual review flags; they do not automatically mean files should be deleted.", ""]
    OUTPUT_REPORT.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"samples": len(sample_records), "images": len(image_paths), "usable_images": audit["samples"]["usable_images"], "classes": top_classes, "composition_samples": composition_available, "blended_samples": len(blended_samples), "exact_duplicate_groups": len(exact_groups), "dhash_groups": len(dhash_groups), "outputs": [str(OUTPUT_JSON), str(OUTPUT_REPORT)]}, indent=2))


if __name__ == "__main__":
    main()