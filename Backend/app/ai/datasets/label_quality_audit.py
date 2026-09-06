"""Audit label quality for the 3-class material_deepfashion dataset."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from app.ai.datasets.mapping import extract_material_from_caption, load_class_mapping
from app.ai.datasets.paths import DATASETS_ROOT, MANIFESTS_DIR, METADATA_DIR, ensure_data_dirs
from app.ai.datasets.validators import (
    collect_image_files,
    find_duplicate_hashes,
    validate_image,
)


def _caption_materials(caption: str, mapping: Dict) -> List[str]:
    keywords = mapping.get("deepfashion_caption_keywords", {})
    caption_lower = caption.lower()
    found = []
    for keyword, target in keywords.items():
        if target and re.search(rf"\b{re.escape(keyword.lower())}\b", caption_lower):
            if target not in found:
                found.append(target)
    return found


def audit_label_quality() -> Dict[str, Any]:
    ensure_data_dirs()
    manifest_path = MANIFESTS_DIR / "material_deepfashion_manifest.json"
    splits_path = MANIFESTS_DIR / "material_deepfashion_splits.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run prepare first.")

    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    mapping = load_class_mapping()
    entries = manifest.get("entries", [])

    class_counts = Counter(e["mapped_label"] for e in entries)
    multi_material_captions = 0
    label_mismatch = 0
    suspected_noisy: List[Dict] = []
    filename_to_labels: Dict[str, set] = defaultdict(set)

    for entry in entries:
        caption = entry.get("caption", "")
        mapped = entry.get("mapped_label", "")
        materials_in_caption = _caption_materials(caption, mapping)

        if len(materials_in_caption) > 1:
            multi_material_captions += 1

        if materials_in_caption and mapped not in materials_in_caption and mapped != "Mixed Fabrics":
            label_mismatch += 1
            if len(suspected_noisy) < 20:
                suspected_noisy.append({
                    "file": Path(entry.get("source", "")).name,
                    "mapped_label": mapped,
                    "materials_in_caption": materials_in_caption,
                    "caption_snippet": caption[:120],
                })

        fname = Path(entry.get("source", "")).name
        filename_to_labels[fname].add(mapped)

    cross_class_duplicates = {
        fname: list(labels)
        for fname, labels in filename_to_labels.items()
        if len(labels) > 1
    }

    processed_root = Path(manifest["entries"][0]["processed"]).parent if entries else None
    image_paths = []
    if processed_root and processed_root.exists():
        image_paths = collect_image_files(processed_root)

    corrupt = []
    for path in image_paths[:500]:
        check = validate_image(path)
        if not check["valid"]:
            corrupt.append(str(path))

    dup_groups = find_duplicate_hashes(image_paths, max_files=len(image_paths))

    split_info = {}
    if splits_path.exists():
        with open(splits_path, "r", encoding="utf-8") as handle:
            split_data = json.load(handle)
        split_info = split_data.get("split_counts", {})

    captions_path = DATASETS_ROOT / "deepfashion" / "captions.json"
    caption_stats = {"available": captions_path.exists()}

    report: Dict[str, Any] = {
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "material_deepfashion",
        "current_model_scope": "3 CLASS ONLY",
        "target_application_classes": 10,
        "trained_classes": sorted(class_counts.keys()),
        "missing_target_classes": [
            c for c in mapping.get("target_material_classes", [])
            if c not in class_counts
        ],
        "total_images": len(entries),
        "images_per_class": dict(class_counts),
        "label_source": "DeepFashion captions.json",
        "label_generation_method": "Keyword substring match via class_mapping.yaml deepfashion_caption_keywords",
        "label_quality_assessment": {
            "reliable_for_production": False,
            "reason": "Caption-derived labels are not verified fabric composition; multi-garment captions; keyword ambiguity",
            "multi_material_captions": multi_material_captions,
            "multi_material_rate": round(multi_material_captions / max(len(entries), 1), 4),
            "label_keyword_mismatch_count": label_mismatch,
            "cross_class_filename_duplicates": len(cross_class_duplicates),
        },
        "suspected_noisy_samples": suspected_noisy,
        "duplicate_hash_groups": len(dup_groups),
        "corrupted_images_sampled": len(corrupt),
        "corrupted_sample_size": min(500, len(image_paths)),
        "train_val_test_counts": split_info,
        "class_balance": {
            "balanced": len(set(class_counts.values())) <= 1,
            "counts": dict(class_counts),
        },
        "known_limitations": [
            "Only 3 of 10 target material classes present",
            "Labels inferred from natural-language captions, not lab analysis",
            "Same garment may mention multiple materials in one caption",
            "Capped at 500 images per class during prepare (not full DeepFashion distribution)",
            "Visual similarity between Cotton/Denim/Mixed may cause model confusion",
        ],
        "caption_file": str(captions_path),
    }

    out_json = METADATA_DIR / "label_quality_audit.json"
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return report


def render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# ML Label Quality Report",
        "",
        f"**Audited:** {report['audited_at']}",
        f"**Scope:** CURRENT MODEL — **{report['current_model_scope']}**",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|------:|",
        f"| Total images | {report['total_images']} |",
        f"| Classes in dataset | {len(report['trained_classes'])} |",
        f"| Target app taxonomy | {report['target_application_classes']} classes |",
        f"| Multi-material captions | {report['label_quality_assessment']['multi_material_captions']} ({report['label_quality_assessment']['multi_material_rate']:.1%}) |",
        f"| Keyword/label mismatches | {report['label_quality_assessment']['label_keyword_mismatch_count']} |",
        f"| Duplicate hash groups | {report['duplicate_hash_groups']} |",
        f"| Corrupted (sample) | {report['corrupted_images_sampled']} |",
        "",
        "## Classes",
        "",
        "### Present (trained)",
    ]
    for cls, count in report["images_per_class"].items():
        lines.append(f"- **{cls}:** {count} images")

    lines.extend(["", "### Missing from target taxonomy (MISSING DATA)", ""])
    for cls in report["missing_target_classes"]:
        lines.append(f"- {cls} — **MISSING DATA**")

    lines.extend([
        "",
        "## Label Provenance",
        "",
        f"- **Source:** {report['label_source']}",
        f"- **Method:** {report['label_generation_method']}",
        f"- **Production-ready:** {report['label_quality_assessment']['reliable_for_production']}",
        f"- **Reason:** {report['label_quality_assessment']['reason']}",
        "",
        "## Splits",
        "",
        f"```json\n{json.dumps(report.get('train_val_test_counts', {}), indent=2)}\n```",
        "",
        "## Why Labels Are Noisy",
        "",
        "1. Captions describe **garments**, not isolated fabric swatches.",
        "2. A single caption may mention cotton pants AND cotton shirt — mapped to one class.",
        "3. Multi-keyword captions become **Mixed Fabrics** even when one material dominates visually.",
        "4. **Denim** is often cotton-based; visual distinction from **Cotton** is subtle.",
        "5. Keyword matching is not verified against pixel-level material composition.",
        "",
        "## Suspected Noisy Samples (sample)",
        "",
    ])
    for sample in report.get("suspected_noisy_samples", [])[:10]:
        lines.append(f"- `{sample['file']}` mapped **{sample['mapped_label']}** but caption mentions {sample['materials_in_caption']}")

    lines.extend(["", "## Known Limitations", ""])
    for item in report["known_limitations"]:
        lines.append(f"- {item}")

    lines.append("")
    lines.append("*Generated by `python -m app.ai.datasets.label_quality_audit` — measured from manifest, not estimated.*")
    return "\n".join(lines)


def main() -> int:
    report = audit_label_quality()
    md = render_markdown(report)
    out_md = Path(__file__).resolve().parents[3].parent / "ML_LABEL_QUALITY_REPORT.md"
    out_md.write_text(md, encoding="utf-8")
    print(json.dumps({
        "total_images": report["total_images"],
        "classes": report["trained_classes"],
        "multi_material_rate": report["label_quality_assessment"]["multi_material_rate"],
        "report": str(out_md),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
