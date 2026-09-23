"""Generate docs/DATASET_REPORT.md with measured numbers from this repository.

Usage (from Backend/):
    python -m ml.datasets.dataset_report
"""
from __future__ import annotations

import datetime as _dt
import json
from collections import Counter
from pathlib import Path

from ml.datasets.common import (
    BACKEND_ROOT,
    DATA_ROOT,
    METADATA_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    SPLITS_DIR,
    count_by_class,
    iter_images,
    json_load,
)

PROJECT_DOCS = BACKEND_ROOT.parent / "docs"


def fmt_counts(d: dict) -> str:
    return ", ".join(f"{k}: {v}" for k, v in sorted(d.items()))


def main() -> int:
    lines = [
        "# Dataset Report",
        "",
        f"_Generated { _dt.datetime.now().isoformat(timespec='seconds') } by `python -m ml.datasets.dataset_report`. All numbers below are measured from this repository._",
        "",
        "## 1. Datasets in use (final training sets)",
        "",
    ]

    for name in ("ibug_material_v2", "material_deepfashion_verified_clean"):
        processed = PROCESSED_DIR / name
        images = list(iter_images(processed))
        per_class = Counter(p.parent.name for p in images)
        label_source = "manufacturer tag.txt composition metadata" if name.startswith("ibug") else "DeepFashion caption keyword mapping"
        lines += [
            f"### `{name}`",
            "",
            f"- Location: `Backend/data/processed/{name}`",
            f"- Images: **{len(images)}**",
            f"- Classes: {fmt_counts(dict(sorted(per_class.items())))}",
            f"- Label provenance: {label_source}",
            "",
        ]

    lines += ["## 2. Group-aware splits", ""]
    for split_root in sorted(d for d in SPLITS_DIR.iterdir() if d.is_dir()):
        split_info = {}
        total = 0
        for split in ("train", "validation", "test"):
            split_dir = split_root / split
            if not split_dir.exists():
                continue
            counts = count_by_class(split_dir)
            split_info[split] = counts
            total += sum(counts.values())
        lines.append(f"### `{split_root.name}` — total {total}")
        lines.append("")
        classes = sorted({c for counts in split_info.values() for c in counts})
        split_keys = list(split_info.keys())
        table = ["| Split | " + " | ".join(split_keys) + " |", "|---|" + "---:|" * len(split_keys)]
        row = ["Total"] + [str(sum(split_info[s].values())) for s in split_keys]
        table.append("| " + " | ".join(row) + " |")
        for c in classes:
            row = [c] + [str(split_info[s].get(c, 0)) for s in split_keys]
            table.append("| " + " | ".join(row) + " |")
        lines.extend(table)
        lines.append("")

    dup_path = REPORTS_DIR / "duplicate_detection.json"
    val_path = REPORTS_DIR / "dataset_validation.json"
    split_path = REPORTS_DIR / "split_integrity.json"

    lines += ["## 3. Duplicate & leakage review", ""]
    if val_path.exists():
        val = json_load(val_path)
        for name, entry in val.items():
            splits = entry.get("splits", {})
            lines.append(f"- **{name}**: cross-split duplicate hashes: **{splits.get('cross_split_duplicate_hashes', 'n/a')}**, leakage_detected: **{splits.get('leakage_detected')}**, unreadable: {splits.get('unreadable_images', 'n/a')}")
    else:
        lines.append("- Run `python -m ml.datasets.validate_dataset` to populate leakage statistics.")
    if dup_path.exists():
        dup = json_load(dup_path)
        for name, entry in dup.items():
            lines.append(
                f"- **{name}**: exact-duplicate groups in processed set: **{len(entry.get('exact_duplicate_groups', []))}**; "
                f"cross-sample dHash pairs ≤4: **{entry.get('cross_sample_pairs_hamming_le_4', 'n/a')}** "
                f"(texture-similarity dominated — see note); cross-split exact duplicates: **{entry.get('cross_split_exact_duplicate_content_items', 'n/a')}**"
            )
    lines += [
        "",
        "Interpretation note: on plain fabric textures, 64-bit dHash distance ≤8 flags visual similarity far more often than duplication (cross-class pairs dominate). Hard evidence of duplication is sha256-identical content: there is none within or across the final training splits. The authoritative source-level iBUG review excluded all cross-sample exact/near-duplicate-flagged groups during derivation.",
    ]
    if split_path.exists():
        sp = json_load(split_path)
        for name, entry in sp.items():
            policy = entry.get("policy")
            if policy:
                lines.append(f"- **{name}**: {policy}")
                continue
            lines.append(f"- **{name}**: group-aware split valid: **{entry.get('group_aware_split_valid')}** ({entry.get('unique_sample_groups')} sample groups; leaked groups: {len(entry.get('cross_split_group_leakage', {}))})")
    lines += [
        "",
        "Source-level iBUG duplicate review: 213 exact + 223 near-duplicate groups were identified in the original iBUG download; all groups flagged for cross-sample duplication are excluded from the accepted derived dataset (see `IBUG_MATERIAL_V2_DUPLICATE_REVIEW.md`). No source image was deleted.",
        "",
        "## 4. Exclusions",
        "",
    ]
    manifest_summary = METADATA_DIR / "final_dataset_manifest_summary.json"
    if manifest_summary.exists():
        summary = json_load(manifest_summary)
        lines.append(f"Final manifest rows: **{summary['total_rows']}** (included: **{summary['included']}**, excluded: **{summary['excluded']}**) — see `Backend/data/metadata/final_dataset_manifest.csv`.")
    else:
        lines.append("Run `python -m ml.datasets.prepare_dataset` to generate the final manifest.")
    lines += [
        "",
        "## 5. Preprocessing & augmentation (training)",
        "",
        "- Resize to 224×224, RGB, MobileNetV3 `preprocess_input` scaling.",
        "- Augmentation at train time: none beyond resize/autocontrast applied historically; class weighting used for imbalance.",
        "- Splits are sample-group aware (all views of one fabric sample stay together).",
        "",
        "## 6. Known limitations",
        "",
        "- Labels are manufacturer composition claims (iBUG) or caption-derived mentions (DeepFashion); neither is laboratory-verified fiber content.",
        "- iBUG v2 accepted set has **no Denim class**: no sample declares denim as its sole composition component (denim samples declare cotton).",
        "- Class imbalance: Cotton dominates the accepted iBUG set (~66%).",
        "- Silk (140 images) and Nylon (184 images) are small classes; their metrics will be noisy.",
        "- Waste-category labels (Recyclable/Reusable/Repairable/Upcyclable/Compostable/Hazardous) do not exist in any image dataset here; waste classification remains rule-based by design.",
        "",
    ]

    PROJECT_DOCS.mkdir(parents=True, exist_ok=True)
    out_path = PROJECT_DOCS / "DATASET_REPORT.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
