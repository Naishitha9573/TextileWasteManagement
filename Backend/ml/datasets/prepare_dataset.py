"""Build the auditable final dataset manifest from the real audit artifacts.

Combines:
- iBUG material v2 accepted/review/rejected sample groups (composition-labeled)
- DeepFashion verified-clean final set (caption-derived labels)
into data/metadata/final_dataset_manifest.csv with inclusion/exclusion reasons.

Usage (from Backend/):
    python -m ml.datasets.prepare_dataset
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from ml.datasets.common import METADATA_DIR, json_load

OUT_PATH = METADATA_DIR / "final_dataset_manifest.csv"

FIELDS = [
    "image_path",
    "source_dataset",
    "original_label",
    "final_label",
    "split",
    "duplicate_group",
    "near_duplicate_group",
    "included",
    "exclusion_reason",
    "sample_or_garment_id",
    "derived_path",
    "split_path",
]


def ibug_rows() -> list:
    audit = json_load(METADATA_DIR / "ibug_material_v2_audit.json")
    composition = {}
    comp_path = METADATA_DIR / "ibug_composition_manifest.csv"
    with open(comp_path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            composition[(row["original_folder"], row["sample_id"])] = row

    rows = []
    accepted_keys = set()
    for entry in audit["manifest_entries"]:
        key = (entry["original_folder"], entry["sample_id"])
        accepted_keys.add(key)
        group_row = composition.get(key, {})
        rows.append({
            "image_path": entry["source_path"],
            "source_dataset": "ibug_fabrics (derived: ibug_material_v2)",
            "original_label": entry["original_folder"],
            "final_label": entry["mapped_class"],
            "split": entry["split"],
            "duplicate_group": "; ".join(filter(None, group_row.get("exact_duplicate_groups", "").replace(", ", ";").split(";"))),
            "near_duplicate_group": "; ".join(filter(None, group_row.get("near_duplicate_groups", "").replace(", ", ";").split(";"))),
            "included": True,
            "exclusion_reason": "",
            "sample_or_garment_id": f"ibug:{key[0]}/{key[1]}",
            "derived_path": entry["derived_path"],
            "split_path": entry["split_path"],
        })

    # Excluded groups: every source image of every non-accepted group.
    excluded_images = 0
    seen_groups = set()
    for key, row in sorted(composition.items()):
        if key in accepted_keys or key in seen_groups:
            continue
        seen_groups.add(key)
        status = row.get("status", "")
        reasons = row.get("review_reasons", "")
        views = [v.strip() for v in row.get("views", "").split(";") if v.strip()]
        image_names = views if views else ["(no view info)"]
        folder, sample = key
        for view in image_names:
            src = f"Backend/datasets/ibug_fabrics/{folder}/{sample}/{view}.png"
            rows.append({
                "image_path": src,
                "source_dataset": "ibug_fabrics (excluded at review)",
                "original_label": folder,
                "final_label": row.get("mapped_class", ""),
                "split": "",
                "duplicate_group": row.get("exact_duplicate_groups", ""),
                "near_duplicate_group": row.get("near_duplicate_groups", ""),
                "included": False,
                "exclusion_reason": f"status={status}; reasons={reasons}",
                "sample_or_garment_id": f"ibug:{folder}/{sample}",
                "derived_path": "",
                "split_path": "",
            })
            excluded_images += 1
    print(f"iBUG rows: {sum(1 for r in rows if r['included'])} included; {excluded_images} excluded-image rows")
    return rows


def deepfashion_rows() -> list:
    final = json_load(METADATA_DIR / "material_dataset_final_manifest.json")
    included_paths = set()
    rows = []
    for entry in final["entries"]:
        included_paths.add(Path(entry["source_path"]).resolve().as_posix().lower())
        rows.append({
            "image_path": entry["source_path"],
            "source_dataset": "material_deepfashion (verified_clean subset)",
            "original_label": entry["label"],
            "final_label": entry["label"],
            "split": entry["split"],
            "duplicate_group": "",
            "near_duplicate_group": "",
            "included": True,
            "exclusion_reason": "",
            "sample_or_garment_id": f"deepfashion:{entry['garment_id']}",
            "derived_path": entry["final_path"],
            "split_path": entry["split_path"],
        })

    # Excluded: all processed material_deepfashion images not in the final clean set.
    processed_root = Path("data/processed/material_deepfashion")
    excluded = 0
    if processed_root.exists():
        for img in processed_root.rglob("*"):
            if not img.is_file():
                continue
            if img.resolve().as_posix().lower() in included_paths:
                continue
            rows.append({
                "image_path": str(img),
                "source_dataset": "material_deepfashion (excluded at review)",
                "original_label": img.parent.name,
                "final_label": "",
                "split": "",
                "duplicate_group": "",
                "near_duplicate_group": "",
                "included": False,
                "exclusion_reason": "not HIGH caption confidence and/or conflicting garment label (see material_label_review.csv)",
                "sample_or_garment_id": "",
                "derived_path": "",
                "split_path": "",
            })
            excluded += 1
    print(f"DeepFashion rows: {len(final['entries'])} included; {excluded} excluded")
    return rows


def main() -> int:
    rows = ibug_rows() + deepfashion_rows()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "total_rows": len(rows),
        "included": sum(1 for r in rows if r["included"]),
        "excluded": sum(1 for r in rows if not r["included"]),
        "by_source": {},
    }
    by_source = {}
    for r in rows:
        key = r["source_dataset"]
        stats = by_source.setdefault(key, {"rows": 0, "included": 0})
        stats["rows"] += 1
        stats["included"] += int(r["included"])
    summary["by_source"] = by_source
    sidecar = METADATA_DIR / "final_dataset_manifest_summary.json"
    sidecar.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({summary['total_rows']} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
