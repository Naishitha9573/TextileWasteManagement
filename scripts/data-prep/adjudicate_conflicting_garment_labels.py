from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
AUDIT_PATH = ROOT / "Backend" / "data" / "metadata" / "material_dataset_validation.json"
SOURCE_MANIFEST_PATH = ROOT / "Backend" / "data" / "manifests" / "material_deepfashion_manifest.json"
GROUPED_REVIEW_PATH = ROOT / "Backend" / "data" / "metadata" / "material_label_review.csv"
GROUPED_SPLIT_ROOT = ROOT / "Backend" / "data" / "splits" / "material_deepfashion_verified"
OUTPUT_CSV = ROOT / "Backend" / "data" / "metadata" / "conflicting_garment_labels.csv"
OUTPUT_REPORT = ROOT / "CONFLICTING_GARMENT_REVIEW.md"


def garment_id(path: str) -> str:
    stem = Path(path).stem
    return stem.split("-id_", 1)[1].split("_", 1)[0] if "-id_" in stem else stem


def conflict_group(labels: set[str]) -> str:
    normalized = frozenset(labels)
    known = {
        frozenset(("Cotton", "Denim")): "Cotton vs Denim",
        frozenset(("Cotton", "Mixed Fabrics")): "Cotton vs Mixed Fabrics",
        frozenset(("Denim", "Mixed Fabrics")): "Denim vs Mixed Fabrics",
    }
    return known.get(normalized, "other")


def main() -> None:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    captions = {
        str(Path(entry["image_path"]).resolve()): entry.get("original_label", "")
        for entry in manifest["entries"]
    }
    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    with GROUPED_REVIEW_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped_rows[row["garment_id"]].append(row)
    split_by_path = {}
    for split in ("train", "validation", "test"):
        for split_path in (GROUPED_SPLIT_ROOT / split).glob("*/*"):
            split_by_path[split_path.name] = split

    audit_by_path = {record["path"]: record for record in audit["records"]}
    all_records: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in audit["records"]:
        path = record["path"]
        review_row = next((row for row in grouped_rows[garment_id(path)] if row["image_path"] == path), None)
        all_records[garment_id(path)].append({
            "garment_id": garment_id(path),
            "image_path": path,
            "current_label": record["label"],
            "caption": captions.get(str((ROOT / path).resolve()), ""),
            "material_mentions": "; ".join(record["caption_material_mentions"]),
            "confidence": record["caption_label_confidence"],
            "split": split_by_path.get(Path(path).name, review_row["split"] if review_row else "UNKNOWN"),
        })

    conflicts = {
        group_id: rows
        for group_id, rows in all_records.items()
        if len({row["current_label"] for row in rows}) > 1
    }
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = ("garment_id", "image_path", "current_label", "caption", "material_mentions", "confidence", "split", "conflict_group")
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group_id in sorted(conflicts):
            rows = conflicts[group_id]
            kind = conflict_group({row["current_label"] for row in rows})
            for row in sorted(rows, key=lambda item: item["image_path"]):
                writer.writerow({**row, "conflict_group": kind})

    report = [
        "# Conflicting Garment Label Review",
        "",
        "This report is an adjudication queue only. No image was deleted, moved, relabeled, or modified, and no model training was performed.",
        "",
        "## Summary",
        "",
        f"- Conflicting garment IDs: **{len(conflicts)}**",
        f"- Images listed: **{sum(len(rows) for rows in conflicts.values())}**",
        "- Automatic relabeling: **none**",
        "- Caption material mentions are evidence only and do not prove fiber composition.",
        "",
        "## Conflict Types",
        "",
    ]
    counts: dict[str, int] = defaultdict(int)
    for rows in conflicts.values():
        counts[conflict_group({row["current_label"] for row in rows})] += 1
    for kind in ("Cotton vs Denim", "Cotton vs Mixed Fabrics", "Denim vs Mixed Fabrics", "other"):
        if counts[kind]:
            report.append(f"- {kind}: **{counts[kind]} garment IDs**")

    report += ["", "## Per-Garment Recommendations", ""]
    for group_id in sorted(conflicts):
        rows = conflicts[group_id]
        labels = sorted({row["current_label"] for row in rows})
        kind = conflict_group(set(labels))
        report += [
            f"### Garment ID `{group_id}`",
            "",
            f"- Conflict: **{kind}**",
            f"- Current labels: **{', '.join(labels)}**",
            "- Recommendation: **HUMAN_REVIEW**",
            "- Rationale: contradictory current material labels occur within one garment ID; caption-derived mentions cannot establish the actual fiber composition.",
            "",
            "| Image | Current label | Material mentions | Confidence | Split | Caption |",
            "|---|---|---|---|---|---|",
        ]
        for row in sorted(rows, key=lambda item: item["image_path"]):
            caption = row["caption"].replace("|", "\\|").replace("\n", " ")
            report.append(f"| `{row['image_path']}` | {row['current_label']} | {row['material_mentions'] or 'none'} | {row['confidence']} | {row['split']} | {caption} |")
        report.append("")

    report += [
        "## Recommendation Definitions",
        "",
        "- `KEEP_LABEL`: use only when the garment has no contradictory material evidence and the current label is supported by sufficiently strong evidence.",
        "- `HUMAN_REVIEW`: required here for every conflicting garment because current labels or material evidence disagree.",
        "- `EXCLUDE_FROM_MATERIAL_TRAINING`: use only if human review determines that the image is not a textile/garment or cannot support any material label.",
        "",
        "All 12 conflicting garment IDs are currently marked `HUMAN_REVIEW`; no automatic keep or exclusion decision was made.",
        "",
        f"Detailed machine-readable rows: `{OUTPUT_CSV.relative_to(ROOT)}`",
        "",
    ]
    OUTPUT_REPORT.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({
        "conflicting_garment_ids": len(conflicts),
        "images_listed": sum(len(rows) for rows in conflicts.values()),
        "conflict_types": dict(counts),
        "recommendations": {"HUMAN_REVIEW": len(conflicts), "KEEP_LABEL": 0, "EXCLUDE_FROM_MATERIAL_TRAINING": 0},
        "outputs": [str(OUTPUT_CSV), str(OUTPUT_REPORT)],
    }, indent=2))


if __name__ == "__main__":
    main()