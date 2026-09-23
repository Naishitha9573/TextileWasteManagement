"""Review cross-class duplicate groups without changing dataset labels or files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS_PATH = ROOT / "reports"
EXACT_PATH = REPORTS_PATH / "exact_duplicates.csv"
NEAR_PATH = REPORTS_PATH / "near_duplicates.csv"
CONFLICT_PATH = REPORTS_PATH / "cross_class_conflicts.csv"
SUMMARY_PATH = REPORTS_PATH / "cross_class_conflict_summary.json"
PROVENANCE_PATH = REPORTS_PATH / "sample_provenance_report.csv"
STATUSES = {
    "LABEL_CONFLICT_REQUIRES_REVIEW",
    "POSSIBLE_DUPLICATE_LABEL",
    "SAME_SAMPLE_DIFFERENT_LABEL",
    "UNKNOWN",
}
CONFLICT_FIELDS = [
    "duplicate_group_id",
    "duplicate_type",
    "class_1",
    "class_2",
    "all_involved_classes",
    "all_file_paths",
    "shared_sha256_hash",
    "perceptual_hash",
    "sample_ids",
    "conflict_status",
]
PROVENANCE_FIELDS = ["sample_id", "class_name", "image_count", "image_paths", "conflicting_classes", "conflict_status"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review duplicate groups crossing original class folders.")
    parser.add_argument("--exact", type=Path, default=EXACT_PATH, help=f"Exact duplicate CSV (default: {EXACT_PATH})")
    parser.add_argument("--near", type=Path, default=NEAR_PATH, help=f"Near duplicate CSV (default: {NEAR_PATH})")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise SystemExit(f"Unable to read {path}: {exc}") from exc


def group_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("duplicate_group_id"):
            groups[row["duplicate_group_id"]].append(row)
    return groups


def sample_id_from_path(file_path: str) -> str | None:
    parts = Path(file_path.replace("\\", "/")).parts
    if len(parts) >= 3 and parts[1]:
        return parts[1]
    return None


def normalized_pairs(classes: list[str]) -> list[str]:
    return [" ↔ ".join(sorted((left, right))) for index, left in enumerate(classes) for right in classes[index + 1:]]


def classify_group(group: list[dict[str, str]], duplicate_type: str) -> tuple[str, list[str], list[str]]:
    paths = sorted({row["file_path"] for row in group if row.get("file_path")})
    sample_ids = sorted({sample_id for path in paths if (sample_id := sample_id_from_path(path)) is not None})
    classes = sorted({row["class_name"] for row in group if row.get("class_name")})
    classes_by_sample: dict[str, set[str]] = defaultdict(set)
    for row in group:
        sample_id = sample_id_from_path(row.get("file_path", ""))
        if sample_id is not None:
            classes_by_sample[sample_id].add(row["class_name"])
    same_sample = any(len(sample_classes) > 1 for sample_classes in classes_by_sample.values())
    if same_sample:
        status = "SAME_SAMPLE_DIFFERENT_LABEL"
    elif duplicate_type == "exact" and all(sample_id_from_path(path) is not None for path in paths):
        status = "POSSIBLE_DUPLICATE_LABEL"
    elif duplicate_type == "near":
        status = "LABEL_CONFLICT_REQUIRES_REVIEW" if paths else "UNKNOWN"
    else:
        status = "UNKNOWN"
    return status, sample_ids, paths


def make_conflict(group_id: str, duplicate_type: str, group: list[dict[str, str]]) -> dict[str, Any]:
    classes = sorted({row["class_name"] for row in group if row.get("class_name")})
    status, sample_ids, paths = classify_group(group, duplicate_type)
    perceptual_hashes = sorted({row.get("perceptual_hash", "") for row in group if row.get("perceptual_hash")})
    sha_hashes = sorted({row.get("hash", "") for row in group if row.get("hash")})
    return {
        "duplicate_group_id": group_id,
        "duplicate_type": duplicate_type,
        "class_1": classes[0] if classes else "",
        "class_2": classes[1] if len(classes) > 1 else "",
        "all_involved_classes": json.dumps(classes),
        "all_file_paths": json.dumps(paths),
        "shared_sha256_hash": sha_hashes[0] if duplicate_type == "exact" and len(sha_hashes) == 1 else "",
        "perceptual_hash": "; ".join(perceptual_hashes),
        "sample_ids": json.dumps(sample_ids),
        "conflict_status": status,
        "_classes": classes,
        "_paths": paths,
        "_sample_ids": sample_ids,
    }


def build_provenance(conflicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    classes_by_sample: dict[str, set[str]] = defaultdict(set)
    paths_by_sample_class: dict[tuple[str, str], set[str]] = defaultdict(set)
    statuses_by_sample_class: dict[tuple[str, str], set[str]] = defaultdict(set)
    for conflict in conflicts:
        for path in conflict["_paths"]:
            sample_id = sample_id_from_path(path)
            if sample_id is None:
                continue
            class_name = path.split("/", 1)[0]
            classes_by_sample[sample_id].update(conflict["_classes"])
            paths_by_sample_class[(sample_id, class_name)].add(path)
            statuses_by_sample_class[(sample_id, class_name)].add(conflict["conflict_status"])
    for (sample_id, class_name), paths in paths_by_sample_class.items():
        conflicting = sorted(classes_by_sample[sample_id] - {class_name})
        statuses = statuses_by_sample_class[(sample_id, class_name)]
        status = "SAME_SAMPLE_DIFFERENT_LABEL" if conflicting else sorted(statuses)[0]
        records[(sample_id, class_name)] = {
            "sample_id": sample_id,
            "class_name": class_name,
            "image_count": len(paths),
            "image_paths": json.dumps(sorted(paths)),
            "conflicting_classes": json.dumps(conflicting),
            "conflict_status": status,
        }
    return [records[key] for key in sorted(records)]


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def main() -> int:
    args = parse_args()
    exact_groups = group_rows(read_rows(args.exact))
    near_groups = group_rows(read_rows(args.near))
    conflicts: list[dict[str, Any]] = []
    for duplicate_type, groups in (("exact", exact_groups), ("near", near_groups)):
        for group_id, rows in sorted(groups.items()):
            if len({row.get("class_name", "") for row in rows}) > 1:
                conflicts.append(make_conflict(group_id, duplicate_type, rows))

    exact_conflicts = [conflict for conflict in conflicts if conflict["duplicate_type"] == "exact"]
    near_conflicts = [conflict for conflict in conflicts if conflict["duplicate_type"] == "near"]
    pair_counts = Counter(pair for conflict in conflicts for pair in normalized_pairs(conflict["_classes"]))
    affected_paths = {path for conflict in conflicts for path in conflict["_paths"]}
    affected_classes = sorted({class_name for conflict in conflicts for class_name in conflict["_classes"]})
    status_counts = Counter(conflict["conflict_status"] for conflict in conflicts)
    examples = {
        status: [
            {"duplicate_group_id": conflict["duplicate_group_id"], "duplicate_type": conflict["duplicate_type"], "classes": conflict["_classes"], "file_paths": conflict["_paths"]}
            for conflict in conflicts if conflict["conflict_status"] == status
        ][:5]
        for status in sorted(STATUSES)
    }
    summary = {
        "total_cross_class_exact_groups": len(exact_conflicts),
        "total_cross_class_near_groups": len(near_conflicts),
        "affected_images": len(affected_paths),
        "class_pair_counts": dict(sorted(pair_counts.items())),
        "affected_classes": affected_classes,
        "status_counts": dict(sorted(status_counts.items())),
        "examples_of_each_conflict": examples,
        "sample_id_extraction": "Second path component, e.g. Class/1000/im_1.png -> 1000, when present.",
        "read_only": True,
    }
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    write_csv(CONFLICT_PATH, CONFLICT_FIELDS, conflicts)
    write_csv(PROVENANCE_PATH, PROVENANCE_FIELDS, build_provenance(conflicts))
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("CROSS-CLASS CONFLICT REVIEW")
    print("===========================")
    print(f"Cross-class exact groups: {len(exact_conflicts)}")
    print(f"Cross-class near groups: {len(near_conflicts)}")
    print(f"Affected images: {len(affected_paths)}")
    print(f"Affected classes: {', '.join(affected_classes) or 'None'}")
    print("Class-pair counts:")
    for pair, count in sorted(pair_counts.items()):
        print(f"  {pair}: {count}")
    print("Statuses:")
    for status in sorted(STATUSES):
        print(f"  {status}: {status_counts[status]}")
    print(f"Reports: {CONFLICT_PATH}, {SUMMARY_PATH}, {PROVENANCE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())