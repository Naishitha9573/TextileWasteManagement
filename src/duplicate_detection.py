"""Read-only exact and near duplicate detection for the original iBUG dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = ROOT / "Backend" / "datasets" / "ibug_fabrics"
REPORTS_PATH = ROOT / "reports"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_NEAR_THRESHOLD = 6
PROGRESS_INTERVAL = 250
CSV_FIELDS = [
    "duplicate_group_id",
    "duplicate_type",
    "class_name",
    "file_path",
    "hash",
    "perceptual_hash",
    "perceptual_distance",
    "paired_file_path",
    "paired_class_name",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect image duplicates without modifying the dataset.")
    parser.add_argument("dataset_path", nargs="?", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_NEAR_THRESHOLD,
        help=(
            "Maximum pHash Hamming distance for near-duplicate pairs "
            f"(default: {DEFAULT_NEAR_THRESHOLD}; lower values are stricter)."
        ),
    )
    return parser.parse_args()


def collect_images(dataset_path: Path) -> list[tuple[str, Path]]:
    class_folders = sorted(path for path in dataset_path.iterdir() if path.is_dir())
    images = []
    for class_folder in class_folders:
        for path in sorted(class_folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                images.append((class_folder.name, path))
    return images


def relative_path(path: Path, dataset_path: Path) -> str:
    return path.relative_to(dataset_path).as_posix()


def calculate_hashes(path: Path) -> tuple[str, str]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    with Image.open(path) as image:
        image.load()
        pixels = np.asarray(image.convert("L").resize((32, 32)), dtype=np.float64)
        dct_matrix = _dct_matrix(32)
        frequencies = dct_matrix @ pixels @ dct_matrix.T
        coefficients = frequencies[:8, :8].flatten()[1:]
        median = float(np.median(coefficients))
        bits = "".join("1" if value > median else "0" for value in coefficients)
        perceptual_hash = f"{int(bits, 2):016x}"
    return digest.hexdigest(), perceptual_hash


@lru_cache(maxsize=None)
def _dct_matrix(size: int) -> np.ndarray:
    matrix = np.empty((size, size), dtype=np.float64)
    factor = np.pi / (2 * size)
    for row in range(size):
        scale = np.sqrt(1 / size) if row == 0 else np.sqrt(2 / size)
        for column in range(size):
            matrix[row, column] = scale * np.cos((2 * column + 1) * row * factor)
    return matrix


class PhashIndex:
    """A BK-tree index for Hamming-distance queries over hexadecimal pHashes."""

    def __init__(self) -> None:
        self.root: tuple[int, str, list[dict[str, Any]], dict[int, Any]] | None = None

    @staticmethod
    def distance(left: str, right: str) -> int:
        return (int(left, 16) ^ int(right, 16)).bit_count()

    def add(self, hash_value: str, record: dict[str, Any]) -> None:
        if self.root is None:
            self.root = (0, hash_value, [record], {})
            return
        node = self.root
        while True:
            distance = self.distance(hash_value, node[1])
            if distance == 0:
                node[2].append(record)
                return
            child = node[3].get(distance)
            if child is None:
                node[3][distance] = [len(node[3]), hash_value, [record], {}]
                return
            node = child

    def query(self, hash_value: str, threshold: int) -> list[tuple[int, dict[str, Any]]]:
        matches: list[tuple[int, dict[str, Any]]] = []
        if self.root is None:
            return matches
        pending = [self.root]
        while pending:
            node = pending.pop()
            distance = self.distance(hash_value, node[1])
            if distance <= threshold:
                matches.extend((distance, record) for record in node[2])
            lower = max(1, distance - threshold)
            upper = distance + threshold
            pending.extend(child for edge, child in node[3].items() if lower <= edge <= upper)
        return matches


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def class_names(records: list[dict[str, Any]]) -> list[str]:
    return sorted({record["class_name"] for record in records})


def build_review(summary: dict[str, Any], exact_groups: list[list[dict[str, Any]]], near_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Duplicate Detection Review",
        "",
        "This report is informational only. The original dataset was read-only: no image was deleted, moved, renamed, or modified.",
        "",
        "## Method",
        "",
        f"- Scanned **{summary['total_images_scanned']:,}** supported images recursively across **{summary['class_folders']}** class folders.",
        "- Exact duplicates use SHA-256 of the original file bytes.",
        f"- Near duplicates use pHash with a configurable Hamming-distance threshold of **{summary['near_duplicate_threshold']}**.",
        "- Near results are pair-groups, so each reported distance remains directly interpretable; they are review flags, not removal decisions.",
        "",
        "## Results",
        "",
        f"- Exact duplicate groups: **{summary['exact_duplicate_groups']}** ({summary['exact_duplicate_images']} duplicate images beyond one representative per group).",
        f"- Near duplicate groups: **{summary['near_duplicate_groups']}** ({summary['near_duplicate_images']} unique images involved).",
        f"- Cross-class exact groups: **{summary['cross_class_exact_duplicate_groups']}**.",
        f"- Cross-class near groups: **{summary['cross_class_near_duplicate_groups']}**.",
        f"- Images with hashing errors: **{summary['hashing_errors']}**.",
        "",
        "## Classes Involved",
        "",
        f"- Exact duplicate classes: {', '.join(summary['exact_duplicate_classes']) or 'None'}",
        f"- Near duplicate classes: {', '.join(summary['near_duplicate_classes']) or 'None'}",
        "",
        "## Cross-Class Groups",
        "",
    ]
    cross_exact = [group for group in exact_groups if len(class_names(group)) > 1]
    if cross_exact:
        for index, group in enumerate(cross_exact, start=1):
            lines.append(f"- `EXACT-{index:04d}` ({', '.join(class_names(group))}): " + "; ".join(record["file_path"] for record in group))
    else:
        lines.append("None detected.")
    cross_near = [row for row in near_rows if row["class_name"] != row["paired_class_name"]]
    if cross_near:
        lines.extend(["", "Near duplicate pairs crossing classes:"])
        for row in cross_near:
            lines.append(f"- `{row['duplicate_group_id']}` distance {row['perceptual_distance']}: {row['file_path']} ({row['class_name']}) <> {row['paired_file_path']} ({row['paired_class_name']})")
    lines.extend(["", "## Manual Review Required", "", "- Every near-duplicate pair requires manual review before any cleaning decision.", "- Every cross-class exact or near group requires manual label/provenance review.", "- No duplicate was automatically removed or excluded by this script."])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if args.threshold < 0 or args.threshold > 64:
        raise SystemExit("--threshold must be between 0 and 64 for a 64-bit pHash")
    dataset_path = args.dataset_path.expanduser().resolve()
    if not dataset_path.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {dataset_path}")

    images = collect_images(dataset_path)
    print(f"Scanning {len(images):,} images with pHash threshold {args.threshold}...")
    exact_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    errors: list[dict[str, str]] = []
    near_index = PhashIndex()
    near_rows: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for index, (class_name, path) in enumerate(images, start=1):
        file_path = relative_path(path, dataset_path)
        try:
            sha256, perceptual_hash = calculate_hashes(path)
        except (OSError, SyntaxError, UnidentifiedImageError, ValueError) as exc:
            errors.append({"file_path": file_path, "error": str(exc)})
            continue
        record = {"class_name": class_name, "file_path": file_path, "hash": sha256, "perceptual_hash": perceptual_hash}
        exact_index[sha256].append(record)
        for distance, previous in near_index.query(perceptual_hash, args.threshold):
            pair = tuple(sorted((file_path, previous["file_path"])))
            if pair in seen_pairs or sha256 == previous["hash"]:
                continue
            seen_pairs.add(pair)
            near_rows.extend([
                {"duplicate_group_id": f"NEAR-{len(seen_pairs):04d}", "duplicate_type": "near", **record, "perceptual_distance": distance, "paired_file_path": previous["file_path"], "paired_class_name": previous["class_name"]},
                {"duplicate_group_id": f"NEAR-{len(seen_pairs):04d}", "duplicate_type": "near", **previous, "perceptual_distance": distance, "paired_file_path": file_path, "paired_class_name": class_name},
            ])
        near_index.add(perceptual_hash, record)
        if index % PROGRESS_INTERVAL == 0 or index == len(images):
            print(f"  processed {index:,}/{len(images):,} images; errors: {len(errors)}")

    exact_groups = [group for group in exact_index.values() if len(group) > 1]
    exact_rows = []
    for group_index, group in enumerate(sorted(exact_groups, key=lambda group: group[0]["file_path"]), start=1):
        for record in group:
            exact_rows.append({"duplicate_group_id": f"EXACT-{group_index:04d}", "duplicate_type": "exact", **record, "perceptual_distance": ""})
    near_group_ids = {row["duplicate_group_id"] for row in near_rows}
    exact_cross = [group for group in exact_groups if len(class_names(group)) > 1]
    near_cross = {row["duplicate_group_id"] for row in near_rows if row["class_name"] != row["paired_class_name"]}
    exact_images = sum(len(group) - 1 for group in exact_groups)
    near_images = len({row["file_path"] for row in near_rows})
    summary = {
        "dataset_path": str(dataset_path), "class_folders": len({class_name for class_name, _ in images}), "total_images_scanned": len(images),
        "successfully_hashed_images": len(images) - len(errors), "hashing_errors": len(errors),
        "exact_duplicate_groups": len(exact_groups), "exact_duplicate_images": exact_images,
        "near_duplicate_groups": len(near_group_ids), "near_duplicate_images": near_images,
        "cross_class_duplicate_groups": len(exact_cross) + len(near_cross), "within_class_duplicate_groups": len(exact_groups) - len(exact_cross) + len(near_group_ids) - len(near_cross),
        "cross_class_exact_duplicate_groups": len(exact_cross), "cross_class_near_duplicate_groups": len(near_cross),
        "near_duplicate_threshold": args.threshold, "near_duplicate_method": "imagehash.phash, 64-bit hash, Hamming distance; pair-groups",
        "exact_duplicate_classes": sorted({name for group in exact_groups for name in class_names(group)}),
        "near_duplicate_classes": sorted({row["class_name"] for row in near_rows}), "errors": errors, "read_only": True,
    }
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    write_csv(REPORTS_PATH / "exact_duplicates.csv", exact_rows)
    write_csv(REPORTS_PATH / "near_duplicates.csv", [{key: row.get(key, "") for key in CSV_FIELDS} for row in near_rows])
    (REPORTS_PATH / "duplicate_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (REPORTS_PATH / "duplicate_review.md").write_text(build_review(summary, exact_groups, near_rows), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("total_images_scanned", "exact_duplicate_groups", "exact_duplicate_images", "near_duplicate_groups", "near_duplicate_images", "cross_class_duplicate_groups")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())