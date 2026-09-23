"""Validate the real training datasets: readability, class coverage, leakage.

Usage (from Backend/):
    python -m ml.datasets.validate_dataset
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ml.datasets.common import (
    PROCESSED_DIR,
    REPORTS_DIR,
    SPLITS_DIR,
    TRAINING_DATASETS,
    compute_hashes,
    iter_images,
)


def check_split_leakage(split_root: Path) -> dict:
    """sha256-based cross-split duplicate detection (content leakage)."""
    hash_to_splits = defaultdict(set)
    counts = {}
    unreadable = 0
    for split in ("train", "validation", "test"):
        split_dir = split_root / split
        if not split_dir.exists():
            counts[split] = 0
            continue
        images = list(iter_images(split_dir))
        counts[split] = len(images)
        hashes = compute_hashes(images)
        for path_str, entry in hashes.items():
            if "error" in entry:
                unreadable += 1
                continue
            hash_to_splits[entry["sha256"]].add(split)

    leaks = {h: sorted(s) for h, s in hash_to_splits.items() if len(s) > 1}
    return {
        "split_image_counts": counts,
        "cross_split_duplicate_hashes": len(leaks),
        "leak_examples": list(leaks.items())[:5],
        "unreadable_images": unreadable,
        "leakage_detected": bool(leaks),
    }


def main() -> int:
    results = {}

    for name, meta in TRAINING_DATASETS.items():
        print(f"=== {name} ===")
        processed = PROCESSED_DIR / name
        entry: dict = {"label_source": meta["label_source"]}

        if processed.exists():
            images = list(iter_images(processed))
            entry["processed_images"] = len(images)
            bad = []
            hashes = compute_hashes(images)
            exact_groups = defaultdict(list)
            for path_str, info in hashes.items():
                if "error" in info:
                    bad.append(path_str)
                    continue
                exact_groups[info["sha256"]].append(path_str)
            entry["unreadable_processed_images"] = len(bad)
            entry["processed_exact_duplicate_groups"] = {
                h: [Path(p).name for p in paths]
                for h, paths in exact_groups.items()
                if len(paths) > 1
            }

        split_root = SPLITS_DIR / name
        if split_root.exists():
            entry["splits"] = check_split_leakage(split_root)
            per_split_class_counts = {}
            for split in ("train", "validation", "test"):
                split_dir = split_root / split
                if split_dir.exists():
                    classes = sorted({p.parent.name for p in iter_images(split_dir)})
                    per_split_class_counts[split] = classes
            entry["classes_per_split"] = per_split_class_counts

        results[name] = entry
        print(json.dumps(entry, indent=2)[:2000])

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "dataset_validation.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
