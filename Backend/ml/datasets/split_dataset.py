"""Verify group-aware split integrity (read-only).

Checks that all views of the same physical sample stay in one split and that
no sample group appears in two splits, for every dataset with a group-aware
split under data/splits/.

Usage (from Backend/):
    python -m ml.datasets.split_dataset
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from ml.datasets.common import REPORTS_DIR, SPLITS_DIR, count_by_class, iter_images

IBUG_NAME_RE = re.compile(r"^sample-(?P<folder>.+)-(?P<sample>\d+)-(?P<view>im_\d+)\.(png|jpg)$", re.IGNORECASE)
DEEPFASHION_GARMENT_RE = re.compile(r"(?P<id>\d{8}-\d{2})", re.IGNORECASE)

# Datasets whose images are independent captures with no multi-view sample grouping.
GROUP_POLICY_NOT_APPLICABLE = {"fabric_condition"}


def sample_key(dataset: str, path: Path) -> str | None:
    name = path.name
    if dataset.startswith("ibug"):
        m = IBUG_NAME_RE.match(name)
        if m:
            return f"ibug:{m.group('folder')}/{m.group('sample')}"
        return None
    if "deepfashion" in dataset:
        m = DEEPFASHION_GARMENT_RE.search(name)
        if m:
            return f"deepfashion:{m.group('id')}"
        return None
    return None


def main() -> int:
    results = {}
    overall_ok = True
    for split_root in sorted(d for d in SPLITS_DIR.iterdir() if d.is_dir()):
        dataset = split_root.name
        if dataset in GROUP_POLICY_NOT_APPLICABLE:
            results[dataset] = {
                "group_aware_split_valid": True,
                "policy": "not_applicable (independent captures, no multi-view sample grouping)",
            }
            print(f"{dataset}: group-awareness not applicable")
            continue
        group_to_splits = defaultdict(set)
        unparseable = 0
        total = 0
        for split in ("train", "validation", "test"):
            split_dir = split_root / split
            if not split_dir.exists():
                continue
            for img in iter_images(split_dir):
                total += 1
                key = sample_key(dataset, img)
                if key is None:
                    unparseable += 1
                else:
                    group_to_splits[key].add(split)

        leaks = {k: sorted(s) for k, s in group_to_splits.items() if len(s) > 1}
        per_split = {}
        for split in ("train", "validation", "test"):
            split_dir = split_root / split
            if split_dir.exists():
                per_split[split] = count_by_class(split_dir)
        ok = (not leaks) and unparseable == 0
        overall_ok = overall_ok and ok
        results[dataset] = {
            "total_images": total,
            "unique_sample_groups": len(group_to_splits),
            "unparseable_names": unparseable,
            "cross_split_group_leakage": leaks,
            "leakage_detected": bool(leaks),
            "per_split_class_counts": per_split,
            "group_aware_split_valid": ok,
        }
        print(f"{dataset}: groups={len(group_to_splits)}, leakage={bool(leaks)}, valid={ok}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "split_integrity.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print("ALL_GROUP_AWARE_SPLITS_VALID =", overall_ok)
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
