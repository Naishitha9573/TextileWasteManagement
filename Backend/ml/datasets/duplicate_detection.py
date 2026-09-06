"""Exact + near duplicate detection on the final training datasets.

Methodology (honest pair-level analysis):
- Exact duplicates: identical sha256 content within a dataset or across splits.
- Near-duplicate pairs: dHash Hamming distance <= 8 between DIFFERENT samples,
  reported by distance band. Pairs within the same sample (standard multi-view
  captures of one physical fabric) are counted separately as benign, because
  the group-aware split keeps every view of a sample in the same split.

Note: plain fabric textures are perceptually similar at Hamming 5-8; chained
clustering therefore merges many distinct fabrics. We report PAIRS with
distances instead of union-find clusters to avoid conflating similarity with
duplication. Cross-sample pairs at Hamming <= 4 are the suspicious band.

Read-only: reports groups; never deletes or moves source images.

Usage (from Backend/):
    python -m ml.datasets.duplicate_detection
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
    hamming_hex,
    iter_images,
)

NEAR_DUPLICATE_HAMMING_THRESHOLD = 8
SUSPICIOUS_HAMMING_THRESHOLD = 4


def sample_id_of(filename: str) -> str:
    """sample-<folder>-<id>-im_N.png -> folder/id ; else the filename itself."""
    if filename.startswith("sample-"):
        stem = Path(filename).stem
        return stem.rsplit("-im_", 1)[0]
    return filename


def pairwise_near_analysis(hashes: dict) -> dict:
    items = [(path_str, info["dhash"]) for path_str, info in hashes.items() if "dhash" in info]
    distinct = list({h for _, h in items})

    # Compare distinct hash values only; map back to paths afterwards.
    paths_by_hash = defaultdict(list)
    for path_str, h in items:
        paths_by_hash[h].append(path_str)

    similar_pairs = []  # (dist, hash_a, hash_b)
    n = len(distinct)
    ints = [int(h, 2) for h in distinct]
    print(f"  comparing {n} distinct dHash values ({n*(n-1)//2} pairs)...")
    for i in range(n):
        hi = ints[i]
        for j in range(i + 1, n):
            dist = bin(hi ^ ints[j]).count("1")
            if dist <= NEAR_DUPLICATE_HAMMING_THRESHOLD:
                similar_pairs.append((dist, distinct[i], distinct[j]))

    bands = {"1-4": [], "5-8": []}
    same_sample_pairs = 0
    cross_sample_suspicious = []
    for dist, ha, hb in similar_pairs:
        pa = paths_by_hash[ha][0]
        pb = paths_by_hash[hb][0]
        same = sample_id_of(Path(pa).name) == sample_id_of(Path(pb).name)
        band = "1-4" if dist <= SUSPICIOUS_HAMMING_THRESHOLD else "5-8"
        record = {
            "distance": dist,
            "same_sample": same,
            "a": Path(pa).name,
            "b": Path(pb).name,
        }
        if same:
            same_sample_pairs += 1
        elif band == "1-4":
            cross_sample_suspicious.append(record)
        if not same:
            bands[band].append(record)

    return {
        "distinct_dhash_values": n,
        "pairs_hamming_1_8_cross_sample": len(bands["1-4"]) + len(bands["5-8"]),
        "cross_sample_pairs_hamming_le_4": len(cross_sample_suspicious),
        "cross_sample_suspects_top20": sorted(cross_sample_suspicious, key=lambda r: r["distance"])[:20],
        "cross_sample_pairs_5_8_count": len(bands["5-8"]),
        "same_sample_view_pairs_benign": same_sample_pairs,
    }


def main() -> int:
    results = {}
    for name in TRAINING_DATASETS:
        print(f"=== {name} ===")
        processed = PROCESSED_DIR / name
        entry: dict = {}
        images = list(iter_images(processed))
        print(f"hashing {len(images)} processed images (cached after first run)...")
        hashes = compute_hashes(images)

        exact_groups = defaultdict(list)
        errors = 0
        for path_str, info in hashes.items():
            if "error" in info:
                errors += 1
                continue
            exact_groups[info["sha256"]].append(path_str)
        exact_dup_groups = [sorted(paths) for h, paths in exact_groups.items() if len(paths) > 1]

        entry["processed_images"] = len(images)
        entry["unreadable"] = errors
        entry["exact_duplicate_groups"] = [
            {"members": [Path(p).name for p in g]} for g in exact_dup_groups
        ]
        entry["exact_duplicate_extra_images"] = sum(len(g) - 1 for g in exact_dup_groups)
        entry.update(pairwise_near_analysis({k: v for k, v in hashes.items() if "error" not in v}))

        split_root = SPLITS_DIR / name
        if split_root.exists():
            split_hashes = compute_hashes(list(iter_images(split_root)))
            hash_splits = defaultdict(set)
            path_splits = defaultdict(set)
            for path_str, info in split_hashes.items():
                rel = Path(path_str).relative_to(SPLITS_DIR)
                split_of = rel.parts[0]
                if "sha256" in info:
                    hash_splits[info["sha256"]].add(split_of)
                    path_splits[info["sha256"]].add(str(rel))
            cross_exact = {h: s for h, s in hash_splits.items() if len(s) > 1}
            entry["cross_split_exact_duplicate_content_items"] = len(cross_exact)
            entry["cross_split_examples"] = sorted(path_splits[h] for h in list(cross_exact)[:5])

        results[name] = entry
        print(
            f"exact dup groups: {len(exact_dup_groups)}; "
            f"cross-sample near pairs <=4: {entry['cross_sample_pairs_hamming_le_4']}; "
            f"cross-split exact: {entry.get('cross_split_exact_duplicate_content_items')}"
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "duplicate_detection.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
