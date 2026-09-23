"""Inspect the real datasets used by this project.

Usage (from Backend/):
    python -m ml.datasets.inspect_dataset
"""
from __future__ import annotations

import json
from collections import Counter

from ml.datasets.common import (
    PROCESSED_DIR,
    REPORTS_DIR,
    SPLITS_DIR,
    count_by_class,
    iter_images,
)


def main() -> int:
    report = {"processed": {}, "splits": {}}

    for dataset_dir in sorted(d for d in PROCESSED_DIR.iterdir() if d.is_dir()):
        images = list(iter_images(dataset_dir))
        per_class = Counter(img.parent.name for img in images)
        report["processed"][dataset_dir.name] = {
            "path": str(dataset_dir),
            "images": len(images),
            "classes": dict(sorted(per_class.items())),
        }
        print(f"[processed] {dataset_dir.name}: {len(images)} images, classes={dict(sorted(per_class.items()))}")

    for split_root in sorted(d for d in SPLITS_DIR.iterdir() if d.is_dir()):
        entry = {}
        total = 0
        for split in ("train", "validation", "test"):
            split_dir = split_root / split
            if not split_dir.exists():
                continue
            counts = count_by_class(split_dir)
            entry[split] = counts
            total += sum(counts.values())
        entry["total"] = total
        report["splits"][split_root.name] = entry
        print(f"[splits] {split_root.name}: total={total}")
        for split in ("train", "validation", "test"):
            if split in entry:
                print(f"    {split}: {entry[split]}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "dataset_inspection.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
