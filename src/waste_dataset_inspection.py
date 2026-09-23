"""Inspect the second-hand clothing dataset before any waste-model training."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import ClassLabel, load_dataset

DATASET_ID = "wargoninnovation/clothingdatasetsecondhand"
IMAGE_COLUMN = "image"
LABEL_COLUMN = "usage"
LABEL_NORMALIZATION = {
    "Export": "Export",
    "export": "Export",
    "Reuse": "Reuse",
    "reuse": "Reuse",
    "Recycle": "Recycle",
    "recycle": "Recycle",
    "Rcycle": "Recycle",
    "Energy recovery": "Energy Recovery",
    "Remake": "Remake",
    "Repair": "Repair",
}
NORMALIZED_CLASSES = ["Export", "Reuse", "Recycle", "Energy Recovery", "Remake", "Repair"]
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "models" / "waste_classification" / "waste_dataset_inspection.json"


def _label_candidates(features: dict[str, Any]) -> list[str]:
    names = []
    for name, feature in features.items():
        lowered = name.lower()
        if isinstance(feature, ClassLabel) or any(token in lowered for token in ("label", "categor", "class", "waste", "end_of_life", "usage", "disposal", "recovery")):
            names.append(name)
    return sorted(names, key=lambda name: (0 if name.lower() in {"usage", "waste_category", "end_of_life", "disposal", "recovery"} else 1, name))


def inspect_dataset(streaming: bool = True) -> dict[str, Any]:
    dataset = load_dataset(DATASET_ID, streaming=streaming)
    report: dict[str, Any] = {"dataset": DATASET_ID, "image_column": IMAGE_COLUMN, "target_column": LABEL_COLUMN, "raw_labels": [], "normalized_labels": NORMALIZED_CLASSES, "splits": {}}
    first_split = next(iter(dataset))
    first_features = dataset[first_split].features
    image_columns = [name for name in first_features if "image" in name.lower()]
    label_candidates = _label_candidates(first_features)
    if IMAGE_COLUMN not in first_features:
        raise RuntimeError(f"Required image feature {IMAGE_COLUMN!r} was not found")
    if LABEL_COLUMN not in first_features:
        raise RuntimeError(f"Required target column {LABEL_COLUMN!r} was not found")
    raw_labels: set[str] = set()

    for split_name, split in dataset.items():
        features = split.features
        counts: Counter[str] = Counter()
        label_column = LABEL_COLUMN
        samples = 0
        for row in split:
            samples += 1
            if label_column:
                value = row[label_column]
                if isinstance(features[label_column], ClassLabel):
                    value = features[label_column].int2str(value)
                raw_value = str(value)
                raw_labels.add(raw_value)
                normalized_value = LABEL_NORMALIZATION.get(raw_value)
                if normalized_value is None:
                    counts[f"UNEXPECTED: {raw_value}"] += 1
                else:
                    counts[normalized_value] += 1
        report["splits"][split_name] = {
            "num_samples": samples,
            "columns": list(features),
            "features": {name: str(feature) for name, feature in features.items()},
            "label_column": label_column,
            "raw_labels": sorted(raw_labels),
            "unique_labels": sorted(counts),
            "class_distribution": dict(sorted(counts.items())),
        }
    report["raw_labels"] = sorted(raw_labels)
    report["unexpected_labels"] = sorted(label for label in raw_labels if label not in LABEL_NORMALIZATION)
    report["train_samples"] = report["splits"].get("train", {}).get("num_samples", 0)
    report["test_samples"] = report["splits"].get("test", {}).get("num_samples", 0)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the waste dataset without training.")
    parser.add_argument("--materialize", action="store_true", help="Use regular loading instead of streaming.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    report = inspect_dataset(streaming=not args.materialize)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()