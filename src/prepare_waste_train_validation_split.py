"""Create a reproducible train-only 85/15 split for the waste classifier."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import ClassLabel, load_dataset
from sklearn.model_selection import train_test_split

try:
    from train_waste_classifier import IMAGE_COLUMN, LABEL_COLUMN, LABEL_NORMALIZATION, NORMALIZED_CLASSES
except ImportError:
    from .train_waste_classifier import IMAGE_COLUMN, LABEL_COLUMN, LABEL_NORMALIZATION, NORMALIZED_CLASSES

DATASET_ID = "wargoninnovation/clothingdatasetsecondhand"
ROOT = Path(__file__).resolve().parents[1]
INSPECTION_PATH = ROOT / "models" / "waste_classification" / "waste_dataset_inspection.json"
SPLIT_PATH = ROOT / "models" / "waste_classification" / "waste_train_validation_split.json"
SEED = 42


def prepare_verified_waste_split(
    train_split,
    test_split,
    *,
    image_column: str = IMAGE_COLUMN,
    target_column: str = LABEL_COLUMN,
    validation_fraction: float = 0.15,
    seed: int = SEED,
    output_path: str | Path | None = None,
    require_original_sizes: bool = True,
):
    """Validate labels, protect the untouched original test split, and create a train/validation split only from train data."""
    if hasattr(train_split, "column_names"):
        column_names = list(train_split.column_names)
    else:
        rows = list(train_split)
        if not rows:
            raise ValueError("Train split is empty")
        column_names = list(rows[0].keys())
        train_rows = rows
    if image_column not in column_names:
        raise ValueError(f"Missing image feature: {image_column}")
    if target_column not in column_names:
        raise ValueError(f"Missing target column: {target_column}")

    if not hasattr(train_split, "column_names"):
        train_rows = list(train_split)
    else:
        train_rows = list(train_split)

    normalized_labels = []
    for row in train_rows:
        raw_value = row[target_column]
        if hasattr(raw_value, "item"):
            raw_value = raw_value.item()
        label = LABEL_NORMALIZATION.get(str(raw_value))
        if label is None:
            raise ValueError(f"Unexpected raw label in train split: {raw_value!r}")
        normalized_labels.append(label)

    unique_labels = set(normalized_labels)
    if unique_labels != set(NORMALIZED_CLASSES):
        missing = sorted(set(NORMALIZED_CLASSES) - unique_labels)
        unexpected = sorted(unique_labels - set(NORMALIZED_CLASSES))
        raise ValueError(
            f"Unexpected class set. Expected {NORMALIZED_CLASSES}, got {sorted(unique_labels)}; "
            f"missing={missing}, unexpected={unexpected}"
        )

    train_count = len(train_rows)
    test_rows = list(test_split) if test_split is not None else []
    test_count = len(test_rows)
    if require_original_sizes and (train_count != 30192 or test_count != 12940):
        raise ValueError(
            f"Unexpected split sizes: train={train_count}, test={test_count}. Original split counts must remain unchanged."
        )

    train_idx, val_idx = train_test_split(
        list(range(train_count)),
        test_size=validation_fraction,
        random_state=seed,
        stratify=normalized_labels,
    )
    train_labels = [normalized_labels[i] for i in train_idx]
    val_labels = [normalized_labels[i] for i in val_idx]
    train_distribution = {name: train_labels.count(name) for name in NORMALIZED_CLASSES}
    validation_distribution = {name: val_labels.count(name) for name in NORMALIZED_CLASSES}

    if test_split is not None:
        test_labels = []
        for row in test_rows:
            raw_value = row[target_column]
            if hasattr(raw_value, "item"):
                raw_value = raw_value.item()
            normalized = LABEL_NORMALIZATION.get(str(raw_value))
            if normalized is None:
                raise ValueError(f"Unexpected raw label in test split: {raw_value!r}")
            test_labels.append(normalized)
        if len(test_labels) != test_count:
            raise ValueError("Original test set length changed unexpectedly.")
        if set(test_labels) - set(NORMALIZED_CLASSES):
            raise ValueError("Original test set includes unexpected labels.")

    payload = {
        "dataset": DATASET_ID,
        "image_column": image_column,
        "target_column": target_column,
        "classes": NORMALIZED_CLASSES,
        "seed": seed,
        "source_split": "train",
        "source_train_samples": train_count,
        "train_samples": len(train_idx),
        "validation_samples": len(val_idx),
        "original_test_samples": test_count,
        "original_test_untouched": True,
        "test_used_for_split": False,
        "train_distribution": train_distribution,
        "validation_distribution": validation_distribution,
        "indices": {"train": sorted(train_idx), "validation": sorted(val_idx)},
    }

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return payload


def main() -> int:
    inspection = json.loads(INSPECTION_PATH.read_text(encoding="utf-8"))
    expected_classes = NORMALIZED_CLASSES
    if inspection.get("image_column") != IMAGE_COLUMN or inspection.get("target_column") != LABEL_COLUMN:
        raise RuntimeError("Inspection report does not match the verified image/target columns")
    if inspection.get("normalized_labels") != expected_classes or inspection.get("unexpected_labels") != []:
        raise RuntimeError("Inspection report does not contain exactly the verified normalized classes")
    if inspection.get("train_samples") != 30192 or inspection.get("test_samples") != 12940:
        raise RuntimeError("Inspection report sample counts do not match the original dataset")

    original_train = load_dataset(DATASET_ID, split="train", streaming=True)
    if IMAGE_COLUMN not in original_train.column_names or LABEL_COLUMN not in original_train.column_names:
        raise RuntimeError("Verified image or target column is missing from the original train split")
    original_test = load_dataset(DATASET_ID, split="test", streaming=True)
    manifest = prepare_verified_waste_split(
        original_train,
        original_test,
        validation_fraction=0.15,
        seed=SEED,
        output_path=SPLIT_PATH,
    )

    print("WASTE DATASET READY FOR TRAINING")
    print(f"Image feature: {IMAGE_COLUMN}")
    print(f"Target: {LABEL_COLUMN}")
    print(f"Classes: {expected_classes}")
    print(f"Original train samples: {manifest['source_train_samples']}")
    print(f"Train samples (85%): {manifest['train_samples']}")
    print(f"Validation samples (15%): {manifest['validation_samples']}")
    print(f"Original test samples (untouched): {manifest['original_test_samples']}")
    print(f"Train distribution: {manifest['train_distribution']}")
    print(f"Validation distribution: {manifest['validation_distribution']}")
    print("Unexpected labels: []")
    print("Test set used for split: NO")
    print("Test set modified: NO")
    print(f"Split manifest: {SPLIT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())