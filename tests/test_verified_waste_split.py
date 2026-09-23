import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prepare_waste_train_validation_split import NORMALIZED_CLASSES, prepare_verified_waste_split


class MockDataset(list):
    column_names = ["image", "usage"]


def test_prepare_verified_waste_split_uses_training_only_and_stratifies():
    labels = [
        "Export", "Reuse", "Recycle", "Energy recovery", "Remake", "Repair",
        "Export", "Reuse", "Recycle", "Energy recovery", "Remake", "Repair",
    ] * 4
    train = MockDataset([{"image": f"img-{i}", "usage": label} for i, label in enumerate(labels)])
    test = MockDataset([{"image": f"test-{i}", "usage": "Recycle"} for i in range(12)])

    manifest = prepare_verified_waste_split(
        train,
        test,
        validation_fraction=0.25,
        seed=7,
        require_original_sizes=False,
    )

    assert manifest["source_split"] == "train"
    assert manifest["test_used_for_split"] is False
    assert manifest["original_test_untouched"] is True
    assert set(manifest["train_distribution"]) == set(manifest["classes"])
    assert set(manifest["validation_distribution"]) == set(manifest["classes"])
    assert set(manifest["train_distribution"]) == set(NORMALIZED_CLASSES)
    assert manifest["train_samples"] + manifest["validation_samples"] == len(train)
    assert manifest["original_test_samples"] == len(test)
    assert manifest["train_samples"] > 0
    assert manifest["validation_samples"] > 0
    assert math.isclose(sum(manifest["train_distribution"].values()), manifest["train_samples"])
    assert math.isclose(sum(manifest["validation_distribution"].values()), manifest["validation_samples"])
