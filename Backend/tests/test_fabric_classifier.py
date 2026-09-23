"""Tests for the real EfficientNet fabric inference contract."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

from app.services.fabric_classifier import FabricClassifier


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "models" / "best_efficientnet_b0_class_aware_finetune_v1.pth"
SAMPLE = next((path for path in (ROOT / "dataset" / "processed" / "fabric_9class" / "test").rglob("*") if path.is_file()), None)


@pytest.mark.skipif(not CHECKPOINT.is_file() or SAMPLE is None, reason="trained checkpoint or test image is not available")
def test_real_efficientnet_inference() -> None:
    os.environ["FABRIC_MODEL_PATH"] = str(CHECKPOINT)
    classifier = FabricClassifier()
    assert classifier.is_ready
    assert len(classifier.class_names) == 9
    assert classifier.class_names == [
        "Cotton", "Denim", "Fleece", "Nylon", "Polyester",
        "Silk", "Terrycloth", "Viscose", "Wool",
    ]
    assert classifier.transform.transforms[0].size == 256
    assert classifier.transform.transforms[1].size == (224, 224)
    with Image.open(SAMPLE) as image:
        result = classifier.predict(image)
    probabilities = [item["confidence"] for item in result["top_predictions"]]
    assert len(result["top_predictions"]) == 3
    assert probabilities == sorted(probabilities, reverse=True)
    assert all(0.0 <= value <= 1.0 for value in probabilities)
    assert all(item["class_name"] in classifier.class_names for item in result["top_predictions"])
    assert len(result["probabilities"]) == 9
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-5
    assert result["model"]["version"] == "class_aware_finetune_v1"
    assert result["model"]["architecture"] == "EfficientNet-B0"
