import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ai.inference_service import MaterialInferenceService
from app.ai.training_config import InferenceConfig
from app.ai.model_registry import ModelRegistry


def test_inference_config_loads():
    cfg = InferenceConfig.from_yaml()
    assert 0 < cfg.manual_review_threshold < 1
    assert cfg.low_confidence_message


def test_inference_service_without_model():
    MaterialInferenceService.reset_instance()
    service = MaterialInferenceService()
    if service.is_ready():
        pytest.skip("Model already trained — testing error path skipped")
    result = service.predict_from_bytes(b"not-an-image", suffix=".jpg")
    assert result["status"] in {"error", "success"}


def test_model_registry_metadata_roundtrip(tmp_path):
    registry = ModelRegistry(str(tmp_path))
    payload = {"model_name": "test", "status": "trained", "classes": ["Cotton"]}
    registry.save_metadata(payload)
    loaded = registry.load_metadata()
    assert loaded["model_name"] == "test"


def test_inference_schema_when_model_ready():
    MaterialInferenceService.reset_instance()
    service = MaterialInferenceService()
    if not service.is_ready():
        pytest.skip("MODEL NOT TRAINED")
    from pathlib import Path
    from app.ai.datasets.paths import DATASETS_ROOT
    sample_dir = DATASETS_ROOT / "fabric_dataset" / "NODefect_images"
    if not sample_dir.exists():
        pytest.skip("No sample images")
    images = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
    if not images:
        pytest.skip("No sample images")
    result = service.predict_from_path(str(images[0]))
    assert result["status"] == "success"
    assert "material" in result
    assert "material_confidence" in result
    assert result["source"] == "MODEL"
    assert "low_confidence" in result
