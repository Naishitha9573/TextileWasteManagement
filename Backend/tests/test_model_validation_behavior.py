import io
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms import analyze_image, get_recycling_recommendations
from app.ai.inference_service import MaterialInferenceService
from app.ai.training_config import InferenceConfig
from app.services.recommendation_engine import RecommendationEngine


def _image(color):
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _service_for_probabilities(threshold=0.55):
    service = MaterialInferenceService.__new__(MaterialInferenceService)
    service.class_names = ["Cotton", "Nylon", "Polyester", "Silk", "Wool"]
    service.metadata = {"model_version": "material-mobilenetv3-v0.2-5class-ibug", "metrics": {}}
    service.backend = "test"
    service.inference_config = InferenceConfig(manual_review_threshold=threshold)
    return service


def test_known_supported_classes_are_returned_for_strong_scores():
    service = _service_for_probabilities()
    for index, expected in enumerate(("Cotton", "Nylon", "Polyester", "Silk", "Wool")):
        probabilities = np.full(5, 0.025)
        probabilities[index] = 0.90
        result = service._format_result(index, 0.90, probabilities)
        assert result["material"] == expected
        assert result["supported_classes"] == ["Cotton", "Nylon", "Polyester", "Silk", "Wool"]
        assert result["unknown_or_unsupported"] is False


def test_low_confidence_and_outside_taxonomy_use_unknown():
    service = _service_for_probabilities(threshold=0.75)
    result = service._format_result(0, 0.53, np.array([0.53, 0.30, 0.08, 0.07, 0.05]))
    assert result["material"] == "UNKNOWN / UNSUPPORTED"
    assert result["unknown_or_unsupported"] is True
    assert "Unable to reliably identify" in result["warning"]


def test_purple_color_is_pixel_based_and_not_material_based():
    result = analyze_image(_image((90, 20, 120)), "purple.png")
    assert result["color_name"] == "Purple"
    assert result["color_hex"] == "#5a1478"
    assert result["color_confidence"] > 0
    assert "Denim" not in result["color_name"]


def test_unknown_material_recommendation_is_conservative():
    result = get_recycling_recommendations("UNKNOWN / UNSUPPORTED", "Reusable")
    assert result["strategy"] == "Manual material verification"
    assert result["confidence"] == 0.0


def test_recommendation_confidence_uses_independent_evidence_inputs():
    engine = RecommendationEngine()
    strong = engine.build_recommendation("Cotton", "Good", "Reusable", False, False, 90, 80, 1.0, 1.0, 1.0)
    weak = engine.build_recommendation("Cotton", "Good", "Reusable", False, False, 90, 80, 0.2, 0.5, 0.5)
    assert weak["confidence"] < strong["confidence"]
    assert weak["confidence_inputs"]["material_confidence"] == 0.2
