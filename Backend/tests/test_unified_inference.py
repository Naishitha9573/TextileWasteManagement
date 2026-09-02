"""Unit tests for unified inference flow and provenance tags."""
import io
import pytest
from PIL import Image
from app.ai.inference_service import run_unified_analysis, MaterialInferenceService
from app.ai.deepfashion_service import get_deepfashion_status


def _create_sample_image() -> bytes:
    img = Image.new("RGB", (224, 224), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_unified_analysis_schema_and_provenance():
    img_bytes = _create_sample_image()
    result = run_unified_analysis(
        image_bytes=img_bytes,
        filename="fabric.jpg",
        fabric_type_hint="Cotton",
        condition="Good",
        quantity_kg=2.5,
    )

    assert result["status"] == "success"
    assert "material_prediction" in result
    assert "waste_classification" in result
    assert "circularity_score" in result
    assert "recommendation" in result
    assert "environmental_impact" in result

    # Check source provenance tags
    assert result["waste_classification"]["source"] == "RULE_ENGINE"
    assert result["circularity_score"]["source"] == "RULE_ENGINE"
    assert result["recommendation"]["source"] == "RULE_ENGINE"
    assert result["environmental_impact"]["source"] == "ESTIMATED"

    # Check environmental calculations
    assert result["environmental_impact"]["co2_savings_kg"] > 0
    assert result["environmental_impact"]["water_savings_liters"] > 0
    assert "disclaimer" in result["environmental_impact"]


def test_inference_service_singleton():
    svc1 = MaterialInferenceService.get_instance()
    svc2 = MaterialInferenceService.get_instance()
    assert svc1 is svc2


def test_deepfashion_reports_dataset_without_fabricating_inference():
    status = get_deepfashion_status()
    assert status["dataset_available"] is True
    assert status["image_count"] == 44096
    assert status["attributes_available"] is True
    assert status["bounding_boxes_available"] is True
    assert status["landmarks_available"] is True
    assert status["segmentation_available"] is True
    assert status["retrieval_annotations_available"] is True

    result = run_unified_analysis(image_bytes=None)
    garment = result["garment_analysis"]
    assert garment["status"] == "MODEL_NOT_READY"
    assert garment["category"] is None
    assert garment["attributes"] == []
    assert garment["confidence"] is None
