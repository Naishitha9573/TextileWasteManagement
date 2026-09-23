from pathlib import Path

from material_classes import MATERIAL_CLASSES, VALID_MATERIAL_CLASSES
from dataset_validator import validate_material_dataset


def test_material_class_taxonomy_is_centralized():
    assert isinstance(MATERIAL_CLASSES, list)
    assert len(MATERIAL_CLASSES) >= 3
    assert "Cotton" in MATERIAL_CLASSES
    assert "Mixed Fabrics" in MATERIAL_CLASSES


def test_dataset_validator_reports_real_dataset_state():
    report = validate_material_dataset(Path(__file__).resolve().parents[1] / "data" / "processed" / "material_deepfashion")
    assert isinstance(report, dict)
    assert "total_images" in report
    assert "class_counts" in report
    assert "valid_classes" in report
    assert "missing_classes" in report
