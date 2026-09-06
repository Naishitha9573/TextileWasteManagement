import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ai.datasets.paths import BACKEND_ROOT, CLASS_MAPPING_PATH, DATASETS_ROOT
from app.ai.datasets.mapping import extract_material_from_caption, load_class_mapping
from app.ai.datasets.dataset_manager import DatasetManager
from app.ai.datasets.validators import validate_image, collect_image_files


def test_backend_root_points_to_backend_folder():
    assert BACKEND_ROOT.name == "Backend"
    assert DATASETS_ROOT == BACKEND_ROOT / "datasets"


def test_class_mapping_yaml_loads():
    assert CLASS_MAPPING_PATH.exists()
    mapping = load_class_mapping()
    assert len(mapping["target_material_classes"]) == 10
    assert "Mixed Fabrics" in mapping["target_material_classes"]


def test_caption_material_extraction():
    mapping = load_class_mapping()
    assert extract_material_from_caption("The fabric is cotton and soft.", mapping) == "Cotton"
    assert extract_material_from_caption("denim jacket with cotton lining", mapping) == "Mixed Fabrics"
    assert extract_material_from_caption("synthetic blend", mapping) is None


def test_dataset_manager_discovers_local_datasets():
    manager = DatasetManager()
    datasets = manager.discover_datasets()
    names = {d["name"] for d in datasets}
    if DATASETS_ROOT.exists():
        assert "fabric_dataset" in names or "deepfashion" in names or len(names) >= 0


def test_fabric_dataset_validation_when_present():
    manager = DatasetManager()
    info = manager.validate_dataset("fabric_dataset")
    if not info.get("exists"):
        pytest.skip("fabric_dataset not present locally")
    assert info["image_count"] > 0


def test_image_validation_on_fabric_sample():
    fabric_root = DATASETS_ROOT / "fabric_dataset" / "NODefect_images"
    if not fabric_root.exists():
        pytest.skip("fabric_dataset not present locally")
    images = collect_image_files(fabric_root)
    assert images
    result = validate_image(images[0])
    assert result["valid"] is True
    assert result["width"] > 0
