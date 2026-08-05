import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
import numpy as np


class DatasetManager:
    def __init__(self, root_dir: Optional[str] = None):
        self.root_dir = Path(root_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets"))
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def discover_datasets(self) -> List[Dict[str, object]]:
        discovered = []
        for name in ["tips", "deepfashion", "fashion_mnist", "fabric_dataset", "sustainable_dataset"]:
            path = self.root_dir / name
            if path.exists():
                discovered.append({
                    "name": name,
                    "path": str(path),
                    "exists": True,
                    "file_count": len(list(path.rglob("*"))) if path.exists() else 0,
                })
        return discovered

    def validate_dataset(self, dataset_name: str) -> Dict[str, object]:
        path = self.root_dir / dataset_name
        if not path.exists():
            return {"name": dataset_name, "exists": False, "issues": ["dataset directory missing"]}

        issues = []
        image_files = list(path.rglob("*.jpg")) + list(path.rglob("*.jpeg")) + list(path.rglob("*.png"))
        if not image_files:
            issues.append("no image files found")
        return {"name": dataset_name, "exists": True, "image_count": len(image_files), "issues": issues}

    def build_dataset_index(self) -> Dict[str, object]:
        datasets = self.discover_datasets()
        validations = {item["name"]: self.validate_dataset(item["name"]) for item in datasets}
        return {"datasets": datasets, "validations": validations}

    def create_training_split(self, dataset_name: str, output_dir: Optional[str] = None) -> Dict[str, object]:
        output_path = Path(output_dir or self.root_dir / dataset_name / "processed")
        output_path.mkdir(parents=True, exist_ok=True)
        return {"dataset": dataset_name, "output_dir": str(output_path), "status": "prepared"}

    def summarize(self) -> Dict[str, object]:
        return {"datasets": self.discover_datasets(), "ready": True}
