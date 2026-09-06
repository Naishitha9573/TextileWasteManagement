"""Sklearn training pipeline on real prepared dataset features (no TensorFlow required)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from app.ai.dataset_manager import DatasetManager
from app.ai.evaluate import Evaluator
from app.ai.feature_extractor import load_split_feature_matrix
from app.ai.training_config import TrainingConfig
from app.ai.model_registry import ModelRegistry


class SklearnTrainingPipeline:
    def __init__(
        self,
        root_dir: Optional[str] = None,
        dataset_name: Optional[str] = None,
        config: Optional[TrainingConfig] = None,
    ):
        from app.ai.datasets.paths import BACKEND_ROOT

        self.root_dir = Path(root_dir or BACKEND_ROOT / "models")
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or TrainingConfig()
        self.config.architecture = "sklearn_random_forest"
        self.dataset_manager = DatasetManager()
        self.dataset_name = dataset_name or self.config.dataset_name
        self.registry = ModelRegistry(str(self.root_dir))
        self.evaluator = Evaluator()
        self.metrics_path = self.root_dir / "model_metrics.json"
        self.model_path = self.root_dir / "material_classifier.joblib"
        self.label_map_path = self.root_dir / "label_map.json"

    def train(self) -> Dict[str, object]:
        processed = self.dataset_manager.get_processed_dataset_path(self.dataset_name)
        if not processed.exists():
            raise RuntimeError(f"Processed dataset missing. Run: python -m app.ai.datasets.prepare")

        split_train = self.dataset_manager.get_split_path(self.dataset_name, "train")
        if not split_train.exists():
            self.dataset_manager.create_training_split(
                self.dataset_name, random_state=self.config.random_seed, use_processed=True
            )

        x_train, y_train, class_names = load_split_feature_matrix(
            self.dataset_manager.get_split_path(self.dataset_name, "train")
        )
        x_val, y_val, _ = load_split_feature_matrix(
            self.dataset_manager.get_split_path(self.dataset_name, "validation")
        )
        x_test, y_test, _ = load_split_feature_matrix(
            self.dataset_manager.get_split_path(self.dataset_name, "test")
        )

        if len(x_train) == 0:
            raise RuntimeError("No training features extracted")

        encoder = LabelEncoder()
        encoder.fit(class_names)

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=self.config.random_seed,
            class_weight="balanced",
        )
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        eval_result = self.evaluator.evaluate(y_test, y_pred, class_names)

        val_pred = model.predict(x_val)
        val_acc = float(np.mean(val_pred == y_val)) if len(y_val) else 0.0

        joblib.dump({"model": model, "label_encoder": encoder, "class_names": class_names}, self.model_path)

        label_map = {str(i): name for i, name in enumerate(class_names)}
        with open(self.label_map_path, "w", encoding="utf-8") as handle:
            json.dump(label_map, handle, indent=2)

        metrics = {
            "status": "evaluated",
            "model_name": self.config.model_name,
            "model_version": self.config.model_version,
            "architecture": "sklearn_random_forest",
            "model_path": str(self.model_path),
            "backend": "sklearn",
            "dataset": self.dataset_name,
            "dataset_version": "phase1_prepared",
            "class_names": class_names,
            "label_mapping_note": "Caption-inferred DeepFashion labels; 3 of 10 target materials.",
            "accuracy": eval_result["accuracy"],
            "precision": eval_result["precision"],
            "recall": eval_result["recall"],
            "f1_score": eval_result["f1_score"],
            "confusion_matrix": eval_result["confusion_matrix"],
            "classification_report": eval_result["classification_report"],
            "train_samples": len(x_train),
            "validation_samples": len(x_val),
            "test_samples": len(x_test),
            "validation_accuracy": round(val_acc, 4),
            "num_classes": len(class_names),
            "random_seed": self.config.random_seed,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(self.metrics_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

        payload = {
            "model_name": self.config.model_name,
            "model_version": self.config.model_version,
            "status": "trained",
            "backend": "sklearn",
            "metrics": metrics,
            "classes": class_names,
        }
        self.registry.save_metadata(payload)

        try:
            from app.ai.inference_service import MaterialInferenceService
            MaterialInferenceService.reset_instance()
        except ImportError:
            pass

        return payload


def main() -> int:
    pipeline = SklearnTrainingPipeline()
    result = pipeline.train()
    print(json.dumps({
        "status": result.get("status"),
        "backend": result.get("backend"),
        "classes": result.get("classes"),
        "accuracy": result.get("metrics", {}).get("accuracy"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
