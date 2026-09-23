"""[DEPRECATED] Training configuration for MobileNetV3/sklearn models.

DEPRECATION STATUS: This module is no longer used by the active EfficientNet-B0 inference pipeline.
It is preserved for historical reference and comparison only.

ACTIVE INFERENCE MODEL: EfficientNet-B0 (PyTorch) — see Backend/app/services/fabric_classifier.py
"""
from dataclasses import dataclass, field
from typing import Tuple
import os

try:
    from Backend.material_classes import MODEL_CLASSES
except ImportError:
    from material_classes import MODEL_CLASSES


@dataclass
class TrainingConfig:
    model_name: str = "MobileNetV3 Material Classifier"
    model_version: str = "material-mobilenetv3-v0.4-5class-ibug"
    architecture: str = "mobilenet_v3_small"
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 0.0001
    input_shape: Tuple[int, int, int] = (224, 224, 3)
    head_epochs: int = 15
    finetune_epochs: int = 25
    early_stopping_patience: int = 6
    fine_tune_at: int = 60
    finetune_learning_rate: float = 0.00002
    reduce_lr_factor: float = 0.5
    reduce_lr_patience: int = 2
    reduce_lr_min_lr: float = 1e-7
    label_smoothing: float = 0.05
    dropout_rate: float = 0.4
    dataset_name: str = "ibug_material_v2"
    random_seed: int = 42
    use_class_weights: bool = True
    use_augmentation: bool = True
    label_provenance: str = "manufacturer tag.txt composition metadata (iBUG)"
    # Full application taxonomy — NOT what the current model trains on
    target_classes: Tuple[str, ...] = (
        "Cotton", "Polyester", "Wool", "Silk", "Linen", "Denim",
        "Nylon", "Rayon", "Acrylic", "Mixed Fabrics",
    )
    current_model_classes: Tuple[str, ...] = MODEL_CLASSES


@dataclass
class InferenceConfig:
    high_confidence_threshold: float = 0.75
    medium_confidence_threshold: float = 0.55
    low_confidence_threshold: float = 0.55
    manual_review_threshold: float = 0.55
    reviewable_message: str = (
        "Prediction is reviewable — verify against a second indicator before acting."
    )
    low_confidence_message: str = "Low-confidence prediction — manual inspection recommended."
    unknown_class_label: str = "UNKNOWN / UNSUPPORTED"
    supported_classes: Tuple[str, ...] = MODEL_CLASSES

    @classmethod
    def from_metadata(cls, metrics: dict | None):
        """Validation-calibrated thresholds from the trained model's metrics.

        Falls back to yaml defaults when the model carries no calibration block
        (e.g. older models).
        """
        calibration = (metrics or {}).get("calibration")
        if not isinstance(calibration, dict):
            return cls.from_yaml()
        try:
            high = float(calibration["high_confidence_threshold"])
            medium = float(calibration["medium_confidence_threshold"])
            manual = float(
                calibration.get("manual_review_threshold", medium)
            )
            if not (0.0 < manual <= medium <= high < 1.0):
                raise ValueError("calibration thresholds out of order")
            return cls(
                high_confidence_threshold=high,
                medium_confidence_threshold=medium,
                low_confidence_threshold=manual,
                manual_review_threshold=manual,
            )
        except (KeyError, TypeError, ValueError):
            return cls.from_yaml()

    @classmethod
    def from_yaml(cls, path: str | None = None):
        import yaml
        from app.ai.datasets.paths import BACKEND_ROOT

        config_path = path or str(BACKEND_ROOT / "config" / "inference.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
            return cls(
                high_confidence_threshold=float(data.get("high_confidence_threshold", 0.75)),
                medium_confidence_threshold=float(data.get("medium_confidence_threshold", 0.55)),
                low_confidence_threshold=float(data.get("low_confidence_threshold", 0.55)),
                manual_review_threshold=float(data.get("manual_review_threshold", 0.55)),
                low_confidence_message=data.get(
                    "low_confidence_message",
                    "Low-confidence prediction — manual inspection recommended.",
                ),
                unknown_class_label=data.get("unknown_class_label", "UNKNOWN / UNSUPPORTED"),
                supported_classes=tuple(data.get("supported_classes", MODEL_CLASSES)),
            )
        return cls()

    def confidence_status(self, probability: float) -> str:
        """Map softmax probability to a validated tier (thresholds come from
        model calibration when available — never guessed)."""
        if probability >= self.high_confidence_threshold:
            return "HIGH_CONFIDENCE"
        if probability >= self.medium_confidence_threshold:
            return "MEDIUM_CONFIDENCE"
        return "LOW_CONFIDENCE"
