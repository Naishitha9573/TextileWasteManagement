"""Unified inference service â€” Keras or sklearn backend + end-to-end pipeline.

Runtime model status contract (represents what can actually run right now):
  AVAILABLE   â€” model artifact loaded and validated; real inference possible
  UNAVAILABLE â€” no usable model artifact / dependencies missing
  LOAD_ERROR  â€” artifact exists but failed validation/loading
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from app.ai.dataset_preprocessor import DatasetPreprocessor
from app.ai.feature_extractor import extract_features_from_bytes, extract_features_from_path
from app.ai.model_registry import ModelRegistry
from app.ai.training_config import InferenceConfig
from app.ai.confidence import build_confidence_fields
from material_classes import MODEL_CLASSES
from app.services.scoring_service import ScoringService
from app.services.recommendation_engine import RecommendationEngine

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from app.ai.model_loader import load_keras_material_model

try:
    import tensorflow  # noqa: F401
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


def _log(scope: str, message: str) -> None:
    print(f"[{scope}] {message}")


class MaterialInferenceService:
    _predictor_instance: Optional["MaterialInferenceService"] = None

    def __init__(self) -> None:
        self.preprocessor = DatasetPreprocessor()
        self.registry = ModelRegistry()
        self.inference_config: Optional[InferenceConfig] = None
        self.metadata: Dict[str, Any] = {}
        self.backend: Optional[str] = None
        self.keras_model = None
        self.sklearn_bundle = None
        self.class_names: list[str] = []
        # Runtime state â€” reflects what inference can actually do right now.
        self.runtime_status: str = "UNAVAILABLE"
        self.load_error: Optional[str] = None
        self.input_shape: Optional[tuple] = None
        self._load_model()

    @classmethod
    def get_instance(cls) -> "MaterialInferenceService":
        if cls._predictor_instance is None:
            cls._predictor_instance = cls()
        return cls._predictor_instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._predictor_instance = None

    def _expected_class_names(self) -> list[str]:
        return [str(c) for c in MODEL_CLASSES]

    def _load_model(self) -> None:
        self.metadata = {}
        self.class_names = []
        self.keras_model = None
        self.sklearn_bundle = None
        self.backend = None
        self.runtime_status = "UNAVAILABLE"
        self.load_error = None
        self.input_shape = None

        _log("MODEL", "Loading material classifier...")

        try:
            self.metadata = self.registry.load_metadata() or {}
        except Exception as exc:
            self.runtime_status = "LOAD_ERROR"
            self.load_error = "Model metadata could not be read"
            _log("MODEL", f"ERROR: metadata unreadable ({exc})")
            return

        metrics = self.metadata.get("metrics", {}) if isinstance(self.metadata.get("metrics"), dict) else {}

        # Confidence tiers come from the model's validation calibration when
        # present; yaml defaults are only a legacy fallback.
        try:
            self.inference_config = InferenceConfig.from_metadata(metrics)
        except Exception:
            self.inference_config = InferenceConfig.from_yaml()

        expected = self._expected_class_names()
        self.class_names = (
            metrics.get("class_names")
            or self.metadata.get("classes")
            or list(expected)
        )

        # Validate class mapping: exactly the supported 5-class contract.
        if [str(c) for c in self.class_names] != expected:
            self.runtime_status = "LOAD_ERROR"
            self.load_error = (
                "Model class mapping does not match the supported "
                f"{len(expected)}-class contract"
            )
            _log(
                "MODEL",
                f"ERROR: class mapping mismatch: {self.class_names} != {expected}",
            )
            return

        keras_path = metrics.get("model_path") or str(self.registry.root_dir / "fabric_classifier_model.keras")
        sklearn_path = str(self.registry.root_dir / "material_classifier.joblib")

        if TENSORFLOW_AVAILABLE and keras_path and os.path.exists(keras_path):
            try:
                _log("MODEL", "Loading MobileNetV3-Small...")
                model = load_keras_material_model(keras_path)
                self._validate_keras_model(model)
                self.keras_model = model
                self.backend = "keras"
                self.runtime_status = "AVAILABLE"
                _log("MODEL", "Model loaded successfully")
                _log("MODEL", f"Classes: {', '.join(self.class_names)}")
                _log("MODEL", f"Input shape: {self.input_shape}")
                _log("MODEL", "Output shape: validated (softmax over 5 classes)")
                return
            except Exception as exc:
                self.keras_model = None
                self.backend = None
                self.runtime_status = "LOAD_ERROR"
                self.load_error = "Model file could not be loaded"
                # Full detail stays server-side only.
                _log("MODEL", f"ERROR: Keras model failed to load/validate: {exc}")
                # Fall through so a preserved sklearn fallback can still be tried.

        elif TENSORFLOW_AVAILABLE and keras_path and not os.path.exists(keras_path):
            self.runtime_status = "UNAVAILABLE"
            self.load_error = "Trained model file not found"
            _log("MODEL", "ERROR: model file not found on disk")

        if backend_is_sklearn(self.metadata, metrics) and JOBLIB_AVAILABLE and os.path.exists(sklearn_path):
            try:
                self.sklearn_bundle = joblib.load(sklearn_path)
                bundle_classes = self.sklearn_bundle.get("class_names", self.class_names)
                if [str(c) for c in bundle_classes] != expected:
                    raise ValueError("sklearn class mapping mismatch")
                self.class_names = bundle_classes
                self.backend = "sklearn"
                self.runtime_status = "AVAILABLE"
                self.load_error = None
                _log("MODEL", "Sklearn baseline loaded successfully")
                _log("MODEL", f"Classes: {', '.join(self.class_names)}")
                return
            except Exception as exc:
                self.sklearn_bundle = None
                if self.runtime_status != "LOAD_ERROR":
                    self.runtime_status = "LOAD_ERROR"
                self.load_error = self.load_error or "Baseline model could not be loaded"
                _log("MODEL", f"ERROR: sklearn model failed to load: {exc}")

        if self.runtime_status == "UNAVAILABLE" and not self.load_error:
            self.load_error = (
                "No trained model available. Run POST /api/train or python -m app.ai.train."
            )

    def _validate_keras_model(self, model) -> None:
        """Validate input/output contract with a dry-run forward pass."""
        input_shape = getattr(model, "input_shape", None)
        if not input_shape or len(input_shape) != 4:
            raise ValueError(f"Unexpected input rank: {input_shape}")
        _, height, width, channels = input_shape
        if channels != 3:
            raise ValueError(f"Expected 3-channel RGB input, got {channels}")

        dummy = np.zeros((1, int(height), int(width), int(channels)), dtype=np.float32)
        output = model.predict(dummy, verbose=0)
        output = np.asarray(output)

        if output.ndim != 2 or output.shape[0] != 1:
            raise ValueError(f"Unexpected output shape: {output.shape}")
        if output.shape[1] != len(self.class_names):
            raise ValueError(
                f"Model outputs {output.shape[1]} classes but mapping has "
                f"{len(self.class_names)}"
            )
        if output.shape[1] != len(MODEL_CLASSES):
            raise ValueError(
                f"Expected exactly {len(MODEL_CLASSES)} output classes, got {output.shape[1]}"
            )

        total = float(output.sum())
        if not np.isfinite(output).all() or abs(total - 1.0) > 0.05:
            raise ValueError("Model output is not a valid probability distribution")

        self.input_shape = (int(height), int(width), int(channels))

    def is_ready(self) -> bool:
        return (
            self.runtime_status == "AVAILABLE"
            and (self.keras_model is not None or self.sklearn_bundle is not None)
            and bool(self.class_names)
        )

    def get_public_status(self) -> Dict[str, Any]:
        """Runtime model status safe to expose via API (no filesystem paths)."""
        metrics = self.metadata.get("metrics", {}) if isinstance(self.metadata.get("metrics"), dict) else {}
        available = self.is_ready()
        payload: Dict[str, Any] = {
            "available": available,
            "model_status": "AVAILABLE" if available else self.runtime_status,
            "model_name": self.metadata.get("model_name", "MobileNetV3 Material Classifier"),
            "version": self.metadata.get("model_version") or metrics.get("model_version"),
            "classes": list(self.class_names) if self.class_names else list(MODEL_CLASSES),
            "num_classes": len(self.class_names) if self.class_names else len(MODEL_CLASSES),
            "architecture": metrics.get("architecture"),
            "backend": self.backend,
            "current_model_scope": metrics.get("current_model_scope") or self.metadata.get("current_model_scope"),
            "test_accuracy": metrics.get("accuracy"),
            "label_provenance": metrics.get("label_provenance"),
        }
        if not available and self.load_error:
            payload["error"] = self.load_error
        return payload

    def predict_from_path(self, image_path: str) -> Dict[str, Any]:
        if self.backend == "sklearn":
            return self._predict_sklearn(features=extract_features_from_path(image_path))
        return self._predict_keras(image_path=image_path)

    def predict_from_bytes(self, image_bytes: bytes, suffix: str = ".jpg") -> Dict[str, Any]:
        if self.backend == "sklearn":
            return self._predict_sklearn(features=extract_features_from_bytes(image_bytes))

        if not image_bytes:
            return self._error("Empty upload â€” no image data received.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        try:
            return self._predict_keras(image_path=tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _predict_sklearn(self, features: np.ndarray) -> Dict[str, Any]:
        if not self.is_ready() or not self.sklearn_bundle:
            return self._error("ML model unavailable â€” no model loaded for inference.")
        try:
            model = self.sklearn_bundle["model"]
            batch = features.reshape(1, -1)
            probabilities = model.predict_proba(batch)[0]
            predicted_idx = int(np.argmax(probabilities))
            confidence_ratio = float(probabilities[predicted_idx])
            _log("INFERENCE", f"Predicted material: {self.class_names[predicted_idx]}")
            _log("INFERENCE", f"Confidence: {confidence_ratio * 100:.1f}%")
            return self._format_result(predicted_idx, confidence_ratio, probabilities)
        except Exception as exc:
            _log("INFERENCE", f"ERROR: prediction failed: {exc}")
            return self._error("Prediction failed due to an internal error.")

    def _predict_keras(self, image_path: str) -> Dict[str, Any]:
        if not self.is_ready() or self.keras_model is None:
            reason = self.load_error or "ML model unavailable"
            return self._error(reason)
        try:
            _log("INFERENCE", "Processing image...")
            image = self.preprocessor.preprocess_image(image_path)
            batch = np.expand_dims(image, axis=0)
            probabilities = self.keras_model.predict(batch, verbose=0)[0]
            predicted_idx = int(np.argmax(probabilities))
            confidence_ratio = float(probabilities[predicted_idx])
            _log("INFERENCE", f"Predicted material: {self.class_names[predicted_idx]}")
            _log("INFERENCE", f"Confidence: {confidence_ratio * 100:.1f}%")
            return self._format_result(predicted_idx, confidence_ratio, probabilities)
        except Exception as exc:
            _log("INFERENCE", f"ERROR: prediction failed: {exc}")
            return self._error("Prediction failed â€” image could not be processed by the model.")

    def _format_result(self, predicted_idx: int, confidence_ratio: float, probabilities) -> Dict[str, Any]:
        conf_fields = build_confidence_fields(confidence_ratio, self.inference_config)
        predicted_material = self.class_names[predicted_idx]
        is_unknown = confidence_ratio < self.inference_config.manual_review_threshold
        if is_unknown:
            predicted_material = self.inference_config.unknown_class_label
        all_probabilities = {
            self.class_names[i]: round(float(probabilities[i]) * 100, 1)
            for i in range(len(self.class_names))
        }
        probability_ratios = {
            self.class_names[i]: round(float(probabilities[i]), 4)
            for i in range(len(self.class_names))
        }
        metrics = self.metadata.get("metrics", {}) if isinstance(self.metadata.get("metrics"), dict) else {}

        result: Dict[str, Any] = {
            "status": "success",
            "source": "MODEL",
            "backend": self.backend,
            "material": predicted_material,
            "material_confidence": conf_fields["confidence"],
            "confidence": conf_fields["confidence"],
            "confidence_percent": conf_fields["confidence"],
            "confidence_ratio": conf_fields["confidence_ratio"],
            "confidence_threshold": conf_fields["confidence_threshold"],
            "confidence_status": conf_fields["confidence_status"],
            "manual_review_required": conf_fields["manual_review_required"],
            "confidence_note": conf_fields["confidence_note"],
            "all_probabilities": all_probabilities,
            "probabilities": all_probabilities,
            "probability_ratios": probability_ratios,
            "model_name": self.metadata.get("model_name", "material_classifier"),
            "model_version": self.metadata.get("model_version") or metrics.get("model_version", "unknown"),
            "dataset": metrics.get("dataset"),
            "num_classes": len(self.class_names),
            "classes": list(self.class_names),
            "current_model_scope": metrics.get("current_model_scope", "5 CLASS ONLY"),
            "supported_classes": list(MODEL_CLASSES),
            "model_accuracy": metrics.get("accuracy"),
            "model_test_accuracy": metrics.get("accuracy"),
            # Runtime state â€” never claim TRAINED unless it can actually run.
            "model_available": True,
            "model_status": "AVAILABLE",
            "low_confidence": conf_fields["manual_review_required"],
            "unknown_or_unsupported": is_unknown,
        }
        if is_unknown:
            result["warning"] = "Unable to reliably identify this material with the current model."
            result["recommended_action"] = "manual_inspection"
        elif conf_fields["manual_review_required"]:
            result["warning"] = self.inference_config.low_confidence_message
            result["recommended_action"] = "manual_inspection"
        return result

    def _error(self, message: str) -> Dict[str, Any]:
        return {
            "status": "error",
            "source": "MODEL",
            "message": message,
            "material": None,
            "material_confidence": None,
            "confidence": None,
            "model_available": False,
            "model_status": self.runtime_status,
        }


def backend_is_sklearn(metadata: Dict[str, Any], metrics: Dict[str, Any]) -> bool:
    return (metadata.get("backend") or metrics.get("backend")) == "sklearn"


def run_unified_analysis(
    image_bytes: Optional[bytes] = None,
    filename: str = "sample.jpg",
    fabric_type_hint: str = "Mixed Fabrics",
    condition: str = "Good",
    quantity_kg: float = 1.0,
) -> Dict[str, Any]:
    """
    Unified Inference Flow:
    Image -> Validation -> Preprocessing -> Material Model -> Material Prediction (source: MODEL)
    -> Waste Classification (source: RULE_ENGINE) -> Circularity Score (source: RULE_ENGINE)
    -> Recommendation (source: RULE_ENGINE) -> Environmental Estimate (source: ESTIMATED).
    """
    from algorithms import analyze_image, get_waste_classification, get_recycling_recommendations
    from app.services.environmental_impact_service import calculate_environmental_impact

    # 1. Computer Vision heuristics if image provided
    cv_features = {}
    if image_bytes:
        cv_features = analyze_image(image_bytes, filename)

    # 2. Material Classification from the authoritative EfficientNet-B0 service.
    from io import BytesIO
    from PIL import Image
    from app.services.fabric_classifier import get_fabric_classifier

    service = get_fabric_classifier()
    mat_pred: Dict[str, Any] = {}
    if image_bytes and service.is_ready:
        try:
            raw_pred = service.predict(Image.open(BytesIO(image_bytes)))
            prediction = raw_pred["prediction"]
            mat_pred = {
                "material": prediction["class_name"],
                "confidence": prediction["confidence"] * 100.0,
                "confidence_percent": prediction["confidence"] * 100.0,
                "confidence_ratio": prediction["confidence"],
                "confidence_status": "AVAILABLE",
                "manual_review_required": False,
                "probabilities": {name: value * 100.0 for name, value in raw_pred["probabilities"].items()},
                "probability_ratios": raw_pred["probabilities"],
                "model_name": "EfficientNet-B0",
                "model_version": None,
                "model_available": True,
                "model_status": "AVAILABLE",
                "source": "MODEL",
            }
        except Exception as exc:
            _log("INFERENCE", f"EfficientNet prediction failed: {exc}")
            mat_pred = {
                "material": None,
                "confidence": None,
                "confidence_percent": None,
                "confidence_status": "PREDICTION_ERROR",
                "manual_review_required": True,
                "probabilities": {},
                "probability_ratios": None,
                "model_name": "EfficientNet-B0",
                "model_version": None,
                "model_available": False,
                "model_status": "PREDICTION_ERROR",
                "source": "MODEL_NOT_AVAILABLE",
                "warning": "Prediction failed — image could not be processed.",
            }
    else:
        mat_pred = {
            "material": None,
            "confidence": None,
            "confidence_percent": None,
            "confidence_status": "MANUAL_REVIEW_REQUIRED",
            "manual_review_required": True,
            "probabilities": {},
            "probability_ratios": None,
            "model_name": "EfficientNet-B0",
            "model_version": None,
            "model_available": False,
            "model_status": "MODEL_NOT_READY" if image_bytes else "NO_IMAGE",
            "source": "NO_IMAGE" if not image_bytes else "MODEL_NOT_AVAILABLE",
            "warning": "No image provided" if not image_bytes else (service.load_error or "ML model unavailable"),
        }

    predicted_material = mat_pred.get("material")
    rule_material = predicted_material if predicted_material else "UNKNOWN / UNSUPPORTED"

    # 3. Waste Classification (RULE_ENGINE)
    waste_category = get_waste_classification(condition, False, False)
    waste_pred = {
        "waste_category": waste_category,
        "source": "RULE_ENGINE",
        "condition": condition,
    }

    # 4. Circularity Scoring (RULE_ENGINE via ScoringService)
    scoring_svc = ScoringService()
    scores = scoring_svc.calculate_scores(
        material=rule_material,
        condition=condition,
        waste_category=waste_category,
        damage=False,
        contamination=False,
    )
    _log("RULE_ENGINE", f"Recovery potential: {scores.get('circular_economy_score', 0.0)}%")
    score_pred = {
        "circularity_score": scores.get("circular_economy_score", 0.0),
        "overall_sustainability_score": scores.get("overall_sustainability_score", 0.0),
        "sustainability_rating": scores.get("sustainability_rating"),
        "component_scores": scores,
        "source": "RULE_ENGINE",
    }

    # 5. Recommendation Engine (RULE_ENGINE)
    recs = get_recycling_recommendations(rule_material, waste_category)
    recommendation_pred = {
        "strategy": recs.get("strategy"),
        "options": recs.get("options"),
        "material": rule_material,
        "material_confidence": (float(mat_pred.get("confidence") or 0.0)) / 100.0,
        "condition": condition,
        "condition_confidence": 1.0,
        "confidence": round((float(mat_pred.get("confidence") or 0.0)) / 100.0, 3),
        "source": "RULE_ENGINE",
    }

    # 6. Environmental Impact (ESTIMATED)
    env_impact = calculate_environmental_impact(rule_material, quantity_kg, waste_category)
    env_pred = {
        "co2_savings_kg": env_impact["co2"].get("value"),
        "water_savings_liters": env_impact["water"].get("value"),
        "co2_factor": env_impact["co2"].get("factor"),
        "water_factor": env_impact["water"].get("factor"),
        "factor_status": env_impact.get("calculation_status"),
        "methodology": env_impact.get("basis"),
        "landfill_reduction_kg": round(quantity_kg, 2) if quantity_kg else None,
        "disclaimer": "Values are dynamically calculated estimates based on configured project factors; factors require domain/source validation.",
        "source": "ESTIMATED",
    }

    return {
        "status": "success",
        "material_prediction": mat_pred,
        "waste_classification": waste_pred,
        "circularity_score": score_pred,
        "recommendation": recommendation_pred,
        "environmental_impact": env_pred,
        "cv_features": cv_features,
        "model_scope_warning": f"Supported materials: {', '.join(MODEL_CLASSES)}",
        "confidence_note": "Confidence represents the model's classification score among supported classes and is not laboratory-verified fiber composition.",
    }


