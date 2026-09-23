"""Backward-compatible predictor wrapper around MaterialInferenceService."""
from typing import Dict, Any

from app.ai.inference_service import MaterialInferenceService


def _format_success(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "success",
        "predicted_material": result.get("material"),
        "confidence": result.get("confidence", result.get("material_confidence", 0.0)),
        "confidence_percent": result.get("confidence_percent", result.get("confidence")),
        "confidence_ratio": result.get("confidence_ratio"),
        "confidence_status": result.get("confidence_status"),
        "confidence_threshold": result.get("confidence_threshold"),
        "manual_review_required": result.get("manual_review_required", False),
        "all_probabilities": result.get("all_probabilities", {}),
        "probabilities": result.get("probabilities", result.get("all_probabilities", {})),
        "probability_ratios": result.get("probability_ratios"),
        "model_name": result.get("model_name"),
        "model_version": result.get("model_version"),
        "num_classes": result.get("num_classes"),
        "current_model_scope": result.get("current_model_scope", "5 CLASS ONLY"),
        "supported_classes": result.get("supported_classes", []),
        "model_accuracy": result.get("model_accuracy"),
        "model_test_accuracy": result.get("model_test_accuracy", result.get("model_accuracy")),
        "model_available": True,
        "model_status": result.get("model_status", "AVAILABLE"),
        "unknown_or_unsupported": result.get("unknown_or_unsupported", False),
        "low_confidence": result.get("low_confidence", False),
        "warning": result.get("warning"),
        "source": "MODEL",
    }


class Predictor:
    def __init__(self) -> None:
        self._service = MaterialInferenceService.get_instance()

    def predict(self, image_path: str) -> Dict[str, Any]:
        result = self._service.predict_from_path(image_path)
        if result.get("status") == "success":
            return _format_success(result)
        return {
            "status": "error",
            "message": result.get("message", "Prediction failed"),
            "predicted_material": None,
            "confidence": None,
            "model_available": False,
            "model_status": self._service.runtime_status,
            "source": "MODEL",
        }

    def predict_bytes(self, image_bytes: bytes, suffix: str = ".jpg") -> Dict[str, Any]:
        result = self._service.predict_from_bytes(image_bytes, suffix=suffix)
        if result.get("status") == "success":
            return _format_success(result)
        return {
            "status": "error",
            "message": result.get("message", "Prediction failed"),
            "predicted_material": None,
            "confidence": None,
            "model_available": False,
            "model_status": result.get("model_status") or self._service.runtime_status,
            "source": "MODEL",
        }
