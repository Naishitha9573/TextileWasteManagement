"""Confidence tier helper shared by inference and API layers.

Tiers are derived from validation-calibrated thresholds stored with the
trained model (see TrainingPipeline._calibrate_thresholds); yaml defaults are
only a fallback for models trained before calibration existed.
"""
from app.ai.training_config import InferenceConfig


def build_confidence_fields(confidence_ratio: float, config: InferenceConfig | None = None) -> dict:
    cfg = config or InferenceConfig.from_yaml()
    status = cfg.confidence_status(confidence_ratio)
    manual_review = status in {"LOW_CONFIDENCE", "MEDIUM_CONFIDENCE"}
    if status == "HIGH_CONFIDENCE":
        note = None
    elif status == "MEDIUM_CONFIDENCE":
        note = cfg.reviewable_message
    else:
        note = cfg.low_confidence_message
    return {
        "confidence": round(confidence_ratio * 100, 1),
        "confidence_ratio": round(confidence_ratio, 4),
        "confidence_threshold": cfg.manual_review_threshold,
        "confidence_status": status,
        "manual_review_required": manual_review,
        "review_note": note,
        "confidence_note": "Confidence represents the model's classification score among supported classes and is not laboratory-verified fiber composition.",
        "confidence_semantics": "model confidence among supported classes; not fiber composition percentage",
    }
