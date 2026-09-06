from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from app.ai.inference_service import run_unified_analysis

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.texture_analysis import analyze_texture_image


def analyze_texture_features(image_bytes: bytes, filename: str = "texture.jpg") -> Dict[str, Any]:
    """Run the standalone texture-analysis module on image bytes without affecting training."""
    suffix = Path(filename).suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(image_bytes)
        temp_path = handle.name
    try:
        result = analyze_texture_image(temp_path)
        result["status"] = "success"
        result["image_name"] = filename
        return result
    finally:
        Path(temp_path).unlink(missing_ok=True)


def analyze_textile(
    image_bytes: bytes,
    filename: str = "textile.jpg",
    fabric_type_hint: str = "Mixed Fabrics",
    condition: str = "Good",
    quantity_kg: float = 1.0,
) -> Dict[str, Any]:
    """Thin wrapper around the existing unified inference pipeline.

    This keeps a single source of truth for the analysis flow while exposing a
    simpler API consistent with the milestone requirements.
    """
    result = run_unified_analysis(
        image_bytes=image_bytes,
        filename=filename,
        fabric_type_hint=fabric_type_hint,
        condition=condition,
        quantity_kg=quantity_kg,
    )

    material = result.get("material_prediction", {}).get("material") or result.get("material") or fabric_type_hint
    material_confidence = result.get("material_prediction", {}).get("confidence") or result.get("material_confidence") or 0.0
    waste_category = result.get("waste_classification", {}).get("waste_category") or result.get("waste_category") or "Unknown"
    waste_confidence = result.get("waste_classification", {}).get("confidence") or result.get("waste_confidence") or 0.0
    score = result.get("circularity_score", {}).get("score") or result.get("recyclability_score") or 0
    level = result.get("circularity_score", {}).get("level") or result.get("recyclability_level") or "Unknown"
    recommendation = result.get("recommendation", {}).get("recommendation") or result.get("recommendation") or "Manual inspection recommended"

    return {
        "status": "success" if result.get("status") == "success" else "error",
        "material": material,
        "confidence": float(material_confidence),
        "model_version": result.get("material_prediction", {}).get("model_version") or "v1",
        "waste_category": waste_category,
        "waste_confidence": float(waste_confidence),
        "recyclability": {"score": int(score), "level": level},
        "recommendation": recommendation,
        "raw": result,
    }
