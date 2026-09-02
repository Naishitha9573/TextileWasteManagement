import io
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import List
import os
import tempfile
from PIL import Image

from database import get_db, User
from auth import get_current_user
from app.ai.train import TrainingPipeline
from app.ai.model_registry import ModelRegistry
from app.ai.dataset_manager import DatasetManager
from app.services.fabric_classifier import ModelNotReadyError, get_fabric_classifier
from app.services.color_analysis import analyze_image_color_bytes
from app.core.upload_validator import validate_uploaded_image
from textile_analysis_service import analyze_texture_features

router = APIRouter(prefix="/api", tags=["ai"])

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


@router.post("/train")
def train_models(current_user: User = Depends(get_current_user)):
    if current_user.role not in {"Administrator", "Sustainability Manager"}:
        raise HTTPException(status_code=403, detail="Only administrators and sustainability managers can train models")
    try:
        pipeline = TrainingPipeline()
        result = pipeline.train()
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.post("/retrain")
def retrain_models(current_user: User = Depends(get_current_user)):
    return train_models(current_user)


@router.post("/predict")
async def predict_image(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    content = await file.read()
    valid, message = validate_uploaded_image(content, file.filename, file.content_type or "")
    if not valid:
        raise HTTPException(status_code=400, detail=message)
    try:
        color_result = analyze_image_color_bytes(content)
    except Exception:
        raise HTTPException(status_code=422, detail="The image color could not be analyzed")
    try:
        image = Image.open(io.BytesIO(content))
        result = get_fabric_classifier().predict(image)
    except ModelNotReadyError:
        return {
            "success": True,
            "prediction": None,
            **color_result,
            "error": "MODEL_NOT_READY",
            "message": "The EfficientNet-B0 fabric classification model is still being trained or is not available.",
            "model": {
                "name": "EfficientNet-B0",
                "architecture": "EfficientNet-B0",
                "num_classes": 9,
                "status": "MODEL_NOT_READY",
            },
        }
    except Exception:
        raise HTTPException(status_code=422, detail="The image could not be processed for classification")
    return {**result, **color_result}


@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    """Standalone image analysis using the active EfficientNet classifier."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    content = await file.read()
    try:
        image = Image.open(io.BytesIO(content))
        result = get_fabric_classifier().predict(image)
    except ModelNotReadyError:
        return JSONResponse(status_code=503, content={
            "success": False,
            "error": "MODEL_NOT_READY",
            "message": "The EfficientNet-B0 fabric classification model is still being trained or is not available.",
        })
    except Exception:
        raise HTTPException(status_code=422, detail="The image could not be processed for classification")
    return result


@router.get("/models")
def list_models(current_user: User = Depends(get_current_user)):
    registry = ModelRegistry(str(MODELS_DIR))
    metadata = registry.load_metadata()
    if not metadata:
        return {"models": [], "status": "MODEL NOT TRAINED"}
    service = get_fabric_classifier()
    return {
        "models": [{
            "name": metadata.get("model_name"),
            "version": metadata.get("model_version"),
            "status": metadata.get("status"),
            "classes": metadata.get("classes", []),
        }],
        "status": metadata.get("status", "unknown"),
        "runtime_available": service.is_ready,
    }


@router.get("/training-status")
def training_status(current_user: User = Depends(get_current_user)):
    registry = ModelRegistry(str(MODELS_DIR))
    metadata = registry.load_metadata()
    metrics_path = MODELS_DIR / "model_metrics.json"

    # Runtime state — what inference can actually do right now.
    service = get_fabric_classifier()
    available = service.is_ready
    model_status = "AVAILABLE" if available else "MODEL_NOT_READY"

    if metadata.get("status") == "trained" and metrics_path.exists():
        import json
        with open(metrics_path, "r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        payload = {
            "status": "trained",
            "current_model_scope": metadata.get("current_model_scope", "5 CLASS ONLY"),
            "model_name": metadata.get("model_name"),
            "model_version": metadata.get("model_version"),
            "architecture": "EfficientNet-B0",
            "backend": "pytorch",
            "classes": list(service.class_names),
            "num_classes": len(service.class_names),
            "test_accuracy": metrics.get("accuracy"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "baseline_warning": "Baseline — Not Production Grade",
            "supported_classes": metadata.get("classes", []),
            "model_accuracy": metrics.get("accuracy"),
            # Runtime availability (AVAILABLE / UNAVAILABLE / LOAD_ERROR)
            "available": available,
            "model_status": model_status,
            "confidence_note": "Confidence represents the model's classification score among supported classes and is not laboratory-verified fiber composition.",
            "label_provenance": metrics.get("label_provenance"),
            "label_noise": True,
        }
        if not available and service.load_error:
            payload["error"] = service.load_error
        return payload

    return {
        "status": "not_trained",
        "available": available,
        "model_status": model_status,
        "current_model_scope": "5 CLASS ONLY",
        "error": service.load_error or "MODEL NOT TRAINED — run POST /api/train or python -m app.ai.train",
    }


@router.get("/model/status")
def model_runtime_status():
    """Runtime model availability endpoint (public; no secrets or paths)."""
    service = get_fabric_classifier()
    payload = service.health()
    payload["available"] = service.is_ready
    payload["architecture"] = "EfficientNet-B0"
    payload["classes"] = list(service.class_names) if service.is_ready else None
    payload["status"] = "AVAILABLE" if service.is_ready else "MODEL_NOT_READY"
    return payload


@router.get("/predictions")
def predictions(current_user: User = Depends(get_current_user), db=Depends(get_db)):
    return []


@router.get("/model-metrics")
def model_metrics(current_user: User = Depends(get_current_user)):
    metrics_path = MODELS_DIR / "model_metrics.json"
    if not metrics_path.exists():
        return {"status": "not_evaluated", "message": "Model has not been trained and evaluated yet."}
    import json
    with open(metrics_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@router.get("/dataset-statistics")
def dataset_statistics(current_user: User = Depends(get_current_user)):
    manager = DatasetManager()
    return manager.summarize()
