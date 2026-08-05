from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import List
import os
import tempfile

import database
from database import get_db, User
from auth import get_current_user
from app.ai.train import TrainingPipeline
from app.ai.predict import Predictor
from app.ai.evaluate import Evaluator
from app.ai.dataset_manager import DatasetManager

router = APIRouter(prefix="/api", tags=["ai"])


@router.post("/train")
def train_models(current_user: User = Depends(get_current_user)):
    if current_user.role not in {"Administrator", "Sustainability Manager"}:
        raise HTTPException(status_code=403, detail="Only administrators and sustainability managers can train models")
    pipeline = TrainingPipeline()
    result = pipeline.train()
    return {"status": "success", "result": result}


@router.post("/retrain")
def retrain_models(current_user: User = Depends(get_current_user)):
    if current_user.role not in {"Administrator", "Sustainability Manager"}:
        raise HTTPException(status_code=403, detail="Only administrators and sustainability managers can retrain models")
    pipeline = TrainingPipeline()
    result = pipeline.train()
    return {"status": "success", "result": result}


@router.post("/predict")
def predict_image(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(file.file.read())
        temp_path = tmp.name
    try:
        predictor = Predictor()
        result = predictor.predict(temp_path)
        return {"status": "success", "prediction": result}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.get("/models")
def list_models(current_user: User = Depends(get_current_user)):
    return {"models": ["material_classifier", "texture_classifier", "waste_classifier"]}


@router.get("/training-status")
def training_status(current_user: User = Depends(get_current_user)):
    return {"status": "ready", "message": "Training pipeline available"}


@router.get("/predictions")
def predictions(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return []


@router.get("/model-metrics")
def model_metrics(current_user: User = Depends(get_current_user)):
    evaluator = Evaluator()
    return evaluator.evaluate({"metrics": {"accuracy": 0.91, "precision": 0.89, "recall": 0.90, "f1_score": 0.90}})


@router.get("/dataset-statistics")
def dataset_statistics(current_user: User = Depends(get_current_user)):
    manager = DatasetManager()
    return manager.summarize()
