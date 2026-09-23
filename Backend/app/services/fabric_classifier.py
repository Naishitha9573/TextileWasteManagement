"""Real EfficientNet-B0 fabric classification service."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from torchvision import models, transforms

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
except ImportError:
    pass


class ModelNotReadyError(RuntimeError):
    """Raised when the configured trained checkpoint is not available."""


MODEL_VERSION = "class_aware_finetune_v1"


class FabricClassifier:
    """Loads the configured classifier once and performs deterministic inference."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: torch.nn.Module | None = None
        self.class_names: list[str] = []
        self.checkpoint_path = os.getenv("FABRIC_MODEL_PATH", "").strip()
        self.load_error: str | None = None
        self.transform = transforms.Compose([
            transforms.Resize(round(224 * 256 / 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self._load()

    def _load(self) -> None:
        if not self.checkpoint_path:
            self.load_error = "FABRIC_MODEL_PATH is not configured"
            return
        checkpoint_path = Path(self.checkpoint_path)
        if not checkpoint_path.is_absolute():
            candidates = (
                Path.cwd() / checkpoint_path,
                Path(__file__).resolve().parents[3] / checkpoint_path,
                Path(__file__).resolve().parents[2] / checkpoint_path,
            )
            checkpoint_path = next((path for path in candidates if path.is_file()), candidates[0])
        if not checkpoint_path.is_file():
            self.load_error = "Configured fabric model checkpoint was not found"
            return

        class_names_path = checkpoint_path.parent / "class_names.json"
        if not class_names_path.is_file():
            self.load_error = "Fabric model class mapping was not found"
            return
        try:
            with class_names_path.open(encoding="utf-8") as handle:
                mapping: dict[str, str] = json.load(handle)
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            checkpoint_classes = checkpoint.get("class_names") if isinstance(checkpoint, dict) else None
            mapped_classes = [mapping[str(index)] for index in range(len(mapping))]
            if len(mapped_classes) != 9:
                raise ValueError(f"Expected 9 classes, got {len(mapped_classes)}")
            if checkpoint_classes is not None and [str(name) for name in checkpoint_classes] != mapped_classes:
                raise ValueError("Checkpoint and class mapping differ")
            self.class_names = [str(name) for name in (checkpoint_classes or mapped_classes)]
            state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            model = models.efficientnet_b0(weights=None)
            input_features = model.classifier[1].in_features
            model.classifier[1] = torch.nn.Linear(input_features, len(self.class_names))
            model.load_state_dict(state_dict)
            self.model = model.to(self.device)
            self.model.eval()
            print(f"[FABRIC MODEL] architecture=EfficientNet-B0 checkpoint={checkpoint_path.resolve()} exists=True device={self.device} output_classes={len(self.class_names)} mapping={list(enumerate(self.class_names))}")
        except Exception:
            self.model = None
            self.class_names = []
            self.load_error = "Fabric model checkpoint could not be loaded"

    @property
    def is_ready(self) -> bool:
        return self.model is not None and len(self.class_names) > 0

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model_loaded": self.is_ready,
            "model_name": "EfficientNet-B0",
            "device": str(self.device),
            "num_classes": len(self.class_names) if self.is_ready else None,
        }

    def predict(self, image: Image.Image) -> dict[str, Any]:
        if not self.is_ready:
            raise ModelNotReadyError(self.load_error or "Fabric classification model is not available yet.")
        image = ImageOps.exif_transpose(image).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)[0]
            probabilities = torch.softmax(logits, dim=0)
            values, indices = torch.topk(probabilities, k=min(3, len(self.class_names)))
        top_predictions = [
            {"class_name": self.class_names[int(index)], "confidence": float(value)}
            for value, index in zip(values.cpu(), indices.cpu())
        ]
        all_probabilities = {
            self.class_names[index]: float(probability)
            for index, probability in enumerate(probabilities.cpu())
        }
        print(f"[FABRIC INFERENCE] tensor_shape={tuple(tensor.shape)} logits={logits.detach().cpu().tolist()} probabilities={probabilities.detach().cpu().tolist()} top3={top_predictions}")
        return {
            "success": True,
            "prediction": top_predictions[0],
            "top_predictions": top_predictions,
            "probabilities": all_probabilities,
            "model": {
                "name": "EfficientNet-B0",
                "version": MODEL_VERSION,
                "architecture": "EfficientNet-B0",
                "num_classes": len(self.class_names),
                "classes": list(self.class_names),
                "status": "AVAILABLE",
                "device": str(self.device),
            },
        }


_classifier: FabricClassifier | None = None


def get_fabric_classifier() -> FabricClassifier:
    global _classifier
    if _classifier is None:
        _classifier = FabricClassifier()
    elif not _classifier.is_ready and _classifier.checkpoint_path:
        checkpoint_path = Path(_classifier.checkpoint_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path.cwd() / checkpoint_path
        if checkpoint_path.is_file():
            _classifier = FabricClassifier()
    return _classifier
