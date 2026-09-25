"""Real EfficientNet-B0 fabric classification service."""
from __future__ import annotations

print("[FABRIC] module import started", flush=True)

import json
import os
import traceback
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

try:
    print("[FABRIC] importing python-dotenv...", flush=True)
    from dotenv import load_dotenv

    print("[FABRIC] loading local environment file...", flush=True)
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    print("[FABRIC] local environment file loaded", flush=True)
except ImportError:
    print("[FABRIC] python-dotenv unavailable; continuing", flush=True)
    pass


class ModelNotReadyError(RuntimeError):
    """Raised when the configured trained checkpoint is not available."""


MODEL_VERSION = "class_aware_finetune_v1"
print("[FABRIC] module imports completed", flush=True)


class FabricClassifier:
    """Loads the configured classifier once and performs deterministic inference."""

    def __init__(self, load_model: bool = True) -> None:
        print("[FABRIC] initializing classifier...", flush=True)
        self._torch: Any | None = None
        self._models: Any | None = None
        self._transforms: Any | None = None
        self.device: Any = "unloaded"
        self.model: Any | None = None
        self.class_names: list[str] = []
        self.checkpoint_path = os.getenv("FABRIC_MODEL_PATH", "").strip()
        print(f"[FABRIC] checkpoint path configured: {bool(self.checkpoint_path)}", flush=True)
        self.load_error: str | None = None
        self.transform: Any | None = None
        if load_model:
            self._load_runtime()

    def _load_runtime(self) -> None:
        """Import ML dependencies and load the model only when inference requires it.

        Render Free has only 512 MB RAM, so importing PyTorch or constructing the
        EfficientNet model during FastAPI startup can terminate the service.
        """
        if self._torch is not None:
            return
        print("[FABRIC] importing torch for inference...", flush=True)
        import torch
        from torchvision import models, transforms

        self._torch = torch
        self._models = models
        self._transforms = transforms
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[FABRIC] device selected: {self.device}", flush=True)
        self.transform = transforms.Compose([
            transforms.Resize(round(224 * 256 / 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self._load()
        print(f"[FABRIC] classifier load completed; ready={self.is_ready}", flush=True)

    def _load(self) -> None:
        print("[FABRIC] resolving checkpoint path...", flush=True)
        if not self.checkpoint_path:
            self.load_error = "FABRIC_MODEL_PATH is not configured"
            print("[FABRIC] checkpoint path is not configured", flush=True)
            return
        checkpoint_path = Path(self.checkpoint_path)
        if not checkpoint_path.is_absolute():
            candidates = (
                Path.cwd() / checkpoint_path,
                Path(__file__).resolve().parents[3] / checkpoint_path,
                Path(__file__).resolve().parents[2] / checkpoint_path,
            )
            checkpoint_path = next((path for path in candidates if path.is_file()), candidates[0])
        print(f"[FABRIC] checkpoint path resolved; exists={checkpoint_path.is_file()}", flush=True)
        if not checkpoint_path.is_file():
            self.load_error = "Configured fabric model checkpoint was not found"
            print("[FABRIC] checkpoint file not found", flush=True)
            return

        class_names_path = checkpoint_path.parent / "class_names.json"
        print(f"[FABRIC] checking class mapping; exists={class_names_path.is_file()}", flush=True)
        if not class_names_path.is_file():
            self.load_error = "Fabric model class mapping was not found"
            print("[FABRIC] class mapping file not found", flush=True)
            return
        try:
            print("[FABRIC] loading class mapping...", flush=True)
            with class_names_path.open(encoding="utf-8") as handle:
                mapping: dict[str, str] = json.load(handle)
            print("[FABRIC] class mapping loaded", flush=True)
            print("[FABRIC] loading checkpoint with torch.load...", flush=True)
            checkpoint = self._torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            print("[FABRIC] checkpoint loaded", flush=True)
            checkpoint_classes = checkpoint.get("class_names") if isinstance(checkpoint, dict) else None
            mapped_classes = [mapping[str(index)] for index in range(len(mapping))]
            if len(mapped_classes) != 9:
                raise ValueError(f"Expected 9 classes, got {len(mapped_classes)}")
            if checkpoint_classes is not None and [str(name) for name in checkpoint_classes] != mapped_classes:
                raise ValueError("Checkpoint and class mapping differ")
            self.class_names = [str(name) for name in (checkpoint_classes or mapped_classes)]
            state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            print("[FABRIC] creating EfficientNet-B0...", flush=True)
            model = self._models.efficientnet_b0(weights=None)
            print("[FABRIC] EfficientNet-B0 created", flush=True)
            input_features = model.classifier[1].in_features
            model.classifier[1] = self._torch.nn.Linear(input_features, len(self.class_names))
            print("[FABRIC] loading checkpoint state dict...", flush=True)
            model.load_state_dict(state_dict)
            print("[FABRIC] checkpoint state dict loaded", flush=True)
            print("[FABRIC] moving model to device...", flush=True)
            self.model = model.to(self.device)
            print("[FABRIC] model moved to device", flush=True)
            self.model.eval()
            print("[FABRIC] model set to evaluation mode", flush=True)
            print(f"[FABRIC MODEL] architecture=EfficientNet-B0 checkpoint={checkpoint_path.resolve()} exists=True device={self.device} output_classes={len(self.class_names)} mapping={list(enumerate(self.class_names))}")
        except Exception as exc:
            self.model = None
            self.class_names = []
            self.load_error = f"{type(exc).__name__}: {exc}"
            print(f"[FABRIC] MODEL LOAD FAILED: {self.load_error}", flush=True)
            traceback.print_exc()

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
            "error": self.load_error if not self.is_ready else None,
        }

    def predict(self, image: Image.Image) -> dict[str, Any]:
        self._load_runtime()
        if not self.is_ready:
            raise ModelNotReadyError(self.load_error or "Fabric classification model is not available yet.")
        image = ImageOps.exif_transpose(image).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with self._torch.inference_mode():
            logits = self.model(tensor)[0]
            probabilities = self._torch.softmax(logits, dim=0)
            values, indices = self._torch.topk(probabilities, k=min(3, len(self.class_names)))
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
    print(f"[FABRIC] get_fabric_classifier called; singleton_exists={_classifier is not None}", flush=True)
    if _classifier is None:
        print("[FABRIC] creating singleton classifier...", flush=True)
        _classifier = FabricClassifier(load_model=True)
        print("[FABRIC] singleton classifier created", flush=True)
    print(f"[FABRIC] get_fabric_classifier completed; ready={_classifier.is_ready}", flush=True)
    return _classifier


print("[FABRIC] module import completed", flush=True)
