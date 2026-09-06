"""Independent TensorFlow waste-category inference service."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


class WasteModelNotReadyError(RuntimeError):
    pass


class WasteClassifier:
    def __init__(self) -> None:
        default_root = Path(__file__).resolve().parents[3] / "models" / "waste_classification"
        self.model_path = Path(os.getenv("WASTE_MODEL_PATH", str(default_root)))
        self.model: tf.keras.Model | None = None
        self.class_names: list[str] = []
        self.load_error: str | None = None
        self._load()

    def _load(self) -> None:
        model_path = self.model_path / "best_waste_model.keras" if self.model_path.is_dir() else self.model_path
        class_path = model_path.parent / "class_names.json"
        if not model_path.is_file() or not class_path.is_file():
            self.load_error = "Waste model artifacts are not available; train the verified dataset first."
            return
        try:
            self.class_names = [str(name) for name in json.loads(class_path.read_text(encoding="utf-8"))]
            self.model = tf.keras.models.load_model(model_path)
            if self.model.output_shape[-1] != len(self.class_names):
                raise ValueError("Model output size and class mapping differ")
        except Exception as exc:
            self.model = None
            self.class_names = []
            self.load_error = f"Waste model could not be loaded: {exc}"

    @property
    def is_ready(self) -> bool:
        return self.model is not None and bool(self.class_names)

    def predict(self, image: Image.Image) -> dict[str, Any]:
        if not self.is_ready:
            raise WasteModelNotReadyError(self.load_error or "Waste model is unavailable")
        image = ImageOps.exif_transpose(image).convert("RGB").resize((224, 224))
        # Match train_waste_classifier.py; EfficientNet performs ImageNet normalization internally.
        batch = np.asarray(image, dtype=np.float32)[None, ...]
        probabilities = self.model.predict(batch, verbose=0)[0]
        predicted_index = int(np.argmax(probabilities))
        return {
            "predicted_category": self.class_names[predicted_index],
            "confidence": float(probabilities[predicted_index]),
            "probabilities": {name: float(probabilities[index]) for index, name in enumerate(self.class_names)},
        }


_classifier: WasteClassifier | None = None


def get_waste_classifier() -> WasteClassifier:
    global _classifier
    if _classifier is None:
        _classifier = WasteClassifier()
    return _classifier