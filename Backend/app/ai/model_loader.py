"""[DEPRECATED] Load Keras material classifier artifacts.

DEPRECATION STATUS: This module is no longer used by the active EfficientNet-B0 inference pipeline.
It is preserved for historical reference and comparison only.

ACTIVE INFERENCE MODEL: EfficientNet-B0 (PyTorch) — see Backend/app/services/fabric_classifier.py
"""
from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from tensorflow.keras import Model


def _patch_lambda_preprocess(model: "Model") -> "Model":
    """Ensure Lambda preprocess survives deserialization (anonymous lambda fallback)."""
    from tensorflow.keras.layers import Lambda

    from app.ai.mobilenet_preprocess import mobilenet_v3_preprocess

    for layer in model.layers:
        if isinstance(layer, Lambda) and layer.name == "mobilenet_preprocess":
            layer.function = mobilenet_v3_preprocess
            break
    return model


def load_keras_material_model(
    model_path: str,
    num_classes: Optional[int] = None,
) -> "Model":
    """Load trained .keras model with preprocess custom objects."""
    from tensorflow import keras

    from app.ai.mobilenet_preprocess import mobilenet_v3_preprocess

    keras.config.enable_unsafe_deserialization()
    custom_objects = {"mobilenet_v3_preprocess": mobilenet_v3_preprocess}

    try:
        model = keras.models.load_model(
            model_path,
            safe_mode=False,
            compile=False,
            custom_objects=custom_objects,
        )
        return _patch_lambda_preprocess(model)
    except Exception:
        pass

    if num_classes is None:
        label_map_path = Path(model_path).parent / "label_map.json"
        if label_map_path.exists():
            with open(label_map_path, encoding="utf-8") as handle:
                num_classes = len(json.load(handle))
        else:
            num_classes = 3

    from app.ai.train import build_material_classifier
    from app.ai.training_config import TrainingConfig

    model = build_material_classifier(num_classes, TrainingConfig())

    with zipfile.ZipFile(model_path, "r") as archive:
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp.write(archive.read("model.weights.h5"))
            weights_path = tmp.name
    try:
        model.load_weights(weights_path)
    finally:
        os.unlink(weights_path)

    return model
