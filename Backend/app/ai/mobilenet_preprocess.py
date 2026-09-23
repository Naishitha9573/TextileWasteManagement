"""[DEPRECATED] MobileNetV3 input preprocessing — serializable for Keras model save/load.

DEPRECATION STATUS: This module is no longer used by the active EfficientNet-B0 inference pipeline.
It is preserved for historical reference and comparison only.

ACTIVE INFERENCE MODEL: EfficientNet-B0 (PyTorch) — see Backend/app/services/fabric_classifier.py
"""
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input


def mobilenet_v3_preprocess(img):
    """Scale [0,1] float images to MobileNetV3 expected preprocessed range."""
    return preprocess_input(img * 255.0)
