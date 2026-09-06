"""MobileNetV3 input preprocessing — serializable for Keras model save/load."""
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input


def mobilenet_v3_preprocess(img):
    """Scale [0,1] float images to MobileNetV3 expected preprocessed range."""
    return preprocess_input(img * 255.0)
