"""Reproducible baseline diagnostic for the five-class material model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix

from app.ai.dataset_preprocessor import DatasetPreprocessor
from app.ai.model_loader import load_keras_material_model
from material_classes import MODEL_CLASSES


BACKEND_ROOT = Path(__file__).resolve().parents[1]
SPLIT_ROOT = BACKEND_ROOT / "data" / "splits" / "ibug_material_v2"
MODEL_PATH = BACKEND_ROOT / "models" / "material-mobilenetv3-v0.2-5class-ibug.keras"


def image_files(directory: Path):
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )


def main() -> int:
    classes = list(MODEL_CLASSES)
    preprocessor = DatasetPreprocessor()
    model = load_keras_material_model(str(MODEL_PATH))
    images = []
    labels = []
    paths = []
    split_counts = {}

    for split in ("train", "validation", "test"):
        split_counts[split] = {}
        for index, class_name in enumerate(classes):
            files = image_files(SPLIT_ROOT / split / class_name)
            split_counts[split][class_name] = len(files)
            if split == "test":
                for path in files:
                    images.append(preprocessor.preprocess_image(str(path)))
                    labels.append(index)
                    paths.append(str(path))

    probabilities = model.predict(np.asarray(images), verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    labels_array = np.asarray(labels)
    report = classification_report(
        labels_array,
        predictions,
        labels=range(len(classes)),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(labels_array, predictions, labels=range(len(classes)))
    cotton_indices = np.where(labels_array == 0)[0]

    result = {
        "model_path": str(MODEL_PATH),
        "architecture": "mobilenet_v3_small",
        "input_size": [224, 224, 3],
        "mapping": {str(index): name for index, name in enumerate(classes)},
        "preprocessing": "RGB, resize 224x224, divide by 255, MobileNetV3 preprocess_input once inside model",
        "split_counts": split_counts,
        "test_count": len(labels),
        "accuracy": float(np.mean(predictions == labels_array)),
        "balanced_accuracy": float(balanced_accuracy_score(labels_array, predictions)),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "confusion_matrix": matrix.tolist(),
        "cotton_count": len(cotton_indices),
        "cotton_accuracy": float(np.mean(predictions[cotton_indices] == 0)),
        "cotton_error_rates": {
            name: float(np.mean(predictions[cotton_indices] == index))
            for index, name in enumerate(classes)
            if index != 0
        },
        "per_class": {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": report[name]["support"],
            }
            for name in classes
        },
        "highest_confidence_errors": [
            {
                "image": paths[index],
                "ground_truth": classes[labels[index]],
                "prediction": classes[predictions[index]],
                "confidence": float(probabilities[index, predictions[index]]),
            }
            for index in np.argsort(np.max(probabilities, axis=1))[::-1]
            if predictions[index] != labels[index]
        ][:20],
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
