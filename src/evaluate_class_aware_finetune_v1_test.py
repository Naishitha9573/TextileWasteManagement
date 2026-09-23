"""One-time held-out test evaluation for the class-aware material candidate."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import datasets

from model import CLASS_NAMES, DATASET_ROOT, IMAGE_SIZE, MODELS_ROOT, create_model, get_transforms

CHECKPOINT_PATH = MODELS_ROOT / "best_efficientnet_b0_class_aware_finetune_v1.pth"
OUTPUT_ROOT = MODELS_ROOT / "waste_classification"
METRICS_PATH = OUTPUT_ROOT / "test_metrics.json"
CONFUSION_PATH = OUTPUT_ROOT / "confusion_matrix.png"
REPORT_PATH = OUTPUT_ROOT / "classification_report.txt"
BASELINE_ACCURACY = 0.8365
BASELINE_MACRO_F1 = 0.7955
BASELINE_TOP3 = 0.9743


def main() -> int:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(CHECKPOINT_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    if checkpoint.get("class_names") != CLASS_NAMES:
        raise RuntimeError("Candidate checkpoint class mapping differs from authoritative CLASS_NAMES")
    model = create_model(len(CLASS_NAMES), pretrained=False).to(device)
    if model.classifier[1].out_features != 9:
        raise RuntimeError("Candidate model does not have exactly 9 output classes")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _, test_transform = get_transforms(IMAGE_SIZE, targeted=False)
    test_dataset = datasets.ImageFolder(DATASET_ROOT / "test", transform=test_transform)
    expected_mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    if test_dataset.class_to_idx != expected_mapping:
        raise RuntimeError("Test mapping differs from authoritative CLASS_NAMES")
    loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    actual: list[int] = []
    predicted: list[int] = []
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            actual.extend(labels.numpy())
            predicted.extend(logits.argmax(1).cpu().numpy())
            probabilities.extend(torch.softmax(logits, dim=1).cpu().numpy())
    actual_array = np.asarray(actual)
    predicted_array = np.asarray(predicted)
    probability_array = np.asarray(probabilities)
    labels = list(range(len(CLASS_NAMES)))
    precision, recall, f1_values, support = precision_recall_fscore_support(actual_array, predicted_array, labels=labels, zero_division=0)
    accuracy = float(accuracy_score(actual_array, predicted_array))
    macro_f1 = float(f1_values.mean())
    top3 = float(np.mean([label in np.argsort(probability)[-3:] for label, probability in zip(actual_array, probability_array)]))
    metrics = {
        "experiment": "class_aware_finetune_v1",
        "model_type": "material_classification",
        "checkpoint": str(CHECKPOINT_PATH),
        "test_images": len(actual),
        "class_names": CLASS_NAMES,
        "test_accuracy": accuracy,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": macro_f1,
        "weighted_precision": float(np.average(precision, weights=support)),
        "weighted_recall": float(np.average(recall, weights=support)),
        "weighted_f1": float(np.average(f1_values, weights=support)),
        "top3_accuracy": top3,
        "baseline_test_accuracy": BASELINE_ACCURACY,
        "baseline_test_macro_f1": BASELINE_MACRO_F1,
        "baseline_test_top3_accuracy": BASELINE_TOP3,
        "accuracy_improvement": accuracy - BASELINE_ACCURACY,
        "macro_f1_improvement": macro_f1 - BASELINE_MACRO_F1,
        "top3_improvement": top3 - BASELINE_TOP3,
        "test_evaluation_only": True,
        "training_performed": False,
        "dataset_modified": False,
        "splits_modified": False,
        "baseline_checkpoint_modified": False,
        "candidate_epoch": checkpoint.get("best_epoch"),
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    report = classification_report(actual_array, predicted_array, labels=labels, target_names=CLASS_NAMES, digits=6, zero_division=0)
    REPORT_PATH.write_text(report + "\n\nClass mapping:\n" + "\n".join(f"{i} = {name}" for i, name in enumerate(CLASS_NAMES)) + "\n", encoding="utf-8")
    matrix = confusion_matrix(actual_array, predicted_array, labels=labels)
    figure, axis = plt.subplots(figsize=(9, 7))
    axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=labels, yticks=labels, xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, xlabel="Predicted", ylabel="Actual", title="class_aware_finetune_v1 test confusion matrix")
    axis.tick_params(axis="x", labelrotation=65)
    for row in labels:
        for column in labels:
            axis.text(column, row, matrix[row, column], ha="center", va="center")
    figure.tight_layout()
    figure.savefig(CONFUSION_PATH, dpi=150)
    plt.close(figure)
    print("FINAL WASTE MODEL TEST RESULTS")
    print(f"\nTest Accuracy: {accuracy:.6f}\nTest Macro F1: {macro_f1:.6f}\nTest Weighted F1: {metrics['weighted_f1']:.6f}\nTest Macro Precision: {metrics['macro_precision']:.6f}\nTest Macro Recall: {metrics['macro_recall']:.6f}")
    print("\nPer-class results:")
    for name, class_precision, class_recall, class_f1, class_support in zip(CLASS_NAMES, precision, recall, f1_values, support):
        print(f"{name}: precision={class_precision:.6f}, recall={class_recall:.6f}, f1={class_f1:.6f}, support={int(class_support)}")
    print(f"\nExport: NOT A CHECKPOINT CLASS\nReuse: NOT A CHECKPOINT CLASS\nRecycle: NOT A CHECKPOINT CLASS\nEnergy Recovery: NOT A CHECKPOINT CLASS\nRemake: NOT A CHECKPOINT CLASS\nRepair: NOT A CHECKPOINT CLASS")
    print(f"\nSaved metrics: {METRICS_PATH}\nSaved confusion matrix: {CONFUSION_PATH}\nSaved report: {REPORT_PATH}")
    print("Training: NOT PERFORMED\nDataset: NOT MODIFIED\nSplits: NOT MODIFIED\nBaseline Checkpoint: UNCHANGED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())