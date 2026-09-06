"""Single locked-test evaluation for the targeted augmentation v2 checkpoint."""

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

from model import CLASS_NAMES, DATASET_ROOT, IMAGE_SIZE, MODELS_ROOT, REPORTS_ROOT, create_model, get_transforms

EXPERIMENT_NAME = "targeted_augmentation_v2"
CHECKPOINT_PATH = MODELS_ROOT / "best_efficientnet_b0_targeted_augmentation_v2.pth"
REPORT_PATH = REPORTS_ROOT / "targeted_augmentation_v2_test_final.json"
CLASS_REPORT_PATH = REPORTS_ROOT / "targeted_augmentation_v2_test_classification_report.csv"
CONFUSION_PATH = REPORTS_ROOT / "targeted_augmentation_v2_test_confusion_matrix.png"
BASELINE_TEST_ACCURACY = 0.8365
BASELINE_TEST_MACRO_F1 = 0.7955
BASELINE_TEST_TOP3 = 0.9743


def main() -> int:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(CHECKPOINT_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = create_model(len(CLASS_NAMES), pretrained=False).to(device)
    if checkpoint.get("class_names") != CLASS_NAMES:
        raise RuntimeError("Checkpoint class mapping does not match authoritative CLASS_NAMES")
    if model.classifier[1].out_features != len(CLASS_NAMES):
        raise RuntimeError("Checkpoint architecture output dimension is not 9")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, test_transform = get_transforms(IMAGE_SIZE, targeted=False)
    test_dataset = datasets.ImageFolder(DATASET_ROOT / "test", transform=test_transform)
    expected_mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    if test_dataset.class_to_idx != expected_mapping:
        raise RuntimeError("Test class mapping does not match authoritative CLASS_NAMES")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    actual: list[int] = []
    predicted: list[int] = []
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for images, labels in test_loader:
            logits = model(images.to(device))
            probabilities.extend(torch.softmax(logits, dim=1).cpu().numpy())
            actual.extend(labels.numpy())
            predicted.extend(logits.argmax(1).cpu().numpy())

    actual_array = np.asarray(actual)
    predicted_array = np.asarray(predicted)
    probability_array = np.asarray(probabilities)
    labels = list(range(len(CLASS_NAMES)))
    precision, recall, f1_values, support = precision_recall_fscore_support(actual_array, predicted_array, labels=labels, zero_division=0)
    matrix = confusion_matrix(actual_array, predicted_array, labels=labels)
    top3_accuracy = float(np.mean([label in np.argsort(probability)[-3:] for label, probability in zip(actual_array, probability_array)]))
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "checkpoint": str(CHECKPOINT_PATH),
        "test_images": len(actual),
        "test_accuracy": float(accuracy_score(actual_array, predicted_array)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1_values.mean()),
        "weighted_precision": float(np.average(precision, weights=support)),
        "weighted_recall": float(np.average(recall, weights=support)),
        "weighted_f1": float(np.average(f1_values, weights=support)),
        "top3_accuracy": top3_accuracy,
        "baseline_test_accuracy": BASELINE_TEST_ACCURACY,
        "baseline_test_macro_f1": BASELINE_TEST_MACRO_F1,
        "baseline_test_top3_accuracy": BASELINE_TEST_TOP3,
        "accuracy_improvement": float(accuracy_score(actual_array, predicted_array) - BASELINE_TEST_ACCURACY),
        "macro_f1_improvement": float(f1_values.mean() - BASELINE_TEST_MACRO_F1),
        "top3_improvement": float(top3_accuracy - BASELINE_TEST_TOP3),
        "test_evaluation_only": True,
        "training_performed": False,
        "dataset_modified": False,
        "splits_modified": False,
        "baseline_checkpoint_modified": False,
        "class_names": CLASS_NAMES,
        "checkpoint_epoch": checkpoint.get("epoch"),
    }
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with CLASS_REPORT_PATH.open("w", newline="", encoding="utf-8") as handle:
        handle.write("class,precision,recall,f1,support\n")
        for name, class_precision, class_recall, class_f1, class_support in zip(CLASS_NAMES, precision, recall, f1_values, support):
            handle.write(f"{name},{class_precision:.10f},{class_recall:.10f},{class_f1:.10f},{int(class_support)}\n")
    figure, axis = plt.subplots(figsize=(9, 7))
    axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=labels, yticks=labels, xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, title=f"{EXPERIMENT_NAME} test confusion matrix")
    axis.tick_params(axis="x", labelrotation=65)
    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
            axis.text(column, row, matrix[row, column], ha="center", va="center")
    figure.tight_layout()
    figure.savefig(CONFUSION_PATH, dpi=150)
    plt.close(figure)
    report = classification_report(actual_array, predicted_array, labels=labels, target_names=CLASS_NAMES, digits=6, zero_division=0)
    print("=" * 50)
    print("TARGETED AUGMENTATION V2 - FINAL TEST EVALUATION")
    print("=" * 50)
    print(f"\nMODEL:\nEfficientNet-B0\n\nCHECKPOINT:\n{CHECKPOINT_PATH}\n\nTEST IMAGES:\n{len(actual)}\n\nTEST ACCURACY:\n{metrics['test_accuracy']:.6f}\n\nMACRO PRECISION:\n{metrics['macro_precision']:.6f}\n\nMACRO RECALL:\n{metrics['macro_recall']:.6f}\n\nMACRO F1:\n{metrics['macro_f1']:.6f}\n\nWEIGHTED PRECISION:\n{metrics['weighted_precision']:.6f}\n\nWEIGHTED RECALL:\n{metrics['weighted_recall']:.6f}\n\nWEIGHTED F1:\n{metrics['weighted_f1']:.6f}\n\nTOP-3 ACCURACY:\n{metrics['top3_accuracy']:.6f}\n")
    print("=" * 50)
    print("BASELINE COMPARISON")
    print("=" * 50)
    print(f"\nBASELINE TEST ACCURACY:\n{BASELINE_TEST_ACCURACY:.2%}\n\nV2 TEST ACCURACY:\n{metrics['test_accuracy']:.2%}\n\nACCURACY IMPROVEMENT:\n{metrics['accuracy_improvement']:+.6f}\n\nBASELINE TEST MACRO F1:\n{BASELINE_TEST_MACRO_F1:.2%}\n\nV2 TEST MACRO F1:\n{metrics['macro_f1']:.2%}\n\nMACRO F1 IMPROVEMENT:\n{metrics['macro_f1_improvement']:+.6f}\n\nBASELINE TEST TOP-3:\n{BASELINE_TEST_TOP3:.2%}\n\nV2 TEST TOP-3:\n{metrics['top3_accuracy']:.2%}\n\nTOP-3 IMPROVEMENT:\n{metrics['top3_improvement']:+.6f}\n")
    print("=" * 50)
    print("FINAL DECISION")
    print("=" * 50)
    print("\nRECOMMENDATION:\nV2 IS THE NEW BEST MODEL" if metrics["test_accuracy"] > BASELINE_TEST_ACCURACY and metrics["macro_f1"] > BASELINE_TEST_MACRO_F1 else "\nRECOMMENDATION:\nKEEP BASELINE MODEL")
    print("\n=" * 25)
    print("SAFETY CONFIRMATION")
    print("=" * 50)
    print("\nTRAINING:\nNOT PERFORMED\n\nDATASET:\nNOT MODIFIED\n\nTRAIN/VALIDATION/TEST SPLITS:\nNOT MODIFIED\n\nBASELINE CHECKPOINT:\nUNCHANGED")
    print("\nCLASS MAPPING:")
    for index, name in enumerate(CLASS_NAMES):
        print(f"{index} = {name}")
    print(f"\nClassification report:\n{report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())