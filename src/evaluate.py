"""Evaluate the best validation checkpoint once on the held-out test set."""

from __future__ import annotations

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from model import CLASS_NAMES, MODELS_ROOT, REPORTS_ROOT, create_model, leakage_checks, make_datasets, make_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one saved EfficientNet-B0 checkpoint on the held-out test set.")
    parser.add_argument("--checkpoint", type=Path, default=MODELS_ROOT / "best_efficientnet_b0.pth")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks = leakage_checks()
    datasets_by_split, _ = make_datasets()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("class_names") != CLASS_NAMES:
        raise RuntimeError("Checkpoint class mapping does not match the deterministic project mapping")
    network = create_model(len(CLASS_NAMES), pretrained=False).to(device)
    network.load_state_dict(checkpoint["model_state_dict"])
    network.eval()
    loader = make_loader(datasets_by_split["test"], 32, False, 42)
    actual, predicted, probabilities = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = network(images.to(device))
            probabilities.extend(torch.softmax(logits, 1).cpu().numpy())
            actual.extend(labels.numpy())
            predicted.extend(logits.argmax(1).cpu().numpy())
    actual_array, predicted_array = np.array(actual), np.array(predicted)
    precision, recall, f1, support = precision_recall_fscore_support(actual_array, predicted_array, labels=range(len(CLASS_NAMES)), zero_division=0)
    top3 = np.mean([label in np.argsort(probability)[-3:] for label, probability in zip(actual_array, probabilities)])
    metrics = {"test_accuracy": accuracy_score(actual_array, predicted_array), "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()), "macro_f1": float(f1.mean()), "weighted_precision": float(np.average(precision, weights=support)), "weighted_recall": float(np.average(recall, weights=support)), "weighted_f1": float(np.average(f1, weights=support)), "top_3_accuracy": float(top3), "checkpoint_validation_macro_f1": checkpoint.get("best_validation_macro_f1"), "leakage_checks": checks, "evaluation_count": 1}
    per_class = [{"class_name": name, "precision": float(precision[index]), "recall": float(recall[index]), "f1": float(f1[index]), "support": int(support[index])} for index, name in enumerate(CLASS_NAMES)]
    matrix = confusion_matrix(actual_array, predicted_array, labels=range(len(CLASS_NAMES)))
    pairs = sorted(((matrix[row, column], CLASS_NAMES[row], CLASS_NAMES[column]) for row in range(len(CLASS_NAMES)) for column in range(len(CLASS_NAMES)) if row != column), reverse=True)[:5]
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORTS_ROOT / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (REPORTS_ROOT / "classification_report.csv").open("w", newline="", encoding="utf-8") as handle:
        handle.write("class_name,precision,recall,f1,support\n")
        for row in per_class:
            handle.write(f"{row['class_name']},{row['precision']},{row['recall']},{row['f1']},{row['support']}\n")
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(xticks=range(len(CLASS_NAMES)), yticks=range(len(CLASS_NAMES)), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, xlabel="Predicted", ylabel="Actual", title="EfficientNet-B0 Test Confusion Matrix")
    axis.tick_params(axis="x", labelrotation=65)
    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
            axis.text(column, row, matrix[row, column], ha="center", va="center")
    figure.tight_layout(); figure.savefig(REPORTS_ROOT / "confusion_matrix.png", dpi=150); plt.close(figure)
    weakest = sorted(per_class, key=lambda row: row["f1"])[:3]
    report_lines = [
        "# EfficientNet-B0 Baseline Report", "", "## Dataset Description", "", "The validated leakage-safe 9-class fabric dataset was used without modifying the original iBUG dataset or the cleaned source dataset.",
        "", "## Classes", "", ", ".join(CLASS_NAMES), "", "## Split Counts", "", f"- Train: **{len(datasets_by_split['train']):,}**", f"- Validation: **{len(datasets_by_split['validation']):,}**", f"- Test: **{len(datasets_by_split['test']):,}**", "", "## Model and Transfer Learning", "", "EfficientNet-B0 initialized with ImageNet weights. The classifier head was warmed up with the backbone frozen, then the upper feature layers were fine-tuned. AdamW and validation macro F1 checkpoint selection were used.", "", "## Evaluation", "", "The test set was evaluated once after training using only the best validation-macro-F1 checkpoint.", "", f"- Test accuracy: **{metrics['test_accuracy']:.4f}**", f"- Macro precision: **{metrics['macro_precision']:.4f}**", f"- Macro recall: **{metrics['macro_recall']:.4f}**", f"- Macro F1: **{metrics['macro_f1']:.4f}**", f"- Weighted F1: **{metrics['weighted_f1']:.4f}**", f"- Top-3 accuracy: **{metrics['top_3_accuracy']:.4f}**", "", "## Per-Class Results", "", "| Class | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|",
    ]
    report_lines.extend(f"| {row['class_name']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['support']} |" for row in per_class)
    report_lines.extend(["", "## Confusion Matrix Interpretation", "", "The confusion matrix is saved as `reports/confusion_matrix.png`; off-diagonal counts indicate which fabric classes the baseline confuses most.", "", "## Weakest Classes", "", ", ".join(f"{row['class_name']} (F1={row['f1']:.4f})" for row in weakest), "", "## Most Confused Class Pairs", "", ", ".join(f"{actual_name} -> {predicted_name} ({count})" for count, actual_name, predicted_name in pairs), "", "## Limitations and Next Experiments", "", "This is a baseline trained on imbalanced fabric data. Results may be limited by class scarcity, visual similarity, and residual label ambiguity. Recommended next experiments include calibrated sampling, targeted augmentation, and error review without changing the held-out test set."])
    (REPORTS_ROOT / "baseline_model_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("Confusion matrix:"); print(matrix)
    print("Top confused class pairs:", pairs)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())