"""Validation-only EfficientNet-B0 improvement experiments and final evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import WeightedRandomSampler
from torchvision import datasets

from model import CLASS_NAMES, DATASET_ROOT, IMAGE_SIZE, MODELS_ROOT, REPORTS_ROOT, create_model, get_transforms, leakage_checks, make_loader, seed_everything, training_class_weights

IMPROVED_CHECKPOINT = MODELS_ROOT / "best_efficientnet_b0_improved.pth"
BASELINE_ACCURACY = 0.8365
BASELINE_MACRO_F1 = 0.7955
BASELINE_TOP3 = 0.9743


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run validation-only EfficientNet-B0 improvement experiments.")
    parser.add_argument("--mode", choices=("experiments", "evaluate"), default="experiments")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--checkpoint", type=Path, default=IMPROVED_CHECKPOINT)
    return parser.parse_args()


EXPERIMENTS = (
    {"experiment": "baseline_no_weights", "augmentation": "baseline", "sampling": "random", "loss": "cross_entropy", "mixup": False, "fine_tune_lr": 1e-4},
    {"experiment": "targeted_augmentation", "augmentation": "targeted", "sampling": "random", "loss": "cross_entropy", "mixup": False, "fine_tune_lr": 1e-4},
    {"experiment": "weighted_sampler", "augmentation": "baseline", "sampling": "weighted_random_sampler", "loss": "cross_entropy", "mixup": False, "fine_tune_lr": 1e-4},
    {"experiment": "moderate_weighted_loss", "augmentation": "baseline", "sampling": "random", "loss": "sqrt_inverse_frequency", "mixup": False, "fine_tune_lr": 1e-4},
    {"experiment": "targeted_mixup", "augmentation": "targeted", "sampling": "random", "loss": "cross_entropy", "mixup": True, "fine_tune_lr": 1e-4},
)


def make_train_validation(targeted: bool) -> tuple[datasets.ImageFolder, datasets.ImageFolder]:
    train_transform, validation_transform = get_transforms(IMAGE_SIZE, targeted=targeted)
    train = datasets.ImageFolder(DATASET_ROOT / "train", transform=train_transform)
    validation = datasets.ImageFolder(DATASET_ROOT / "validation", transform=validation_transform)
    expected = {name: index for index, name in enumerate(CLASS_NAMES)}
    if train.class_to_idx != expected or validation.class_to_idx != expected:
        raise RuntimeError("Train/validation class mappings differ")
    return train, validation


def loss_weights(train: datasets.ImageFolder, strategy: str, device: torch.device) -> torch.Tensor | None:
    if strategy == "cross_entropy":
        return None
    inverse = training_class_weights(train)
    weights = torch.sqrt(inverse) if strategy == "sqrt_inverse_frequency" else inverse
    return (weights / weights.mean()).to(device)


def validation_pass(network: nn.Module, loader, loss_fn: nn.Module, device: torch.device) -> tuple[float, float, float, np.ndarray]:
    network.eval(); losses = []; actual = []; predicted = []
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = network(images); losses.append(loss_fn(logits, labels).item() * labels.size(0))
            actual.extend(labels.cpu().numpy()); predicted.extend(logits.argmax(1).cpu().numpy())
    return sum(losses) / len(loader.dataset), accuracy_score(actual, predicted), f1_score(actual, predicted, average="macro", zero_division=0), confusion_matrix(actual, predicted, labels=range(len(CLASS_NAMES)))


def mixup(images: torch.Tensor, labels: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    mixing = float(np.random.beta(alpha, alpha)); permutation = torch.randperm(images.size(0), device=images.device)
    return mixing * images + (1 - mixing) * images[permutation], labels, labels[permutation], mixing


def run_one(spec: dict[str, object], config: argparse.Namespace, device: torch.device) -> dict[str, object]:
    seed_everything(config.seed)
    train, validation = make_train_validation(bool(spec["augmentation"] == "targeted"))
    sampler = None
    if spec["sampling"] == "weighted_random_sampler":
        weights = training_class_weights(train)
        sample_weights = torch.tensor([weights[target] for target in train.targets], dtype=torch.double)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True, generator=torch.Generator().manual_seed(config.seed))
    train_loader = make_loader(train, config.batch_size, True, config.seed, config.num_workers, sampler)
    validation_loader = make_loader(validation, config.batch_size, False, config.seed, config.num_workers)
    weights = loss_weights(train, str(spec["loss"]), device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    network = create_model(len(CLASS_NAMES), pretrained=True).to(device)
    for parameter in network.features.parameters(): parameter.requires_grad = False
    optimizer = AdamW(filter(lambda parameter: parameter.requires_grad, network.parameters()), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = {"f1": -1.0, "accuracy": 0.0, "weighted_f1": 0.0, "epoch": 0, "matrix": None}
    stale = 0; started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        if epoch == config.warmup_epochs + 1:
            for parameter in network.features[-3:].parameters(): parameter.requires_grad = True
            optimizer = AdamW(filter(lambda parameter: parameter.requires_grad, network.parameters()), lr=float(spec["fine_tune_lr"]), weight_decay=1e-4)
            scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=2)
        network.train(); train_loss = 0.0
        for batch_index, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = network(images)
                if spec["mixup"]:
                    mixed, first, second, mixing = mixup(images, labels, config.mixup_alpha)
                    logits = network(mixed); loss = mixing * loss_fn(logits, first) + (1 - mixing) * loss_fn(logits, second)
                else: loss = loss_fn(logits, labels)
            if device.type == "cuda": scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else: loss.backward(); optimizer.step()
            train_loss += loss.item() * labels.size(0)
            if batch_index == 1 or batch_index == len(train_loader) or batch_index % 25 == 0: print(f"[{spec['experiment']}] epoch {epoch}/{config.epochs} batch {batch_index}/{len(train_loader)} loss={loss.item():.4f}", flush=True)
        validation_loss, validation_accuracy, validation_f1, matrix = validation_pass(network, validation_loader, loss_fn, device)
        actual_support = matrix.sum(axis=1); per_class_f1 = np.divide(np.diag(matrix) * 2, actual_support + matrix.sum(axis=0), out=np.zeros(len(CLASS_NAMES), dtype=float), where=(actual_support + matrix.sum(axis=0)) != 0)
        weighted_f1 = float(np.average(per_class_f1, weights=actual_support))
        scheduler.step(validation_f1)
        print(f"[{spec['experiment']}] epoch {epoch}: train_loss={train_loss / len(train_loader.dataset):.4f} val_loss={validation_loss:.4f} val_accuracy={validation_accuracy:.4f} val_macro_f1={validation_f1:.4f}", flush=True)
        if validation_f1 > best["f1"]:
            best = {"f1": validation_f1, "accuracy": validation_accuracy, "weighted_f1": weighted_f1, "epoch": epoch, "matrix": matrix}; stale = 0
            torch.save({"model_state_dict": network.state_dict(), "class_names": CLASS_NAMES, "epoch": epoch, "validation_accuracy": validation_accuracy, "validation_macro_f1": validation_f1, "validation_weighted_f1": weighted_f1, "training_config": vars(config), "experiment": spec}, MODELS_ROOT / f"experiment_{spec['experiment']}_best.pth")
        else:
            stale += 1
            if stale >= config.patience: break
    matrix = np.asarray(best["matrix"]); figure, axis = plt.subplots(figsize=(9, 7)); axis.imshow(matrix, cmap="Blues"); axis.set(xticks=range(len(CLASS_NAMES)), yticks=range(len(CLASS_NAMES)), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, title=f"{spec['experiment']} validation confusion matrix"); axis.tick_params(axis="x", labelrotation=65)
    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)): axis.text(column, row, matrix[row, column], ha="center", va="center")
    figure.tight_layout(); figure.savefig(REPORTS_ROOT / f"experiment_{spec['experiment']}_validation_confusion_matrix.png", dpi=150); plt.close(figure)
    return {"experiment": spec["experiment"], "augmentation": spec["augmentation"], "sampling": spec["sampling"], "loss": spec["loss"], "mixup": spec["mixup"], "learning_rate": config.learning_rate, "best_epoch": best["epoch"], "validation_accuracy": best["accuracy"], "validation_macro_f1": best["f1"], "validation_weighted_f1": best["weighted_f1"], "training_time": round(time.perf_counter() - started, 2), "checkpoint": str(MODELS_ROOT / f"experiment_{spec['experiment']}_best.pth")}


def run_experiments(config: argparse.Namespace) -> int:
    if config.epochs < 1 or config.warmup_epochs >= config.epochs: raise SystemExit("epochs must be greater than warmup-epochs")
    checks = leakage_checks(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Validation-only improvement experiments; test set is not loaded or evaluated.")
    results = [run_one(spec, config, device) for spec in EXPERIMENTS]
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    fields = ["experiment", "augmentation", "sampling", "loss", "mixup", "learning_rate", "best_epoch", "validation_accuracy", "validation_macro_f1", "validation_weighted_f1", "training_time", "checkpoint"]
    with (REPORTS_ROOT / "model_improvement_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(results)
    best = max(results, key=lambda row: row["validation_macro_f1"]); (REPORTS_ROOT / "model_improvement_selection.json").write_text(json.dumps({"selected_by": "validation_macro_f1", "best_experiment": best, "baseline_test_metrics": {"accuracy": BASELINE_ACCURACY, "macro_f1": BASELINE_MACRO_F1, "top3": BASELINE_TOP3}, "leakage_checks": checks, "test_evaluated": False}, indent=2) + "\n", encoding="utf-8")
    print("\nEXPERIMENT COMPARISON (validation only)")
    for row in results: print(f"{row['experiment']}: epoch={row['best_epoch']} accuracy={row['validation_accuracy']:.4f} macro_f1={row['validation_macro_f1']:.4f} weighted_f1={row['validation_weighted_f1']:.4f} time={row['training_time']}s")
    print(f"Recommended next test evaluation: {best['experiment']} by validation Macro F1.")
    return 0


def final_evaluate(config: argparse.Namespace) -> int:
    checks = leakage_checks(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, validation_transform = get_transforms(IMAGE_SIZE); test = datasets.ImageFolder(DATASET_ROOT / "test", transform=validation_transform)
    if test.class_to_idx != {name: index for index, name in enumerate(CLASS_NAMES)}: raise RuntimeError("Test class mapping mismatch")
    if not config.checkpoint.is_file(): raise SystemExit(f"Improved checkpoint does not exist: {config.checkpoint}")
    checkpoint = torch.load(config.checkpoint, map_location=device, weights_only=False); network = create_model(len(CLASS_NAMES), pretrained=False).to(device); network.load_state_dict(checkpoint["model_state_dict"]); network.eval(); loader = make_loader(test, config.batch_size, False, config.seed, config.num_workers)
    actual = []; predicted = []; probabilities = []
    with torch.inference_mode():
        for images, labels in loader:
            logits = network(images.to(device)); probabilities.extend(torch.softmax(logits, 1).cpu().numpy()); actual.extend(labels.numpy()); predicted.extend(logits.argmax(1).cpu().numpy())
    actual, predicted = np.array(actual), np.array(predicted); precision, recall, f1, support = precision_recall_fscore_support(actual, predicted, labels=range(len(CLASS_NAMES)), zero_division=0); matrix = confusion_matrix(actual, predicted, labels=range(len(CLASS_NAMES))); pairs = sorted(((int(matrix[row, column]), CLASS_NAMES[row], CLASS_NAMES[column]) for row in range(len(CLASS_NAMES)) for column in range(len(CLASS_NAMES)) if row != column), reverse=True)[:5]; write_rows = [{"class_name": name, "precision": float(precision[index]), "recall": float(recall[index]), "f1": float(f1[index]), "support": int(support[index])} for index, name in enumerate(CLASS_NAMES)]; metrics = {"test_accuracy": float(accuracy_score(actual, predicted)), "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()), "macro_f1": float(f1.mean()), "weighted_precision": float(np.average(precision, weights=support)), "weighted_recall": float(np.average(recall, weights=support)), "weighted_f1": float(np.average(f1, weights=support)), "top_3_accuracy": float(np.mean([label in np.argsort(probability)[-3:] for label, probability in zip(actual, probabilities)])), "checkpoint": str(config.checkpoint), "checkpoint_epoch": checkpoint.get("epoch"), "checkpoint_validation_macro_f1": checkpoint.get("validation_macro_f1"), "baseline": {"test_accuracy": BASELINE_ACCURACY, "macro_f1": BASELINE_MACRO_F1, "top_3_accuracy": BASELINE_TOP3}, "improvement_over_baseline": {"test_accuracy": float(accuracy_score(actual, predicted) - BASELINE_ACCURACY), "macro_f1": float(f1.mean() - BASELINE_MACRO_F1), "top_3_accuracy": float(np.mean([label in np.argsort(probability)[-3:] for label, probability in zip(actual, probabilities)]) - BASELINE_TOP3)}, "leakage_checks": checks, "evaluation_count": 1}
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True); (REPORTS_ROOT / "improved_test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8");
    with (REPORTS_ROOT / "improved_classification_report.csv").open("w", newline="", encoding="utf-8") as handle: writer = csv.DictWriter(handle, fieldnames=list(write_rows[0])); writer.writeheader(); writer.writerows(write_rows)
    figure, axis = plt.subplots(figsize=(9, 7)); axis.imshow(matrix, cmap="Blues"); axis.set(xticks=range(len(CLASS_NAMES)), yticks=range(len(CLASS_NAMES)), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, title="Improved model test confusion matrix"); axis.tick_params(axis="x", labelrotation=65); figure.tight_layout(); figure.savefig(REPORTS_ROOT / "improved_confusion_matrix.png", dpi=150); plt.close(figure)
    weakest = sorted(write_rows, key=lambda row: row["f1"])[:3]
    report = ["# Improved EfficientNet-B0 Report", "", "The locked test set was evaluated once, after the improved checkpoint was selected using validation Macro F1. The original dataset, cleaned dataset, labels, and split were not modified.", "", "## Results", "", f"- Test accuracy: **{metrics['test_accuracy']:.4f}**", f"- Macro F1: **{metrics['macro_f1']:.4f}**", f"- Weighted F1: **{metrics['weighted_f1']:.4f}**", f"- Top-3 accuracy: **{metrics['top_3_accuracy']:.4f}**", "", "## Baseline Comparison", "", f"- Baseline accuracy: **{BASELINE_ACCURACY:.4f}**", f"- Baseline Macro F1: **{BASELINE_MACRO_F1:.4f}**", f"- Baseline Top-3: **{BASELINE_TOP3:.4f}**", f"- Accuracy delta: **{metrics['improvement_over_baseline']['test_accuracy']:+.4f}**", f"- Macro F1 delta: **{metrics['improvement_over_baseline']['macro_f1']:+.4f}**", "", "## Per-Class Results", "", "| Class | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
    report.extend(f"| {row['class_name']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['support']} |" for row in write_rows)
    report.extend(["", "## Weakest Classes", "", ", ".join(f"{row['class_name']} (F1={row['f1']:.4f})" for row in weakest), "", "## Most Common Confusion Pairs", "", ", ".join(f"{actual_name} -> {predicted_name} ({count})" for count, actual_name, predicted_name in pairs), "", "Test metrics were not used during training or model selection."])
    (REPORTS_ROOT / "model_improvement_report.md").write_text("\n".join(report) + "\n", encoding="utf-8"); print(json.dumps(metrics, indent=2)); return 0


if __name__ == "__main__":
    configuration = parse_args(); raise SystemExit(run_experiments(configuration) if configuration.mode == "experiments" else final_evaluate(configuration))