"""Validation-only normal-sampling fabric augmentation experiment."""

from __future__ import annotations

import csv
import json
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import CLASS_NAMES, DATASET_ROOT, MEAN, MODELS_ROOT, REPORTS_ROOT, STD, create_model, get_transforms, seed_everything

EXPERIMENT_NAME = "targeted_augmentation_v2"
BASELINE_VALIDATION_ACCURACY = 0.8541
BASELINE_VALIDATION_MACRO_F1 = 0.8333
CHECKPOINT_PATH = MODELS_ROOT / f"best_efficientnet_b0_{EXPERIMENT_NAME}.pth"
HISTORY_PATH = REPORTS_ROOT / f"{EXPERIMENT_NAME}_history.csv"
CONFUSION_PATH = REPORTS_ROOT / f"{EXPERIMENT_NAME}_validation_confusion_matrix.png"
FINAL_REPORT_PATH = REPORTS_ROOT / f"{EXPERIMENT_NAME}_final.json"
ERROR_PATH = REPORTS_ROOT / f"{EXPERIMENT_NAME}_error.txt"
HISTORY_FIELDS = ("epoch", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy", "validation_macro_f1", "validation_weighted_f1", "learning_rate", "elapsed_seconds")


def build_train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def save_history(row: dict[str, float | int]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        if HISTORY_PATH.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)


def validation_pass(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> tuple[float, float, float, float, np.ndarray]:
    model.eval()
    losses: list[float] = []
    actual: list[int] = []
    predicted: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            losses.append(loss_fn(logits, labels).item() * labels.size(0))
            actual.extend(labels.cpu().numpy())
            predicted.extend(logits.argmax(1).cpu().numpy())
    matrix = confusion_matrix(actual, predicted, labels=list(range(len(CLASS_NAMES))))
    precision, recall, f1_values, support = precision_recall_fscore_support(actual, predicted, labels=list(range(len(CLASS_NAMES))), zero_division=0)
    return (sum(losses) / len(loader.dataset), accuracy_score(actual, predicted), float(f1_values.mean()), float(np.average(f1_values, weights=support)), matrix)


def write_confusion_matrix(matrix: np.ndarray) -> None:
    figure, axis = plt.subplots(figsize=(9, 7))
    axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=range(len(CLASS_NAMES)), yticks=range(len(CLASS_NAMES)), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, title=f"{EXPERIMENT_NAME} validation confusion matrix")
    axis.tick_params(axis="x", labelrotation=65)
    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
            axis.text(column, row, matrix[row, column], ha="center", va="center")
    figure.tight_layout()
    figure.savefig(CONFUSION_PATH, dpi=150)
    plt.close(figure)


def main() -> int:
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    train_dataset = datasets.ImageFolder(DATASET_ROOT / "train", transform=build_train_transform())
    _, validation_transform = get_transforms(224, targeted=False)
    validation_dataset = datasets.ImageFolder(DATASET_ROOT / "validation", transform=validation_transform)
    expected_mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    if train_dataset.class_to_idx != expected_mapping or validation_dataset.class_to_idx != expected_mapping:
        raise RuntimeError("Train/validation class mappings differ from CLASS_NAMES")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available(), generator=torch.Generator().manual_seed(42))
    validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    model = create_model(len(CLASS_NAMES), pretrained=True).to(device)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=2)
    best = {"epoch": 0, "accuracy": 0.0, "macro_f1": -1.0, "weighted_f1": 0.0, "matrix": None}
    stale = 0
    started = time.perf_counter()
    last_epoch = 0
    last_validation_accuracy = 0.0
    last_validation_macro_f1 = 0.0
    try:
        for epoch in range(1, 21):
            last_epoch = epoch
            if epoch == 3:
                for parameter in model.features[-3:].parameters():
                    parameter.requires_grad = True
                optimizer = AdamW(filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
                scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=2)
            model.train()
            train_loss = 0.0
            train_actual: list[int] = []
            train_predicted: list[int] = []
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * labels.size(0)
                train_actual.extend(labels.cpu().numpy())
                train_predicted.extend(logits.argmax(1).cpu().numpy())
            validation_loss, validation_accuracy, validation_macro_f1, validation_weighted_f1, matrix = validation_pass(model, validation_loader, loss_fn, device)
            last_validation_accuracy = validation_accuracy
            last_validation_macro_f1 = validation_macro_f1
            scheduler.step(validation_macro_f1)
            row = {"epoch": epoch, "train_loss": train_loss / len(train_dataset), "train_accuracy": accuracy_score(train_actual, train_predicted), "validation_loss": validation_loss, "validation_accuracy": validation_accuracy, "validation_macro_f1": validation_macro_f1, "validation_weighted_f1": validation_weighted_f1, "learning_rate": optimizer.param_groups[0]["lr"], "elapsed_seconds": round(time.perf_counter() - started, 2)}
            save_history(row)
            print(f"[{EXPERIMENT_NAME}] epoch={epoch} train_loss={row['train_loss']:.4f} train_accuracy={row['train_accuracy']:.4f} validation_loss={validation_loss:.4f} validation_accuracy={validation_accuracy:.4f} validation_macro_f1={validation_macro_f1:.4f}", flush=True)
            if validation_macro_f1 > best["macro_f1"]:
                best = {"epoch": epoch, "accuracy": validation_accuracy, "macro_f1": validation_macro_f1, "weighted_f1": validation_weighted_f1, "matrix": matrix}
                torch.save({"model_state_dict": model.state_dict(), "class_names": CLASS_NAMES, "epoch": epoch, "validation_accuracy": validation_accuracy, "validation_macro_f1": validation_macro_f1, "validation_weighted_f1": validation_weighted_f1, "experiment_name": EXPERIMENT_NAME}, CHECKPOINT_PATH)
                stale = 0
            else:
                stale += 1
                if stale >= 5:
                    print(f"Early stopping at epoch {epoch}; patience reached.", flush=True)
                    break
        if best["matrix"] is None:
            raise RuntimeError("No best validation checkpoint was produced")
        write_confusion_matrix(np.asarray(best["matrix"]))
        final_payload = {"experiment": EXPERIMENT_NAME, "best_epoch": best["epoch"], "best_validation_accuracy": best["accuracy"], "best_validation_macro_f1": best["macro_f1"], "best_validation_weighted_f1": best["weighted_f1"], "baseline_validation_accuracy": BASELINE_VALIDATION_ACCURACY, "baseline_validation_macro_f1": BASELINE_VALIDATION_MACRO_F1, "test_used": False, "dataset_modified": False, "baseline_checkpoint_modified": False, "checkpoint": str(CHECKPOINT_PATH)}
        FINAL_REPORT_PATH.write_text(json.dumps(final_payload, indent=2) + "\n", encoding="utf-8")
        print("=" * 50)
        print("TARGETED AUGMENTATION V2 — FINAL RESULT")
        print("=" * 50)
        print(f"\nExperiment:\n{EXPERIMENT_NAME}\n\nBaseline Validation Accuracy:\n{BASELINE_VALIDATION_ACCURACY * 100:.2f}%\n\nBaseline Validation Macro F1:\n{BASELINE_VALIDATION_MACRO_F1 * 100:.2f}%\n\nNew Validation Accuracy:\n{best['accuracy'] * 100:.2f}%\n\nNew Validation Macro F1:\n{best['macro_f1'] * 100:.2f}%\n\nNew Validation Weighted F1:\n{best['weighted_f1'] * 100:.2f}%\n\nBest Epoch:\n{best['epoch']}\n\nMacro F1 Improvement:\n{best['macro_f1'] - BASELINE_VALIDATION_MACRO_F1:.6f}\n\nAccuracy Improvement:\n{best['accuracy'] - BASELINE_VALIDATION_ACCURACY:.6f}\n\nImproved:\n{'YES' if best['macro_f1'] > BASELINE_VALIDATION_MACRO_F1 else 'NO'}\n\nCheckpoint:\n{CHECKPOINT_PATH}\n\nHistory:\n{HISTORY_PATH}\n\nFinal Report:\n{FINAL_REPORT_PATH}\n\nTest Set:\nNOT USED\n\nDataset:\nNOT MODIFIED\n\nBaseline Checkpoint:\nUNCHANGED\n")
        print("=" * 50)
        return 0
    except Exception:
        ERROR_PATH.write_text(traceback.format_exc() + f"\nLast completed epoch: {last_epoch}\nLast validation Macro F1: {last_validation_macro_f1}\nLast validation accuracy: {last_validation_accuracy}\n", encoding="utf-8")
        print(traceback.format_exc(), flush=True)
        print(f"Last completed epoch: {last_epoch}", flush=True)
        print(f"Last validation Macro F1: {last_validation_macro_f1}", flush=True)
        print(f"Last validation accuracy: {last_validation_accuracy}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())