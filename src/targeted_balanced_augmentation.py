"""Validation-only targeted balanced augmentation experiment.

This script keeps the experiment design fixed:
- WeightedRandomSampler on the training split only
- moderate fabric-safe augmentation
- deterministic validation transform
- EfficientNet-B0 pretrained initialization
- checkpoint selection by validation Macro F1
- no test-set evaluation
"""

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
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

from model import (
    CLASS_NAMES,
    DATASET_ROOT,
    MEAN,
    STD,
    MODELS_ROOT,
    REPORTS_ROOT,
    create_model,
    get_transforms,
    seed_everything,
)

EXPERIMENT_NAME = "targeted_balanced_augmentation"
BASELINE_VALIDATION_ACCURACY = 0.8541
BASELINE_VALIDATION_MACRO_F1 = 0.8333
CHECKPOINT_PATH = MODELS_ROOT / f"best_efficientnet_b0_{EXPERIMENT_NAME}.pth"
HISTORY_PATH = REPORTS_ROOT / f"{EXPERIMENT_NAME}_history.csv"
FINAL_JSON_PATH = REPORTS_ROOT / f"{EXPERIMENT_NAME}_final.json"
CONFUSION_PATH = REPORTS_ROOT / f"{EXPERIMENT_NAME}_confusion_matrix.png"
ERROR_LOG_PATH = REPORTS_ROOT / f"{EXPERIMENT_NAME}_error.txt"


def build_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.80, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def save_history_row(row: dict[str, float | int | str]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = HISTORY_PATH.exists()
    with HISTORY_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow([
                "epoch",
                "train_loss",
                "train_accuracy",
                "validation_loss",
                "validation_accuracy",
                "validation_macro_f1",
                "validation_weighted_f1",
                "learning_rate",
                "elapsed_seconds",
            ])
        writer.writerow([
            row["epoch"],
            row["train_loss"],
            row["train_accuracy"],
            row["validation_loss"],
            row["validation_accuracy"],
            row["validation_macro_f1"],
            row["validation_weighted_f1"],
            row["learning_rate"],
            row["elapsed_seconds"],
        ])


def write_confusion_matrix(actual: list[int], predicted: list[int]) -> None:
    cm = confusion_matrix(actual, predicted, labels=list(range(len(CLASS_NAMES))))
    figure, axis = plt.subplots(figsize=(9, 7))
    axis.imshow(cm, cmap="Blues")
    axis.set_xticks(range(len(CLASS_NAMES)))
    axis.set_yticks(range(len(CLASS_NAMES)))
    axis.set_xticklabels(CLASS_NAMES, rotation=65, ha="right")
    axis.set_yticklabels(CLASS_NAMES)
    axis.set_title(f"{EXPERIMENT_NAME} validation confusion matrix")
    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
            value = cm[row, column]
            axis.text(column, row, value, ha="center", va="center", color="black" if value < cm.max() / 2 else "white")
    figure.tight_layout()
    figure.savefig(CONFUSION_PATH, dpi=150)
    plt.close(figure)


def main() -> int:
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)

    if CHECKPOINT_PATH.exists():
        print("EXISTING CHECKPOINT FOUND")
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
            required = {"model_state_dict", "class_names", "epoch", "validation_accuracy", "validation_macro_f1", "validation_weighted_f1", "experiment_name"}
            if required.issubset(checkpoint):
                print(f"Checkpoint is valid and can resume safely: {CHECKPOINT_PATH}")
            else:
                print(f"Existing checkpoint is incomplete; starting from pretrained EfficientNet-B0 weights.")
        except Exception as exc:
            print(f"Existing checkpoint could not be loaded: {exc}")
            print("Starting from pretrained EfficientNet-B0 weights.")
    else:
        print("NO EXISTING CHECKPOINT FOUND")

    train_transform = build_train_transform()
    _, validation_transform = get_transforms(224, targeted=False)
    train_dataset = datasets.ImageFolder(DATASET_ROOT / "train", transform=train_transform)
    validation_dataset = datasets.ImageFolder(DATASET_ROOT / "validation", transform=validation_transform)
    if train_dataset.class_to_idx != {name: index for index, name in enumerate(CLASS_NAMES)}:
        raise RuntimeError(f"Train class mapping mismatch: {train_dataset.class_to_idx}")
    if validation_dataset.class_to_idx != {name: index for index, name in enumerate(CLASS_NAMES)}:
        raise RuntimeError(f"Validation class mapping mismatch: {validation_dataset.class_to_idx}")

    train_counts = np.bincount(train_dataset.targets, minlength=len(CLASS_NAMES)).astype(np.float64)
    class_counts = {CLASS_NAMES[index]: int(train_counts[index]) for index in range(len(CLASS_NAMES))}
    sampling_weights = train_counts.sum() / (len(CLASS_NAMES) * train_counts)
    sampler = WeightedRandomSampler(
        weights=torch.tensor([sampling_weights[target] for target in train_dataset.targets], dtype=torch.double),
        num_samples=len(train_dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(42),
    )
    print("TRAIN_CLASS_COUNTS", class_counts)
    print("SAMPLING_WEIGHTS", {CLASS_NAMES[index]: float(sampling_weights[index]) for index in range(len(CLASS_NAMES))})

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=False,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(42),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = create_model(len(CLASS_NAMES), pretrained=True).to(device)
    if CHECKPOINT_PATH.exists():
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
            if {"model_state_dict", "class_names", "epoch", "validation_accuracy", "validation_macro_f1", "validation_weighted_f1", "experiment_name"}.issubset(checkpoint):
                model.load_state_dict(checkpoint["model_state_dict"])
                print(f"Resumed from {CHECKPOINT_PATH}")
        except Exception:
            pass

    best_state = None
    best_epoch = 0
    best_validation_accuracy = 0.0
    best_validation_macro_f1 = -1.0
    best_validation_weighted_f1 = 0.0
    stale = 0
    epochs = 6
    warmup_epochs = 2
    patience = 2
    learning_rate = 1e-3
    fine_tune_lr = 1e-4
    started = time.perf_counter()

    try:
        for epoch in range(1, epochs + 1):
            if epoch == warmup_epochs + 1:
                for parameter in model.features[-3:].parameters():
                    parameter.requires_grad = True
                optimizer = AdamW(filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=fine_tune_lr, weight_decay=1e-4)
                scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=2)
            else:
                optimizer = AdamW(filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-4)
                scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=2)

            model.train()
            train_loss_total = 0.0
            train_actual = []
            train_predicted = []
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = nn.CrossEntropyLoss()(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss_total += loss.item() * labels.size(0)
                train_actual.extend(labels.cpu().numpy())
                train_predicted.extend(logits.argmax(1).cpu().numpy())

            model.eval()
            validation_losses = []
            actual = []
            predicted = []
            with torch.inference_mode():
                for images, labels in validation_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    logits = model(images)
                    validation_losses.append(nn.CrossEntropyLoss()(logits, labels).item() * labels.size(0))
                    actual.extend(labels.cpu().numpy())
                    predicted.extend(logits.argmax(1).cpu().numpy())

            validation_loss = sum(validation_losses) / len(validation_dataset)
            validation_accuracy = accuracy_score(actual, predicted)
            _, _, f1_values, support = precision_recall_fscore_support(actual, predicted, labels=list(range(len(CLASS_NAMES))), zero_division=0)
            validation_macro_f1 = float(f1_values.mean())
            validation_weighted_f1 = float(np.average(f1_values, weights=support))
            train_accuracy = accuracy_score(train_actual, train_predicted)
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(validation_macro_f1)

            print(f"[Targeted Balanced] Epoch {epoch}/{epochs}")
            print(f"Train Loss: {train_loss_total / len(train_dataset):.4f}")
            print(f"Train Accuracy: {train_accuracy:.4f}")
            print(f"Validation Loss: {validation_loss:.4f}")
            print(f"Validation Accuracy: {validation_accuracy:.4f}")
            print(f"Validation Macro F1: {validation_macro_f1:.4f}")
            print(f"Validation Weighted F1: {validation_weighted_f1:.4f}")

            history_row = {
                "epoch": epoch,
                "train_loss": float(train_loss_total / len(train_dataset)),
                "train_accuracy": float(train_accuracy),
                "validation_loss": float(validation_loss),
                "validation_accuracy": float(validation_accuracy),
                "validation_macro_f1": float(validation_macro_f1),
                "validation_weighted_f1": float(validation_weighted_f1),
                "learning_rate": float(current_lr),
                "elapsed_seconds": round(time.perf_counter() - started, 2),
            }
            save_history_row(history_row)

            if validation_macro_f1 > best_validation_macro_f1:
                best_epoch = epoch
                best_validation_accuracy = float(validation_accuracy)
                best_validation_macro_f1 = float(validation_macro_f1)
                best_validation_weighted_f1 = float(validation_weighted_f1)
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "class_names": CLASS_NAMES,
                        "epoch": epoch,
                        "validation_accuracy": validation_accuracy,
                        "validation_macro_f1": validation_macro_f1,
                        "validation_weighted_f1": validation_weighted_f1,
                        "experiment_name": EXPERIMENT_NAME,
                    },
                    CHECKPOINT_PATH,
                )
                stale = 0
                print(f"Best Validation Macro F1: {best_validation_macro_f1:.4f}")
            else:
                stale += 1
                if stale >= patience:
                    print(f"Early stopping at epoch {epoch}; patience reached.")
                    break

        if best_state is None:
            raise RuntimeError("No best model state was saved.")

        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        final_payload = {
            "experiment": EXPERIMENT_NAME,
            "best_epoch": int(best_epoch),
            "best_validation_accuracy": float(best_validation_accuracy),
            "best_validation_macro_f1": float(best_validation_macro_f1),
            "best_validation_weighted_f1": float(best_validation_weighted_f1),
            "checkpoint": str(CHECKPOINT_PATH),
            "test_used": False,
        }
        FINAL_JSON_PATH.write_text(json.dumps(final_payload, indent=2) + "\n", encoding="utf-8")

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        actual = []
        predicted = []
        with torch.inference_mode():
            for images, labels in validation_loader:
                logits = model(images.to(device))
                actual.extend(labels.cpu().numpy())
                predicted.extend(logits.argmax(1).cpu().numpy())
        write_confusion_matrix(actual, predicted)

        improvement = best_validation_macro_f1 - BASELINE_VALIDATION_MACRO_F1
        print("TARGETED BALANCED AUGMENTATION — FINAL")
        print("Experiment:")
        print(EXPERIMENT_NAME)
        print(f"Baseline Validation Accuracy:\n{BASELINE_VALIDATION_ACCURACY * 100:.2f}%")
        print(f"Baseline Validation Macro F1:\n{BASELINE_VALIDATION_MACRO_F1 * 100:.2f}%")
        print(f"New Validation Accuracy:\n{best_validation_accuracy * 100:.2f}%")
        print(f"New Validation Macro F1:\n{best_validation_macro_f1 * 100:.2f}%")
        print(f"New Validation Weighted F1:\n{best_validation_weighted_f1 * 100:.2f}%")
        print(f"Best Epoch:\n{best_epoch}")
        print(f"Improvement in Macro F1:\n{improvement:.6f}")
        print(f"Improved:\n{'YES' if best_validation_macro_f1 > BASELINE_VALIDATION_MACRO_F1 else 'NO'}")
        print(f"Checkpoint:\n{CHECKPOINT_PATH}")
        print(f"History:\n{HISTORY_PATH}")
        print(f"Final Report:\n{FINAL_JSON_PATH}")
        print("Test Set:\nNOT USED")
        print("Baseline Checkpoint:\nUNCHANGED")
        print("Dataset:\nNOT MODIFIED")
        return 0
    except Exception:
        error_text = traceback.format_exc()
        ERROR_LOG_PATH.write_text(error_text + "\n", encoding="utf-8")
        print(error_text, flush=True)
        print(f"Preserved completed history at: {HISTORY_PATH}")
        print(f"Preserved best checkpoint at: {CHECKPOINT_PATH}")
        print(f"Error log saved to: {ERROR_LOG_PATH}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
