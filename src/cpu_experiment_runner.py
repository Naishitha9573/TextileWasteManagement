"""CPU-only EfficientNet-B0 experiment runner for train/validation experiments."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets

from model import CLASS_NAMES, DATASET_ROOT, IMAGE_SIZE, create_model, get_transforms, seed_everything

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "finetuning" / "cpu_smoke_test"
PROTECTED_CHECKPOINT = ROOT / "models" / "best_efficientnet_b0.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a CPU-only EfficientNet-B0 experiment.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default="adamw")
    parser.add_argument("--scheduler", choices=("cosine", "plateau"), default="plateau")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def validate_dataset(train_dataset: datasets.ImageFolder, validation_dataset: datasets.ImageFolder) -> None:
    expected = {name: index for index, name in enumerate(CLASS_NAMES)}
    if train_dataset.class_to_idx != expected or validation_dataset.class_to_idx != expected:
        raise RuntimeError("Train/validation class mapping does not match the protected 9-class mapping")


def metrics(actual: list[int], predicted: list[int]) -> tuple[float, float, float, dict]:
    report = classification_report(actual, predicted, labels=list(range(len(CLASS_NAMES))), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    return accuracy_score(actual, predicted), report["macro avg"]["f1-score"], report["weighted avg"]["f1-score"], report


def run_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device, optimizer: torch.optim.Optimizer | None) -> tuple[float, float, float, float, dict, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    actual: list[int] = []
    predicted: list[int] = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_fn(logits, labels)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * labels.size(0)
        actual.extend(labels.cpu().tolist())
        predicted.extend(logits.argmax(1).detach().cpu().tolist())
    accuracy, macro_f1, weighted_f1, report = metrics(actual, predicted)
    matrix = confusion_matrix(actual, predicted, labels=list(range(len(CLASS_NAMES))))
    return total_loss / len(loader.dataset), accuracy, macro_f1, weighted_f1, report, matrix


def main() -> int:
    config = parse_args()
    if config.epochs < 1 or config.batch_size < 1 or config.num_workers < 0:
        raise SystemExit("epochs and batch-size must be positive; num-workers must not be negative")
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {config.output_dir}")
    if not PROTECTED_CHECKPOINT.is_file():
        raise SystemExit(f"Protected checkpoint is missing: {PROTECTED_CHECKPOINT}")
    if not (DATASET_ROOT / "train").is_dir() or not (DATASET_ROOT / "validation").is_dir():
        raise SystemExit(f"Train/validation dataset directories are missing under {DATASET_ROOT}")

    torch.manual_seed(config.seed)
    device = torch.device("cpu")
    train_transform, validation_transform = get_transforms(IMAGE_SIZE)
    train_dataset = datasets.ImageFolder(DATASET_ROOT / "train", transform=train_transform)
    validation_dataset = datasets.ImageFolder(DATASET_ROOT / "validation", transform=validation_transform)
    validate_dataset(train_dataset, validation_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=False, generator=torch.Generator().manual_seed(config.seed))
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=False)
    model = create_model(len(CLASS_NAMES), pretrained=not config.no_pretrained).to(device)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    optimizer_type = Adam if config.optimizer == "adam" else AdamW
    optimizer = optimizer_type(filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=2) if config.scheduler == "plateau" else CosineAnnealingLR(optimizer, T_max=config.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    config.output_dir.mkdir(parents=True, exist_ok=False)
    history: list[dict[str, object]] = []
    best_macro_f1 = -1.0
    for epoch in range(1, config.epochs + 1):
        epoch_started = time.perf_counter()
        train_loss, train_accuracy, _, _, _, _ = run_epoch(model, train_loader, loss_fn, device, optimizer)
        with torch.inference_mode():
            validation_loss, validation_accuracy, validation_macro_f1, validation_weighted_f1, report, matrix = run_epoch(model, validation_loader, loss_fn, device, None)
        if config.scheduler == "plateau":
            scheduler.step(validation_macro_f1)
        else:
            scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy, "validation_loss": validation_loss, "validation_accuracy": validation_accuracy, "validation_macro_f1": validation_macro_f1, "validation_weighted_f1": validation_weighted_f1, "learning_rate": optimizer.param_groups[0]["lr"], "elapsed_seconds": round(time.perf_counter() - epoch_started, 2)}
        history.append(row)
        print(row, flush=True)
        if validation_macro_f1 > best_macro_f1:
            best_macro_f1 = validation_macro_f1
            torch.save({"model_state_dict": model.state_dict(), "class_names": CLASS_NAMES, "best_validation_macro_f1": best_macro_f1, "best_validation_accuracy": validation_accuracy, "best_validation_weighted_f1": validation_weighted_f1, "best_epoch": epoch}, config.output_dir / "best_efficientnet_b0_cpu.pth")
            (config.output_dir / "classification_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            np.savetxt(config.output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    with (config.output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    (config.output_dir / "config.json").write_text(json.dumps({**vars(config), "output_dir": str(config.output_dir), "device": str(device), "dataset_root": str(DATASET_ROOT), "protected_checkpoint": str(PROTECTED_CHECKPOINT), "test_used": False}, indent=2, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())