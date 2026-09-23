"""Sequential CPU-only fine-tuning experiments for the protected class-aware baseline."""

from __future__ import annotations

import argparse
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import CLASS_NAMES, DATASET_ROOT, IMAGE_SIZE, MEAN, MODELS_ROOT, STD, create_model, seed_everything

ROOT = Path(__file__).resolve().parents[1]
BASELINE_CHECKPOINT = MODELS_ROOT / "best_efficientnet_b0_class_aware_finetune_v1.pth"
PROTECTED_CHECKPOINTS = (MODELS_ROOT / "best_efficientnet_b0.pth", BASELINE_CHECKPOINT)
BASELINE_METRICS = {"validation_accuracy": 0.8851351351351351, "validation_macro_f1": 0.8667165937848992, "validation_weighted_f1": 0.8877556310140541}
FINE_TUNING_DEPTH_FEATURE_INDICES = {
    "classifier_only": (),
    "last_block": (8,),
    "last_2_blocks": (7, 8),
    "last_3_blocks": (6, 7, 8),
}
DEFAULT_FINE_TUNING_DEPTH = "last_3_blocks"
SUPPORTED_OPTIMIZERS = ("adamw", "adam", "sgd")
DEFAULT_OPTIMIZER = "adamw"
SUPPORTED_DROPOUTS = (0.0, 0.2, 0.3, 0.5)
DEFAULT_DROPOUT = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one CPU-only learning-rate experiment.")
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--classifier-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--fine-tuning-depth", choices=tuple(FINE_TUNING_DEPTH_FEATURE_INDICES), default=DEFAULT_FINE_TUNING_DEPTH)
    parser.add_argument("--optimizer", choices=SUPPORTED_OPTIMIZERS, default=DEFAULT_OPTIMIZER)
    parser.add_argument("--dropout", type=float, choices=SUPPORTED_DROPOUTS, default=DEFAULT_DROPOUT)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def validation_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(round(IMAGE_SIZE * 256 / 224)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def validate_inputs(config: argparse.Namespace) -> None:
    if config.epochs < 1 or config.warmup_epochs >= config.epochs or config.patience < 1 or config.num_workers < 0:
        raise SystemExit("Invalid epoch, warmup, patience, or worker configuration")
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {config.output_dir}")
    if not all(path.is_file() for path in PROTECTED_CHECKPOINTS):
        raise SystemExit("One or more protected checkpoints are missing")
    if not all((DATASET_ROOT / split).is_dir() for split in ("train", "validation", "test")):
        raise SystemExit(f"Dataset splits are missing under {DATASET_ROOT}")


def class_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    counts = np.bincount(dataset.targets, minlength=len(CLASS_NAMES)).astype(np.float64)
    inverse = counts.sum() / (len(CLASS_NAMES) * counts)
    weights = np.sqrt(inverse)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def validate_mapping(train_dataset: datasets.ImageFolder, validation_dataset: datasets.ImageFolder) -> None:
    expected = {name: index for index, name in enumerate(CLASS_NAMES)}
    if len(CLASS_NAMES) != 9 or train_dataset.class_to_idx != expected or validation_dataset.class_to_idx != expected:
        raise RuntimeError(f"Class mapping mismatch: train={train_dataset.class_to_idx}, validation={validation_dataset.class_to_idx}, expected={expected}")


def fine_tuning_feature_indices(depth: str) -> tuple[int, ...]:
    try:
        return FINE_TUNING_DEPTH_FEATURE_INDICES[depth]
    except KeyError as exc:
        raise ValueError(f"Unsupported fine-tuning depth: {depth}") from exc


def set_trainable_layers(model: nn.Module, depth: str) -> tuple[int, ...]:
    feature_indices = fine_tuning_feature_indices(depth)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    for feature_index in feature_indices:
        for parameter in model.features[feature_index].parameters():
            parameter.requires_grad = True
    return feature_indices


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def optimizer_parameter_groups(model: nn.Module, feature_indices: tuple[int, ...], classifier_learning_rate: float, feature_learning_rate: float) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = [{"params": [parameter for parameter in model.classifier.parameters() if parameter.requires_grad], "lr": classifier_learning_rate}]
    feature_parameters = [parameter for feature_index in feature_indices for parameter in model.features[feature_index].parameters() if parameter.requires_grad]
    if feature_parameters:
        groups.append({"params": feature_parameters, "lr": feature_learning_rate})
    return groups


def build_optimizer(optimizer_name: str, parameter_groups: list[dict[str, object]], weight_decay: float) -> torch.optim.Optimizer:
    optimizer_types = {"adamw": AdamW, "adam": Adam, "sgd": SGD}
    try:
        optimizer_type = optimizer_types[optimizer_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}") from exc
    optimizer_kwargs: dict[str, object] = {"params": parameter_groups, "weight_decay": weight_decay}
    if optimizer_name == "sgd":
        optimizer_kwargs["momentum"] = 0.0
    return optimizer_type(**optimizer_kwargs)


def run_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device, optimizer: torch.optim.Optimizer | None) -> tuple[float, float, float, float, float, float, dict, np.ndarray, np.ndarray]:
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
    precision, recall, f1, support = precision_recall_fscore_support(actual, predicted, labels=list(range(len(CLASS_NAMES))), zero_division=0)
    report = classification_report(actual, predicted, labels=list(range(len(CLASS_NAMES))), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    matrix = confusion_matrix(actual, predicted, labels=list(range(len(CLASS_NAMES))))
    return (total_loss / len(loader.dataset), accuracy_score(actual, predicted), float(precision.mean()), float(recall.mean()), float(f1.mean()), float(np.average(f1, weights=support)), report, matrix, f1)


def save_confusion_matrix(matrix: np.ndarray, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 7))
    axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=range(len(CLASS_NAMES)), yticks=range(len(CLASS_NAMES)), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, xlabel="Predicted", ylabel="Actual", title="Validation confusion matrix")
    axis.tick_params(axis="x", labelrotation=65)
    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
            axis.text(column, row, matrix[row, column], ha="center", va="center")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_history(history: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def build_best_metrics(config: argparse.Namespace, model: nn.Module, device: torch.device, best_epoch: int, best_key: tuple[float, float, float], history: list[dict[str, object]], selected_feature_indices: tuple[int, ...], elapsed_seconds: float) -> dict[str, object]:
    best_weighted_f1 = next(row["validation_weighted_f1"] for row in history if row["epoch"] == best_epoch)
    return {
        "experiment": f"LR-{config.learning_rate:g}",
        "learning_rate": config.learning_rate,
        "optimizer": config.optimizer,
        "dropout": config.dropout,
        "fine_tuning_depth": config.fine_tuning_depth,
        "trainable_feature_indices": list(selected_feature_indices),
        "trainable_parameter_count": trainable_parameter_count(model),
        "best_epoch": best_epoch,
        "validation_accuracy": best_key[1],
        "validation_macro_f1": best_key[0],
        "validation_weighted_f1": best_weighted_f1,
        "baseline_validation_accuracy": BASELINE_METRICS["validation_accuracy"],
        "baseline_validation_macro_f1": BASELINE_METRICS["validation_macro_f1"],
        "baseline_validation_weighted_f1": BASELINE_METRICS["validation_weighted_f1"],
        "improvement_accuracy_percentage_points": (best_key[1] - BASELINE_METRICS["validation_accuracy"]) * 100,
        "improvement_macro_f1_percentage_points": (best_key[0] - BASELINE_METRICS["validation_macro_f1"]) * 100,
        "improvement_weighted_f1_percentage_points": (best_weighted_f1 - BASELINE_METRICS["validation_weighted_f1"]) * 100,
        "device": str(device),
        "dataset_root": str(DATASET_ROOT),
        "checkpoint_source": str(BASELINE_CHECKPOINT),
        "test_used": False,
        "elapsed_seconds": round(elapsed_seconds, 2),
    }


def main() -> int:
    config = parse_args()
    validate_inputs(config)
    seed_everything(config.seed)
    device = torch.device("cpu")
    train_dataset = datasets.ImageFolder(DATASET_ROOT / "train", transform=train_transform())
    validation_dataset = datasets.ImageFolder(DATASET_ROOT / "validation", transform=validation_transform())
    validate_mapping(train_dataset, validation_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=False, generator=torch.Generator().manual_seed(config.seed))
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=False)
    model = create_model(len(CLASS_NAMES), pretrained=False, dropout=config.dropout).to(device)
    baseline = torch.load(BASELINE_CHECKPOINT, map_location="cpu", weights_only=False)
    if baseline.get("class_names") != CLASS_NAMES:
        raise RuntimeError("Protected baseline class mapping mismatch")
    model.load_state_dict(baseline["model_state_dict"])
    selected_feature_indices = set_trainable_layers(model, "classifier_only")
    weights = class_weights(train_dataset).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=config.label_smoothing)
    optimizer = build_optimizer(config.optimizer, optimizer_parameter_groups(model, selected_feature_indices, config.classifier_learning_rate, config.learning_rate), config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, config.epochs - config.warmup_epochs), eta_min=config.learning_rate / 10)
    config.output_dir.mkdir(parents=True, exist_ok=False)
    history: list[dict[str, object]] = []
    best_key = (-1.0, -1.0, -1.0)
    best_epoch = 0
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        epoch_started = time.perf_counter()
        if epoch == config.warmup_epochs + 1:
            stale = 0
            selected_feature_indices = set_trainable_layers(model, config.fine_tuning_depth)
            optimizer = build_optimizer(config.optimizer, optimizer_parameter_groups(model, selected_feature_indices, config.classifier_learning_rate, config.learning_rate), config.weight_decay)
            scheduler = CosineAnnealingLR(optimizer, T_max=max(1, config.epochs - config.warmup_epochs), eta_min=config.learning_rate / 10)
        train_loss, train_accuracy, _, _, _, _, _, _, _ = run_epoch(model, train_loader, loss_fn, device, optimizer)
        with torch.inference_mode():
            validation_loss, validation_accuracy, macro_precision, macro_recall, validation_macro_f1, validation_weighted_f1, report, matrix, per_class_f1 = run_epoch(model, validation_loader, loss_fn, device, None)
        scheduler.step()
        elapsed = round(time.perf_counter() - epoch_started, 2)
        row = {"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy, "validation_loss": validation_loss, "validation_accuracy": validation_accuracy, "validation_macro_precision": macro_precision, "validation_macro_recall": macro_recall, "validation_macro_f1": validation_macro_f1, "validation_weighted_f1": validation_weighted_f1, "learning_rate": config.learning_rate if epoch > config.warmup_epochs else config.classifier_learning_rate, "epoch_time_seconds": elapsed}
        history.append(row)
        save_history(history, config.output_dir / "training_history.csv")
        print(f"LR={config.learning_rate} {row}", flush=True)
        if epoch <= config.warmup_epochs:
            continue
        key = (validation_macro_f1, validation_accuracy, float(np.mean(per_class_f1)))
        if key > best_key:
            best_key = key
            best_epoch = epoch
            stale = 0
            torch.save({"model_state_dict": model.state_dict(), "class_names": CLASS_NAMES, "experiment": f"lr_{config.learning_rate:g}", "best_epoch": epoch, "best_validation_accuracy": validation_accuracy, "best_validation_macro_f1": validation_macro_f1, "best_validation_weighted_f1": validation_weighted_f1}, config.output_dir / "best_checkpoint.pth")
            (config.output_dir / "classification_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            np.savetxt(config.output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
            save_confusion_matrix(matrix, config.output_dir / "confusion_matrix.png")
        else:
            stale += 1
            if stale >= config.patience:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break
    if best_epoch < config.warmup_epochs + 1:
        raise RuntimeError(f"Invalid best_epoch={best_epoch}; best checkpoint must come from fine-tuning phase.")
    save_history(history, config.output_dir / "training_history.csv")
    selected_feature_indices = fine_tuning_feature_indices(config.fine_tuning_depth)
    best_metrics = build_best_metrics(config, model, device, best_epoch, best_key, history, selected_feature_indices, time.perf_counter() - started)
    (config.output_dir / "metrics.json").write_text(json.dumps(best_metrics, indent=2) + "\n", encoding="utf-8")
    (config.output_dir / "config.json").write_text(json.dumps({**vars(config), "output_dir": str(config.output_dir), "device": str(device), "dataset_root": str(DATASET_ROOT), "checkpoint_source": str(BASELINE_CHECKPOINT), "protected_checkpoints": [str(path) for path in PROTECTED_CHECKPOINTS], "class_names": CLASS_NAMES, "fine_tuning_depth": config.fine_tuning_depth, "trainable_feature_indices": list(selected_feature_indices), "trainable_parameter_count": trainable_parameter_count(model), "test_used": False, "total_elapsed_seconds": round(time.perf_counter() - started, 2)}, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(best_metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise