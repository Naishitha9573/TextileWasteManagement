"""Experiment 1: compare loss weighting strategies using validation data only."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets

from model import CLASS_NAMES, DATASET_ROOT, IMAGE_SIZE, create_model, get_transforms, make_loader, seed_everything, training_class_weights


ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = ROOT / "reports"
MODELS_ROOT = ROOT / "models"
EXPERIMENTS = (
    ("A", "CrossEntropyLoss", "inverse_frequency"),
    ("B", "CrossEntropyLoss", "none"),
    ("C", "CrossEntropyLoss", "sqrt_inverse_frequency"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare class-weighted losses without using the test set.")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def datasets_for_experiment() -> tuple[datasets.ImageFolder, datasets.ImageFolder]:
    train_transform, validation_transform = get_transforms(IMAGE_SIZE)
    train = datasets.ImageFolder(DATASET_ROOT / "train", transform=train_transform)
    validation = datasets.ImageFolder(DATASET_ROOT / "validation", transform=validation_transform)
    expected = {class_name: index for index, class_name in enumerate(CLASS_NAMES)}
    if train.class_to_idx != expected or validation.class_to_idx != expected:
        raise RuntimeError(f"Train/validation class mapping mismatch: {train.class_to_idx}, {validation.class_to_idx}")
    return train, validation


def weights_for_strategy(train: datasets.ImageFolder, strategy: str) -> torch.Tensor:
    inverse = training_class_weights(train)
    if strategy == "none":
        return torch.ones_like(inverse)
    if strategy == "sqrt_inverse_frequency":
        weights = torch.sqrt(inverse)
        return weights / weights.mean()
    return inverse


def evaluate(network: nn.Module, loader, loss_fn: nn.Module, device: torch.device) -> tuple[float, float, float, float, float, float, np.ndarray]:
    network.eval()
    losses, actual, predicted = [], [], []
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = network(images)
            losses.append(loss_fn(logits, labels).item() * labels.size(0))
            actual.extend(labels.cpu().numpy())
            predicted.extend(logits.argmax(1).cpu().numpy())
    return (sum(losses) / len(loader.dataset), accuracy_score(actual, predicted), precision_score(actual, predicted, average="macro", zero_division=0), recall_score(actual, predicted, average="macro", zero_division=0), f1_score(actual, predicted, average="macro", zero_division=0), f1_score(actual, predicted, average="weighted", zero_division=0), confusion_matrix(actual, predicted, labels=range(len(CLASS_NAMES))))


def run_experiment(experiment_id: str, loss_name: str, strategy: str, config: argparse.Namespace, device: torch.device) -> dict[str, object]:
    seed_everything(config.seed)
    train, validation = datasets_for_experiment()
    train_loader = make_loader(train, config.batch_size, True, config.seed, config.num_workers)
    validation_loader = make_loader(validation, config.batch_size, False, config.seed, config.num_workers)
    weights = weights_for_strategy(train, strategy).to(device)
    network = create_model(len(CLASS_NAMES), pretrained=not config.no_pretrained).to(device)
    for parameter in network.features.parameters():
        parameter.requires_grad = False
    optimizer = AdamW(filter(lambda parameter: parameter.requires_grad, network.parameters()), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=2)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best: dict[str, object] = {"validation_macro_f1": -1.0, "validation_accuracy": 0.0, "validation_weighted_f1": 0.0, "epoch": 0, "matrix": None}
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        if epoch == config.warmup_epochs + 1:
            for parameter in network.features[-3:].parameters():
                parameter.requires_grad = True
            optimizer = AdamW(filter(lambda parameter: parameter.requires_grad, network.parameters()), lr=config.fine_tune_learning_rate, weight_decay=1e-4)
            scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=2)
        network.train()
        train_loss_total = 0.0
        for batch_index, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                loss = loss_fn(network(images), labels)
            if device.type == "cuda":
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
            train_loss_total += loss.item() * labels.size(0)
            if batch_index == 1 or batch_index == len(train_loader) or batch_index % 25 == 0:
                print(f"[{experiment_id}] epoch {epoch}/{config.epochs} batch {batch_index}/{len(train_loader)} loss={loss.item():.4f}", flush=True)
        validation_loss, validation_accuracy, validation_precision, validation_recall, validation_macro_f1, validation_weighted_f1, matrix = evaluate(network, validation_loader, loss_fn, device)
        scheduler.step(validation_macro_f1)
        train_loss = train_loss_total / len(train_loader.dataset)
        print(f"[{experiment_id}] epoch {epoch}: train_loss={train_loss:.4f} val_loss={validation_loss:.4f} val_accuracy={validation_accuracy:.4f} val_macro_f1={validation_macro_f1:.4f}", flush=True)
        if validation_macro_f1 > float(best["validation_macro_f1"]):
            best = {"validation_macro_f1": validation_macro_f1, "validation_accuracy": validation_accuracy, "validation_weighted_f1": validation_weighted_f1, "epoch": epoch, "matrix": matrix}
            stale = 0
            torch.save({"model_state_dict": network.state_dict(), "class_names": CLASS_NAMES, "validation_macro_f1": validation_macro_f1, "experiment_id": experiment_id}, MODELS_ROOT / f"experiment_1_{experiment_id}_best.pth")
        else:
            stale += 1
            if stale >= config.patience:
                break
    matrix = np.asarray(best["matrix"])
    figure, axis = plt.subplots(figsize=(10, 8)); image = axis.imshow(matrix, cmap="Blues"); figure.colorbar(image, ax=axis)
    axis.set(xticks=range(len(CLASS_NAMES)), yticks=range(len(CLASS_NAMES)), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, xlabel="Predicted", ylabel="Actual", title=f"Experiment 1 {experiment_id} Validation Confusion Matrix")
    axis.tick_params(axis="x", labelrotation=65)
    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
            axis.text(column, row, matrix[row, column], ha="center", va="center")
    figure.tight_layout(); figure.savefig(REPORTS_ROOT / f"experiment_1_{experiment_id}_validation_confusion_matrix.png", dpi=150); plt.close(figure)
    return {"experiment_id": experiment_id, "loss_function": loss_name, "class_weight_strategy": strategy, "best_validation_accuracy": best["validation_accuracy"], "best_validation_macro_f1": best["validation_macro_f1"], "best_validation_weighted_f1": best["validation_weighted_f1"], "best_epoch": best["epoch"], "training_time": round(time.perf_counter() - started, 2)}


def main() -> int:
    config = parse_args()
    if config.epochs < 1 or config.warmup_epochs >= config.epochs:
        raise SystemExit("epochs must be positive and greater than warmup-epochs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Validation-only Experiment 1; device={device}; seed={config.seed}; test set is not loaded or evaluated.")
    results = [run_experiment(experiment_id, loss_name, strategy, config, device) for experiment_id, loss_name, strategy in EXPERIMENTS]
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    fields = ["experiment_id", "loss_function", "class_weight_strategy", "best_validation_accuracy", "best_validation_macro_f1", "best_validation_weighted_f1", "best_epoch", "training_time"]
    with (REPORTS_ROOT / "experiment_1_loss_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(results)
    print("\nEXPERIMENT 1 — CPU SCREENING RUN")
    print("A: inverse frequency")
    print("B: no weights")
    print("C: sqrt inverse frequency")
    print("\nEXPERIMENT 1 LOSS COMPARISON")
    print("ID Strategy Accuracy Macro-F1 Weighted-F1 Epoch Time(s)")
    for result in results:
        print(f"{result['experiment_id']} {result['class_weight_strategy']} {result['best_validation_accuracy']:.4f} {result['best_validation_macro_f1']:.4f} {result['best_validation_weighted_f1']:.4f} {result['best_epoch']} {result['training_time']}")
        print(f"\nExperiment {result['experiment_id']}")
        print(f"Best epoch: {result['best_epoch']}")
        print(f"Best validation accuracy: {result['best_validation_accuracy']:.4f}")
        print(f"Best validation Macro F1: {result['best_validation_macro_f1']:.4f}")
        print(f"Best validation weighted F1: {result['best_validation_weighted_f1']:.4f}")
        print(f"Training time: {result['training_time']} seconds")
    recommended = max(results, key=lambda result: result["best_validation_macro_f1"])
    print(f"Recommendation for next test evaluation: Experiment {recommended['experiment_id']} ({recommended['class_weight_strategy']}) based on validation Macro F1={recommended['best_validation_macro_f1']:.4f}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())