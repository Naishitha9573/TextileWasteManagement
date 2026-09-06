"""Train the EfficientNet-B0 fabric baseline using validation-only checkpoint selection."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.data import Subset

from model import CLASS_NAMES, MODELS_ROOT, REPORTS_ROOT, create_model, leakage_checks, make_datasets, make_loader, seed_everything, training_class_weights


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 baseline.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", "--head-lr", dest="learning_rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", "--fine-tune-lr", dest="fine_tune_learning_rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-samples", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true", help="Use ordinary CrossEntropyLoss without class weights.")
    parser.add_argument("--checkpoint-name", default="best_efficientnet_b0.pth")
    parser.add_argument("--history-name", default="training_history.csv")
    parser.add_argument("--plot-prefix", default="")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def run_epoch(network, loader, loss_fn, device, optimizer=None, scaler=None, amp_enabled=False, label="train"):
    training = optimizer is not None
    network.train(training)
    losses, actual, predicted = [], [], []
    started = time.perf_counter()
    for batch_index, (images, labels) in enumerate(loader, start=1):
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(training), torch.autocast(device_type="cuda", enabled=amp_enabled):
            logits = network(images)
            loss = loss_fn(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                else:
                    loss.backward(); optimizer.step()
        if training and (batch_index == 1 or batch_index == len(loader) or batch_index % 10 == 0):
            print(f"  {label} batch {batch_index}/{len(loader)} loss={loss.item():.4f} elapsed={time.perf_counter() - started:.1f}s", flush=True)
        losses.append(loss.item() * labels.size(0))
        actual.extend(labels.cpu().numpy())
        predicted.extend(logits.argmax(1).detach().cpu().numpy())
    return sum(losses) / len(loader.dataset), accuracy_score(actual, predicted), precision_score(actual, predicted, average="macro", zero_division=0), recall_score(actual, predicted, average="macro", zero_division=0), f1_score(actual, predicted, average="macro", zero_division=0)


def balanced_smoke_indices(targets: list[int], limit: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    by_class = {class_index: np.flatnonzero(np.asarray(targets) == class_index).tolist() for class_index in range(len(CLASS_NAMES))}
    per_class = max(1, limit // len(CLASS_NAMES))
    selected = []
    for class_index in range(len(CLASS_NAMES)):
        rng.shuffle(by_class[class_index])
        selected.extend(by_class[class_index][:per_class])
    return selected


def main() -> int:
    config = args()
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.num_workers < 0 or config.image_size < 32 or config.smoke_samples < 1:
        raise SystemExit("num-workers must be >= 0, image-size must be >= 32, and smoke-samples must be positive")
    batch_size = config.batch_size or (64 if device.type == "cuda" else 16)
    epochs = 1 if config.smoke_test else config.epochs
    if config.smoke_test:
        print(f"SMOKE TEST: exactly one epoch on up to {config.smoke_samples} training samples; no checkpoint or test evaluation will be performed.")
    checks = leakage_checks()
    datasets_by_split, _ = make_datasets(image_size=config.image_size, include_test=False)
    if config.smoke_test:
        indices = balanced_smoke_indices(datasets_by_split["train"].targets, min(config.smoke_samples, len(datasets_by_split["train"])), config.seed)
        datasets_by_split["train"] = Subset(datasets_by_split["train"], indices)
    loaders = {split: make_loader(dataset, batch_size, split == "train", config.seed, config.num_workers) for split, dataset in datasets_by_split.items() if split != "test" or not config.smoke_test}
    weights = training_class_weights(datasets_by_split["train"] if hasattr(datasets_by_split["train"], "targets") else datasets_by_split["train"].dataset).to(device)
    if config.smoke_test:
        base_targets = datasets_by_split["train"].dataset.targets
        weights = training_class_weights(type("SubsetDataset", (), {"targets": [base_targets[index] for index in datasets_by_split["train"].indices]})()).to(device)
    network = create_model(len(CLASS_NAMES), pretrained=not config.no_pretrained).to(device)
    for parameter in network.features.parameters():
        parameter.requires_grad = False
    optimizer = AdamW(filter(lambda parameter: parameter.requires_grad, network.parameters()), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=2)
    loss_fn = nn.CrossEntropyLoss(weight=None if config.no_class_weights else weights)
    print(f"DEVICE: {'CUDA' if device.type == 'cuda' else 'CPU'}\nTRAIN IMAGES: {len(datasets_by_split['train'])}\nVALIDATION IMAGES: {len(datasets_by_split['validation'])}\nTEST IMAGES: not loaded during training\nCLASSES: {len(CLASS_NAMES)}\nBATCH SIZE: {batch_size}\nIMAGE SIZE: {config.image_size}")
    print(f"class_weight_strategy={'none' if config.no_class_weights else 'inverse_frequency'}")
    print(f"class_weights={weights.detach().cpu().tolist() if not config.no_class_weights else 'not used'}")
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history, best_f1, stale = [], -1.0, 0
    for epoch in range(1, epochs + 1):
        if epoch == config.warmup_epochs + 1 and not config.smoke_test:
            for parameter in network.features[-3:].parameters():
                parameter.requires_grad = True
            optimizer = AdamW(filter(lambda parameter: parameter.requires_grad, network.parameters()), lr=config.fine_tune_learning_rate, weight_decay=1e-4)
            scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=2)
        epoch_started = time.perf_counter()
        train_metrics = run_epoch(network, loaders["train"], loss_fn, device, optimizer, scaler, amp_enabled, "train")
        with torch.inference_mode():
            validation_metrics = run_epoch(network, loaders["validation"], loss_fn, device, label="validation")
        learning_rate = optimizer.param_groups[0]["lr"]
        row = [epoch, train_metrics[0], train_metrics[1], validation_metrics[0], validation_metrics[1], validation_metrics[2], validation_metrics[3], validation_metrics[4], learning_rate]
        history.append(row)
        scheduler.step(validation_metrics[4])
        print(f"Epoch {epoch}/{epochs} Train Loss {train_metrics[0]:.4f} Train Accuracy {train_metrics[1]:.4f} Validation Loss {validation_metrics[0]:.4f} Validation Accuracy {validation_metrics[1]:.4f} Validation Macro F1 {validation_metrics[4]:.4f} elapsed={time.perf_counter() - epoch_started:.1f}s", flush=True)
        if not config.smoke_test and validation_metrics[4] > best_f1:
            best_f1, stale = validation_metrics[4], 0
            torch.save({"model_state_dict": network.state_dict(), "class_names": CLASS_NAMES, "best_validation_macro_f1": best_f1, "class_weight_strategy": "none" if config.no_class_weights else "inverse_frequency"}, MODELS_ROOT / config.checkpoint_name)
        else:
            stale += 1
            if stale >= config.patience:
                print("Early stopping.")
                break
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    with (REPORTS_ROOT / config.history_name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy", "validation_macro_precision", "validation_macro_recall", "validation_macro_f1", "learning_rate"])
        writer.writerows(history)
    epochs = [row[0] for row in history]
    for filename, values, title, ylabel in (
        (f"{config.plot_prefix}training_loss.png", ([row[1] for row in history], [row[3] for row in history]), "Training and validation loss", "Loss"),
        (f"{config.plot_prefix}training_accuracy.png", ([row[2] for row in history], [row[4] for row in history]), "Training and validation accuracy", "Accuracy"),
        (f"{config.plot_prefix}validation_f1.png", ([row[7] for row in history],), "Validation macro F1", "Macro F1"),
    ):
        figure, axis = plt.subplots(figsize=(8, 5))
        for index, series in enumerate(values):
            axis.plot(epochs, series, label=("train" if filename != "validation_f1.png" and index == 0 else "validation"))
        axis.set(title=title, xlabel="Epoch", ylabel=ylabel); axis.legend(); axis.grid(alpha=0.25)
        figure.tight_layout(); figure.savefig(REPORTS_ROOT / filename, dpi=150); plt.close(figure)
    config_data = {"seed": config.seed, "python_version": platform.python_version(), "pytorch_version": torch.__version__, "torchvision_version": __import__("torchvision").__version__, "device": str(device), "model": "EfficientNet-B0", "optimizer": "AdamW", "learning_rate": config.learning_rate, "fine_tune_learning_rate": config.fine_tune_learning_rate, "batch_size": batch_size, "epochs_requested": epochs, "warmup_epochs": config.warmup_epochs, "image_size": config.image_size, "num_workers": config.num_workers, "class_weight_strategy": "none" if config.no_class_weights else "inverse_frequency", "class_weights": None if config.no_class_weights else weights.detach().cpu().tolist(), "checkpoint_name": config.checkpoint_name, "history_name": config.history_name, "plot_prefix": config.plot_prefix, "test_loaded_during_training": False, "leakage_checks": checks, "smoke_test": config.smoke_test}
    (MODELS_ROOT / ("training_config_no_weights.json" if config.no_class_weights else "training_config.json")).write_text(json.dumps(config_data, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())