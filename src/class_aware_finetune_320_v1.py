"""Validation-only 320x320 class-aware EfficientNet-B0 fine-tuning experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

try:
    from model import CLASS_NAMES, DATASET_ROOT, MEAN, ROOT, STD, create_model, get_transforms, seed_everything
except ImportError:
    from .model import CLASS_NAMES, DATASET_ROOT, MEAN, ROOT, STD, create_model, get_transforms, seed_everything

EXPERIMENT_NAME = "class_aware_finetune_320_v1"
BASELINE_VALIDATION_ACCURACY = 0.8554
BASELINE_VALIDATION_MACRO_F1 = 0.8277
BASELINE_VALIDATION_WEIGHTED_F1 = 0.8560
MODELS_ROOT = ROOT / "models" / "experiments" / EXPERIMENT_NAME
REPORTS_ROOT = ROOT / "reports" / "experiments" / EXPERIMENT_NAME
CHECKPOINT_PATH = MODELS_ROOT / "best_checkpoint.pth"
V2_CHECKPOINT_PATH = ROOT / "models" / "best_efficientnet_b0_targeted_augmentation_v2.pth"
CHAMPION_CHECKPOINT_PATH = ROOT / "models" / "best_efficientnet_b0_class_aware_finetune_v1.pth"
HISTORY_PATH = REPORTS_ROOT / "history.csv"
FINAL_PATH = REPORTS_ROOT / "final.json"
CONFUSION_PATH = REPORTS_ROOT / "validation_confusion_matrix.png"
ERROR_PATH = REPORTS_ROOT / "error.txt"
HISTORY_FIELDS = ("epoch", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy", "validation_macro_f1", "validation_weighted_f1", "learning_rate", "elapsed_seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated 320x320 class-aware validation-only EfficientNet-B0 fine-tuning.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from the existing best model weights without overwriting them")
    parser.add_argument("--resume-checkpoint", type=Path, default=CHECKPOINT_PATH)
    return parser.parse_args()


def experiment_files() -> tuple[Path, ...]:
    return (CHECKPOINT_PATH, HISTORY_PATH, FINAL_PATH, CONFUSION_PATH, ERROR_PATH)


def file_snapshot() -> dict[str, int]:
    return {str(path): sum(1 for item in path.rglob("*") if item.is_file()) for path in (DATASET_ROOT / "train", DATASET_ROOT / "validation")}


def sample_groups(root: Path) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = {}
    for split in ("train", "validation"):
        for path in sorted((root / split).rglob("*")):
            if path.is_file():
                relative = path.relative_to(root / split)
                parts = relative.parts
                if len(parts) < 2:
                    raise RuntimeError(f"Cannot extract sample ID from {relative}")
                groups.setdefault(parts[-2], set()).add(split)
    return groups


def image_hashes(root: Path) -> dict[str, set[str]]:
    hashes: dict[str, set[str]] = {}
    for split in ("train", "validation"):
        split_hashes: set[str] = set()
        for path in sorted((root / split).rglob("*")):
            if path.is_file():
                try:
                    with Image.open(path) as image:
                        image.verify()
                    split_hashes.add(hashlib.sha256(path.read_bytes()).hexdigest())
                except (OSError, SyntaxError, UnidentifiedImageError, ValueError) as exc:
                    raise RuntimeError(f"Unreadable image {path}: {exc}") from exc
        hashes[split] = split_hashes
    return hashes


def safety_checks() -> dict[str, object]:
    if not DATASET_ROOT.is_dir() or not (DATASET_ROOT / "train").is_dir() or not (DATASET_ROOT / "validation").is_dir() or not (DATASET_ROOT / "test").is_dir():
        raise RuntimeError("Dataset, train, validation, and test directories must exist")
    if not (ROOT / "models" / "best_efficientnet_b0_no_weights.pth").is_file():
        raise RuntimeError("Baseline checkpoint is missing")
    if not V2_CHECKPOINT_PATH.is_file():
        raise RuntimeError("V2 checkpoint is missing")
    if not CHAMPION_CHECKPOINT_PATH.is_file():
        raise RuntimeError("Champion checkpoint is missing")
    hashes = image_hashes(DATASET_ROOT)
    groups = sample_groups(DATASET_ROOT)
    if hashes["train"] & hashes["validation"]:
        raise RuntimeError("Duplicate image content detected across train/validation")
    leaked_groups = {group: sorted(splits) for group, splits in groups.items() if len(splits) > 1}
    if leaked_groups:
        raise RuntimeError(f"Sample-ID leakage detected: {leaked_groups}")
    return {"dataset_modified": False, "splits_modified": False, "baseline_modified": False, "v2_checkpoint_modified": False, "champion_checkpoint_modified": False, "test_used": False, "class_mapping_verified": "PASS", "train_validation_duplicate_hashes": 0, "leaked_sample_groups": 0, "file_snapshot": file_snapshot()}


def build_train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def save_history(row: dict[str, object]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_empty = not HISTORY_PATH.exists() or HISTORY_PATH.stat().st_size == 0
    with HISTORY_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        if is_empty:
            writer.writeheader()
        writer.writerow(row)


def validation_pass(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> tuple[float, float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    actual: list[int] = []
    predicted: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            losses.append(loss_fn(logits, labels.to(device)).item() * labels.size(0))
            actual.extend(labels.numpy())
            predicted.extend(logits.argmax(1).cpu().numpy())
    matrix = confusion_matrix(actual, predicted, labels=list(range(len(CLASS_NAMES))))
    precision, recall, f1_values, support = precision_recall_fscore_support(actual, predicted, labels=list(range(len(CLASS_NAMES))), zero_division=0)
    return sum(losses) / len(loader.dataset), accuracy_score(actual, predicted), float(f1_values.mean()), float(np.average(f1_values, weights=support)), matrix, precision, recall, f1_values


def write_confusion_matrix(matrix: np.ndarray) -> None:
    figure, axis = plt.subplots(figsize=(9, 7))
    axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=range(len(CLASS_NAMES)), yticks=range(len(CLASS_NAMES)), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, xlabel="Predicted", ylabel="Actual", title=f"{EXPERIMENT_NAME} validation confusion matrix")
    axis.tick_params(axis="x", labelrotation=65)
    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
            axis.text(column, row, matrix[row, column], ha="center", va="center")
    figure.tight_layout()
    figure.savefig(CONFUSION_PATH, dpi=150)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.warmup_epochs >= args.epochs or args.patience < 1 or not 0 <= args.label_smoothing < 1:
        raise SystemExit("Invalid epochs, warmup, patience, or label-smoothing configuration")
    if args.image_size != 320:
        raise SystemExit("This dedicated experiment requires --image-size 320")
    resume_checkpoint = args.resume_checkpoint.expanduser().resolve()
    if args.resume:
        if not resume_checkpoint.is_file():
            raise SystemExit(f"Resume checkpoint does not exist: {resume_checkpoint}")
        resume_root = ROOT / "models" / "experiments" / EXPERIMENT_NAME / "resumed"
        resume_report_root = ROOT / "reports" / "experiments" / EXPERIMENT_NAME / "resumed"
        if any(path.exists() for path in (resume_root / "best_checkpoint.pth", resume_report_root / "history.csv")):
            continuation_index = 1
            while True:
                candidate_root = ROOT / "models" / "experiments" / EXPERIMENT_NAME / f"resumed_continuation_{continuation_index}"
                candidate_report_root = ROOT / "reports" / "experiments" / EXPERIMENT_NAME / f"resumed_continuation_{continuation_index}"
                if not any(path.exists() for path in (candidate_root / "best_checkpoint.pth", candidate_report_root / "history.csv", candidate_report_root / "final.json", candidate_report_root / "validation_confusion_matrix.png", candidate_report_root / "error.txt")):
                    resume_root = candidate_root
                    resume_report_root = candidate_report_root
                    break
                continuation_index += 1
        global MODELS_ROOT, REPORTS_ROOT, CHECKPOINT_PATH, HISTORY_PATH, FINAL_PATH, CONFUSION_PATH, ERROR_PATH
        MODELS_ROOT = resume_root
        REPORTS_ROOT = resume_report_root
        CHECKPOINT_PATH = MODELS_ROOT / "best_checkpoint.pth"
        HISTORY_PATH = REPORTS_ROOT / "history.csv"
        FINAL_PATH = REPORTS_ROOT / "final.json"
        CONFUSION_PATH = REPORTS_ROOT / "validation_confusion_matrix.png"
        ERROR_PATH = REPORTS_ROOT / "error.txt"
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    if any(path.exists() for path in experiment_files()):
        raise SystemExit("Experiment output already exists; refusing to overwrite it")
    checks = safety_checks()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform = build_train_transform(args.image_size)
    _, validation_transform = get_transforms(args.image_size, targeted=False)
    train_dataset = datasets.ImageFolder(DATASET_ROOT / "train", transform=train_transform)
    validation_dataset = datasets.ImageFolder(DATASET_ROOT / "validation", transform=validation_transform)
    expected_mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    if train_dataset.class_to_idx != expected_mapping or validation_dataset.class_to_idx != expected_mapping or len(CLASS_NAMES) != 9:
        raise RuntimeError(f"Class mapping mismatch: train={train_dataset.class_to_idx}, validation={validation_dataset.class_to_idx}, expected={expected_mapping}")
    train_counts = np.bincount(train_dataset.targets, minlength=len(CLASS_NAMES)).astype(np.float64)
    inverse_frequency = train_counts.sum() / (len(CLASS_NAMES) * train_counts)
    class_weights = np.sqrt(inverse_frequency)
    class_weights /= class_weights.mean()
    print("SAFETY CHECK\n============\nDataset modified: NO\nSplits modified: NO\nBaseline modified: NO\nChampion checkpoint modified: NO\nV2 checkpoint modified: NO\nTest set used: NO\nClass mapping verified: PASS")
    print(f"\nSEED: {args.seed}\nCUDA available: {torch.cuda.is_available()}\nDevice: {device}\nPyTorch version: {torch.__version__}")
    print(f"\nCLASS MAPPING")
    for index, name in enumerate(CLASS_NAMES):
        print(f"{index} = {name}")
    print(f"\nTRAIN CLASS COUNTS: {dict(zip(CLASS_NAMES, train_counts.astype(int)))}")
    print(f"CLASS WEIGHTS (sqrt inverse frequency, mean-normalized): {dict(zip(CLASS_NAMES, class_weights.tolist()))}")
    print("AUGMENTATION: RandomResizedCrop(scale=(0.85, 1.0)), HorizontalFlip(p=0.5), Rotation(8), ColorJitter(0.10, 0.10, 0.05, 0.02)")
    sampler = None
    if args.balanced_sampler:
        sample_weights = torch.tensor([inverse_frequency[target] for target in train_dataset.targets], dtype=torch.double)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True, generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), generator=torch.Generator().manual_seed(args.seed))
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    model = create_model(len(CLASS_NAMES), pretrained=True).to(device)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device), label_smoothing=args.label_smoothing)
    optimizer = AdamW(filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup_epochs), eta_min=args.fine_tune_learning_rate)
    best = {"epoch": 0, "accuracy": 0.0, "macro_f1": -1.0, "weighted_f1": 0.0, "matrix": None}
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        if checkpoint.get("class_names") != CLASS_NAMES:
            raise RuntimeError("Resume checkpoint class mapping differs from authoritative CLASS_NAMES")
        model.load_state_dict(checkpoint["model_state_dict"])
        history_last_epoch = 0
        source_history = ROOT / "reports" / "experiments" / EXPERIMENT_NAME / "resumed" / "history.csv"
        if source_history.is_file():
            with source_history.open(newline="", encoding="utf-8") as handle:
                history_last_epoch = max((int(row["epoch"]) for row in csv.DictReader(handle)), default=0)
        start_epoch = max(int(checkpoint.get("best_epoch", 0)), history_last_epoch) + 1
        for parameter in model.features[-3:].parameters():
            parameter.requires_grad = True
        optimizer = AdamW([{"params": model.classifier.parameters(), "lr": args.learning_rate}, {"params": model.features[-3:].parameters(), "lr": args.fine_tune_learning_rate}], weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup_epochs), eta_min=args.fine_tune_learning_rate / 10)
        best = {"epoch": int(checkpoint.get("best_epoch", 0)), "accuracy": float(checkpoint.get("best_validation_accuracy", 0.0)), "macro_f1": float(checkpoint.get("best_validation_macro_f1", -1.0)), "weighted_f1": float(checkpoint.get("best_validation_weighted_f1", 0.0)), "matrix": None}
        print(f"\nExperiment: {EXPERIMENT_NAME}\nResuming from epoch: {start_epoch}\nStarting from checkpoint: {resume_checkpoint}\nCheckpoint contains model weights only; optimizer and scheduler state will be reconstructed.")
    stale = 0
    started = time.perf_counter()
    last_completed_epoch = 0
    last_validation_accuracy = 0.0
    last_validation_macro_f1 = 0.0
    last_matrix = None
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            if epoch == args.warmup_epochs + 1:
                for parameter in model.features[-3:].parameters():
                    parameter.requires_grad = True
                optimizer = AdamW([{"params": model.classifier.parameters(), "lr": args.learning_rate}, {"params": model.features[-3:].parameters(), "lr": args.fine_tune_learning_rate}], weight_decay=1e-4)
                scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup_epochs), eta_min=args.fine_tune_learning_rate / 10)
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
                train_predicted.extend(logits.argmax(1).detach().cpu().numpy())
            validation_loss, validation_accuracy, validation_macro_f1, validation_weighted_f1, matrix, precision, recall, f1_values = validation_pass(model, validation_loader, loss_fn, device)
            last_matrix = matrix
            scheduler.step()
            last_completed_epoch = epoch
            last_validation_accuracy = validation_accuracy
            last_validation_macro_f1 = validation_macro_f1
            elapsed = round(time.perf_counter() - started, 2)
            row = {"epoch": epoch, "train_loss": train_loss / len(train_dataset), "train_accuracy": accuracy_score(train_actual, train_predicted), "validation_loss": validation_loss, "validation_accuracy": validation_accuracy, "validation_macro_f1": validation_macro_f1, "validation_weighted_f1": validation_weighted_f1, "learning_rate": optimizer.param_groups[0]["lr"], "elapsed_seconds": elapsed}
            save_history(row)
            print(f"Epoch {epoch}/{args.epochs} | Train Loss {row['train_loss']:.4f} | Train Accuracy {row['train_accuracy']:.4f} | Validation Loss {validation_loss:.4f} | Validation Accuracy {validation_accuracy:.4f} | Validation Macro F1 {validation_macro_f1:.4f} | Validation Weighted F1 {validation_weighted_f1:.4f} | Learning Rate {row['learning_rate']:.7f} | Elapsed {elapsed}s")
            print("Per-class F1:", dict(zip(CLASS_NAMES, [round(float(value), 4) for value in f1_values])))
            if validation_macro_f1 > best["macro_f1"]:
                best = {"epoch": epoch, "accuracy": validation_accuracy, "macro_f1": validation_macro_f1, "weighted_f1": validation_weighted_f1, "matrix": matrix}
                torch.save({"model_state_dict": model.state_dict(), "class_names": CLASS_NAMES, "experiment_name": EXPERIMENT_NAME, "best_validation_macro_f1": validation_macro_f1, "best_validation_accuracy": validation_accuracy, "best_validation_weighted_f1": validation_weighted_f1, "best_epoch": epoch}, CHECKPOINT_PATH)
                stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"Early stopping at epoch {epoch}; patience reached.")
                    break
        if best["matrix"] is None:
            best["matrix"] = last_matrix
        if best["matrix"] is None:
            raise RuntimeError("No best validation checkpoint was produced")
        write_confusion_matrix(np.asarray(best["matrix"]))
        final_payload = {"experiment": EXPERIMENT_NAME, "best_epoch": best["epoch"], "best_validation_accuracy": best["accuracy"], "best_validation_macro_f1": best["macro_f1"], "best_validation_weighted_f1": best["weighted_f1"], "checkpoint": str(CHECKPOINT_PATH), "resumed_from": str(resume_checkpoint) if args.resume else None, "test_used": False, "training_dataset_modified": False, "splits_modified": False, "baseline_checkpoint_modified": False, "class_names": CLASS_NAMES, "training_configuration": vars(args), "class_weights": dict(zip(CLASS_NAMES, class_weights.tolist())), "safety_checks": checks}
        FINAL_PATH.write_text(json.dumps(final_payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nEXPERIMENT: {EXPERIMENT_NAME}\nBASELINE:\nValidation Accuracy: {BASELINE_VALIDATION_ACCURACY:.2%}\nValidation Macro F1: {BASELINE_VALIDATION_MACRO_F1:.2%}\nNEW MODEL:\nBest Epoch: {best['epoch']}\nValidation Accuracy: {best['accuracy']:.2%}\nValidation Macro F1: {best['macro_f1']:.2%}\nValidation Weighted F1: {best['weighted_f1']:.2%}\nIMPROVEMENT:\nAccuracy: {best['accuracy'] - BASELINE_VALIDATION_ACCURACY:+.6f}\nMacro F1: {best['macro_f1'] - BASELINE_VALIDATION_MACRO_F1:+.6f}\nWeighted F1: {best['weighted_f1'] - BASELINE_VALIDATION_WEIGHTED_F1:+.6f}\nCHECKPOINT:\n{CHECKPOINT_PATH}\nTEST SET:\nNOT USED\nDATASET:\nNOT MODIFIED\nSPLITS:\nNOT MODIFIED\nBASELINE CHECKPOINT:\nUNCHANGED\nRECOMMENDATION:\n{'Candidate model is better on validation Macro F1. Keep it as a candidate and wait for explicit approval before test evaluation.' if best['macro_f1'] > BASELINE_VALIDATION_MACRO_F1 else 'Experiment did not beat the baseline validation Macro F1. Keep the baseline model.'}")
        return 0
    except Exception:
        ERROR_PATH.write_text(traceback.format_exc() + f"\nLast completed epoch: {last_completed_epoch}\nLast validation Macro F1: {last_validation_macro_f1}\nLast validation accuracy: {last_validation_accuracy}\n", encoding="utf-8")
        print(traceback.format_exc(), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())