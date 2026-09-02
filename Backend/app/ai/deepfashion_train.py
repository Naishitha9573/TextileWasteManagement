from __future__ import annotations

import csv
import gc
import json
import random
import re
import time
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from app.ai.deepfashion_model import DeepFashionGarmentClassifier, build_training_transform, build_inference_transform

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "deepfashion"
IMAGE_ROOT = DATASET_ROOT / "images"
MODEL_DIR = PROJECT_ROOT / "models" / "deepfashion"
SPLIT_ROOT = PROJECT_ROOT / "data" / "splits" / "deepfashion_category"

CATEGORY_RE = re.compile(r"^(?:MEN|WOMEN)-(?P<category>.+?)-id_(?P<sample_id>\d{8}-\d{2})", re.IGNORECASE)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class DeepFashionImageDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform: transforms.Compose | None = None):
        """Dataset that loads images directly from source paths without copying."""
        self.samples = samples
        self.transform = transform or build_inference_transform(224)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(label)


def list_dataset_images(dataset_root: Path) -> list[Path]:
    image_root = dataset_root / "images"
    if not image_root.exists():
        return []
    files = []
    for item in image_root.rglob("*"):
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(item)
    return sorted(files)


def inspect_dataset() -> dict[str, Any]:
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"DeepFashion root not found: {DATASET_ROOT}")

    image_paths = list_dataset_images(DATASET_ROOT)
    category_counts: dict[str, int] = defaultdict(int)
    sample_sizes: Counter[str] = Counter()
    corrupt_paths: list[str] = []
    parsed_count = 0
    seen_samples: set[str] = set()
    sample_to_paths: dict[str, list[Path]] = defaultdict(list)
    non_parseable: list[Path] = []

    for path in image_paths:
        match = CATEGORY_RE.match(path.name)
        if not match:
            non_parseable.append(path)
            continue
        category = match.group("category")
        sample_id = match.group("sample_id")
        category_counts[category] += 1
        parsed_count += 1
        sample_sizes[sample_id] += 1
        sample_to_paths[sample_id].append(path)
        seen_samples.add(sample_id)

    duplicate_sample_groups = {sample_id: paths for sample_id, paths in sample_to_paths.items() if len(paths) > 1}
    summary = {
        "dataset_root": str(DATASET_ROOT),
        "image_count": len(image_paths),
        "parsed_category_images": parsed_count,
        "categories": dict(sorted(category_counts.items())),
        "num_categories": len(category_counts),
        "num_samples": len(sample_to_paths),
        "duplicate_sample_groups": len(duplicate_sample_groups),
        "duplicate_group_examples": {sample_id: [str(p.relative_to(DATASET_ROOT)) for p in paths[:3]] for sample_id, paths in sorted(duplicate_sample_groups.items())[:5]},
        "corrupt_images": corrupt_paths[:10],
        "non_parseable_files": [str(p.relative_to(DATASET_ROOT)) for p in non_parseable[:10]],
        "sample_size_distribution_top": {sample_id: count for sample_id, count in sample_sizes.most_common(10)},
    }
    return summary


def build_group_aware_split(test_size: float = 0.15, validation_size: float = 0.15, seed: int = 42) -> dict[str, Any]:
    """Build in-memory split manifest without copying images. Group by sample ID to keep all views together."""
    if not IMAGE_ROOT.exists():
        raise FileNotFoundError(f"Image root not found: {IMAGE_ROOT}")

    # Step 1: Group all images by (category, sample_id) pair
    sample_to_paths: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in list_dataset_images(DATASET_ROOT):
        match = CATEGORY_RE.match(path.name)
        if not match:
            continue
        category = match.group("category")
        sample_id = match.group("sample_id")
        sample_to_paths[(category, sample_id)].append(path)

    # Step 2: Get all (category, sample_id) pairs and shuffle them globally
    all_sample_pairs = sorted(sample_to_paths.keys())  # sorted for reproducibility
    rng = random.Random(seed)
    rng.shuffle(all_sample_pairs)
    
    class_names = sorted(set(cat for cat, _ in sample_to_paths.keys()))
    
    # Step 3: Split the sample pairs into train/val/test by boundaries
    n_pairs = len(all_sample_pairs)
    train_end = int(n_pairs * 0.70)
    val_end = int(n_pairs * 0.85)
    
    train_pairs = set(all_sample_pairs[:train_end])
    val_pairs = set(all_sample_pairs[train_end:val_end])
    test_pairs = set(all_sample_pairs[val_end:])
    
    # Step 4: Build split_images dict by assigning paths based on their (category, sample_id) pair
    split_images: dict[str, dict[str, list[Path]]] = {split: defaultdict(list) for split in ("train", "validation", "test")}
    
    for (category, sample_id), paths in sample_to_paths.items():
        if (category, sample_id) in train_pairs:
            split_images["train"][category].extend(paths)
        elif (category, sample_id) in val_pairs:
            split_images["validation"][category].extend(paths)
        elif (category, sample_id) in test_pairs:
            split_images["test"][category].extend(paths)
    
    # Step 5: Verify no sample pair appears in multiple splits (should be guaranteed by construction)
    leakage = False
    pair_to_split: dict[tuple[str, str], str] = {}
    for split_name in ("train", "validation", "test"):
        for class_name, paths in split_images[split_name].items():
            for path in paths:
                match = CATEGORY_RE.match(path.name)
                if match:
                    pair_key = (class_name, match.group("sample_id"))
                    if pair_key in pair_to_split:
                        if pair_to_split[pair_key] != split_name:
                            leakage = True
                    else:
                        pair_to_split[pair_key] = split_name
    
    counts = {split: sum(len(paths) for paths in per_class.values()) for split, per_class in split_images.items()}
    class_dist = {
        split: {class_name: len(paths) for class_name, paths in sorted(per_class.items())}
        for split, per_class in split_images.items()
    }

    return {
        "split_images": split_images,
        "class_names": class_names,
        "counts": counts,
        "class_distribution": class_dist,
        "leakage_detected": leakage,
        "valid_split": not leakage,
    }


def compute_class_weights(class_counts: Counter[int]) -> torch.Tensor:
    total = sum(class_counts.values())
    weights = []
    for idx in sorted(class_counts):
        proportion = class_counts[idx] / total
        weights.append(1.0 / proportion if proportion > 0 else 1.0)
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    weights_tensor = weights_tensor / weights_tensor.mean()
    return weights_tensor


def train_model() -> dict[str, Any]:
    print("[DEBUG] Starting train_model()", flush=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("[DEBUG] Inspecting dataset...", flush=True)
    summary = inspect_dataset()
    print("[DEBUG] Dataset inspection complete", flush=True)
    print("DEEPFASHION DATASET SUMMARY")
    print(json.dumps({
        "image_count": summary["image_count"],
        "num_categories": summary["num_categories"],
        "categories": summary["categories"],
        "duplicate_sample_groups": summary["duplicate_sample_groups"],
        "corrupt_images": len(summary["corrupt_images"]),
    }, indent=2, sort_keys=True))

    split_summary = build_group_aware_split()
    print("DEEPFASHION SPLIT SUMMARY")
    print(json.dumps({
        "train": split_summary["counts"]["train"],
        "validation": split_summary["counts"]["validation"],
        "test": split_summary["counts"]["test"],
        "classes": len(split_summary["class_names"]),
        "leakage_detected": split_summary["leakage_detected"],
        "distribution": split_summary["class_distribution"],
    }, indent=2, sort_keys=True))
    if split_summary["leakage_detected"]:
        raise RuntimeError("Group-aware split leaked the same sample across multiple splits.")

    class_names = split_summary["class_names"]
    split_images = split_summary["split_images"]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    # Build in-memory sample lists from split_images without copying files
    train_samples: list[tuple[Path, int]] = []
    val_samples: list[tuple[Path, int]] = []
    test_samples: list[tuple[Path, int]] = []
    
    for class_name, paths in split_images["train"].items():
        for path in paths:
            train_samples.append((path, class_to_idx[class_name]))
    for class_name, paths in split_images["validation"].items():
        for path in paths:
            val_samples.append((path, class_to_idx[class_name]))
    for class_name, paths in split_images["test"].items():
        for path in paths:
            test_samples.append((path, class_to_idx[class_name]))

    print("[DEBUG] Creating datasets...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEBUG] Using device: {device}", flush=True)
    input_size = 224
    train_dataset = DeepFashionImageDataset(train_samples, transform=build_training_transform(input_size))
    val_dataset = DeepFashionImageDataset(val_samples, transform=build_inference_transform(input_size))
    test_dataset = DeepFashionImageDataset(test_samples, transform=build_inference_transform(input_size))

    if len(train_dataset) == 0:
        raise RuntimeError("No train images found after split generation.")

    print("[DEBUG] Creating DataLoaders...", flush=True)
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    print("[DEBUG] DataLoaders created", flush=True)
    print(f"[DeepFashion][CPU-safe] Using batch_size={batch_size}, num_workers=0, lazy image loading to reduce RAM pressure.", flush=True)

    print("[DEBUG] Building model...", flush=True)
    model = DeepFashionGarmentClassifier(num_classes=len(class_names), input_size=input_size).to(device)

    # Compute class counts from split_images directly (don't load all images)
    print("[DEBUG] Computing class weights...", flush=True)
    class_counts = Counter()
    for class_name, paths in split_images["train"].items():
        class_counts[class_to_idx[class_name]] = len(paths)
    weights = compute_class_weights(class_counts).to(device)
    print("[DEBUG] Class weights computed", flush=True)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    latest_checkpoint_path = MODEL_DIR / "deepfashion_latest_checkpoint.pth"
    checkpoint_path = MODEL_DIR / "best_deepfashion_garment_model.pth"
    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_val_macro_f1 = -1.0
    best_state = None
    best_epoch = -1
    history: list[dict[str, float]] = []
    patience = 3
    stale_epochs = 0
    start_epoch = 1

    if latest_checkpoint_path.exists():
        print(f"[DeepFashion] Resuming from checkpoint: {latest_checkpoint_path}", flush=True)
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        if checkpoint.get("model_state_dict") is not None:
            model.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("history") is not None:
            history = checkpoint["history"]
        if checkpoint.get("best_state_dict") is not None:
            best_state = checkpoint["best_state_dict"]
        if checkpoint.get("best_epoch") is not None:
            best_epoch = int(checkpoint["best_epoch"])
        if checkpoint.get("best_val_accuracy") is not None:
            best_val_accuracy = float(checkpoint["best_val_accuracy"])
        if checkpoint.get("best_val_loss") is not None:
            best_val_loss = float(checkpoint["best_val_loss"])
        if checkpoint.get("best_val_macro_f1") is not None:
            best_val_macro_f1 = float(checkpoint["best_val_macro_f1"])
        if checkpoint.get("epoch") is not None:
            start_epoch = int(checkpoint["epoch"]) + 1

    print("[DEBUG] Starting training loop...", flush=True)
    for epoch in range(start_epoch, 11):
        epoch_start_time = time.perf_counter()
        print(f"[DEBUG] Epoch {epoch}/10 starting", flush=True)
        model.train()
        train_loss_total = 0.0
        train_correct = 0
        train_total = 0
        total_batches = len(train_loader)
        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            if batch_idx % 10 == 0 or batch_idx == total_batches:
                elapsed = time.perf_counter() - epoch_start_time
                batch_loss = loss.item()
                print(
                    f"[DeepFashion] Epoch {epoch}/{10} | Batch {batch_idx}/{total_batches} | Loss: {batch_loss:.4f} | elapsed: {elapsed:.0f}s",
                    flush=True,
                )

            del images, labels, logits, loss, preds
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        train_loss = train_loss_total / max(1, train_total)
        train_accuracy = train_correct / max(1, train_total)

        model.eval()
        val_loss_total = 0.0
        val_correct = 0
        val_total = 0
        val_all_preds: list[int] = []
        val_all_labels: list[int] = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = loss_fn(logits, labels)
                val_loss_total += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_all_preds.extend(preds.detach().cpu().numpy().tolist())
                val_all_labels.extend(labels.detach().cpu().numpy().tolist())
                del images, labels, logits, loss, preds
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        val_loss = val_loss_total / max(1, val_total)
        val_accuracy = val_correct / max(1, val_total)
        if len(val_all_labels) > 0:
            from sklearn.metrics import precision_recall_fscore_support
            _, _, val_f1, _ = precision_recall_fscore_support(val_all_labels, val_all_preds, average="macro", zero_division=0)
        else:
            val_f1 = 0.0

        print(
            f"[DeepFashion] Epoch {epoch}/{10}\n"
            f"Train Loss: {train_loss:.4f}\n"
            f"Validation Loss: {val_loss:.4f}\n"
            f"Validation Accuracy: {val_accuracy:.4f}\n"
            f"Validation Macro F1: {val_f1:.4f}",
            flush=True,
        )
        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}")
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_macro_f1": float(val_f1),
        })

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_val_macro_f1 = float(val_f1)
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        latest_checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_accuracy": best_val_accuracy,
            "best_val_macro_f1": best_val_macro_f1,
            "history": history,
            "best_state_dict": best_state,
            "class_names": class_names,
        }
        torch.save(latest_checkpoint, latest_checkpoint_path)
        if best_state is not None:
            torch.save({
                "model_state_dict": best_state,
                "class_names": class_names,
                "num_classes": len(class_names),
                "input_size": input_size,
                "best_validation_accuracy": best_val_accuracy,
                "best_validation_loss": best_val_loss,
                "best_epoch": best_epoch,
                "history": history,
            }, checkpoint_path)

        print(f"BEST EPOCH: {best_epoch}", flush=True)
        print(f"BEST VALIDATION ACCURACY: {best_val_accuracy:.4f}", flush=True)
        print(f"BEST VALIDATION LOSS: {best_val_loss:.4f}", flush=True)
        print(f"BEST VALIDATION MACRO F1: {best_val_macro_f1:.4f}", flush=True)
        print(f"CHECKPOINT: {latest_checkpoint_path}", flush=True)
        print(f"[DeepFashion] Latest checkpoint saved: {latest_checkpoint_path}", flush=True)

    if best_state is None:
        raise RuntimeError("Model training failed before producing a valid validation checkpoint.")

    if best_state is None:
        raise RuntimeError("Model training failed before producing a valid validation checkpoint.")

    checkpoint_path = MODEL_DIR / "best_deepfashion_garment_model.pth"
    torch.save({
        "model_state_dict": best_state,
        "class_names": class_names,
        "num_classes": len(class_names),
        "input_size": input_size,
        "best_validation_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
        "history": history,
    }, checkpoint_path)

    class_names_path = MODEL_DIR / "class_names.json"
    class_names_path.write_text(json.dumps({"class_names": class_names, "class_to_idx": {name: idx for idx, name in enumerate(class_names)}}, indent=2), encoding="utf-8")

    config_path = MODEL_DIR / "deepfashion_training_config.json"
    config_path.write_text(json.dumps({
        "dataset": "DeepFashion",
        "task": "Garment category classification",
        "input_size": input_size,
        "num_classes": len(class_names),
        "train_count": len(train_dataset),
        "validation_count": len(val_dataset),
        "test_count": len(test_dataset),
        "split_root": str(SPLIT_ROOT),
        "class_names": class_names,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_val_accuracy,
    }, indent=2), encoding="utf-8")

    model.load_state_dict(best_state)
    model.eval()
    test_loss_total = 0.0
    test_correct = 0
    test_total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            test_loss_total += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    test_loss = test_loss_total / max(1, test_total)
    test_accuracy = test_correct / max(1, test_total)
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    conf = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    metrics = {
        "best_epoch": best_epoch,
        "best_validation_accuracy": float(best_val_accuracy),
        "final_training_accuracy": max(h["train_accuracy"] for h in history) if history else 0.0,
        "final_validation_accuracy": history[-1]["val_accuracy"] if history else 0.0,
        "test_accuracy": float(test_accuracy),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1": float(f1),
        "test_loss": float(test_loss),
        "confusion_matrix": conf.tolist(),
    }
    (MODEL_DIR / "deepfashion_training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"BEST EPOCH: {best_epoch}")
    print(f"BEST VALIDATION ACCURACY: {best_val_accuracy:.4f}")
    print(f"BEST VALIDATION MACRO F1: {best_val_macro_f1:.4f}")
    print(f"CHECKPOINT: {checkpoint_path}")
    print("FINAL METRICS")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved class names: {class_names_path}")
    return {
        "checkpoint_path": str(checkpoint_path),
        "class_names_path": str(class_names_path),
        "split_summary": split_summary,
        "dataset_summary": summary,
        "metrics": metrics,
    }


if __name__ == "__main__":
    train_model()
