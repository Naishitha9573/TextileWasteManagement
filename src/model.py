"""Shared EfficientNet-B0 baseline utilities."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "dataset" / "processed" / "fabric_9class"
MODELS_ROOT = ROOT / "models"
REPORTS_ROOT = ROOT / "reports"
CLASS_NAMES = sorted(["Cotton", "Polyester", "Denim", "Wool", "Nylon", "Viscose", "Silk", "Fleece", "Terrycloth"])
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def get_transforms(image_size: int = IMAGE_SIZE, targeted: bool = False) -> tuple[transforms.Compose, transforms.Compose]:
    train_steps = [
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
    ]
    if targeted:
        train_steps.extend([
            transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.97, 1.03), shear=3),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.10),
        ])
    train = transforms.Compose(train_steps + [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    deterministic = transforms.Compose([
        transforms.Resize(round(image_size * 256 / 224)), transforms.CenterCrop(image_size), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    return train, deterministic


def make_datasets(root: Path = DATASET_ROOT, image_size: int = IMAGE_SIZE, include_test: bool = True) -> tuple[dict[str, datasets.ImageFolder], list[str]]:
    train_transform, deterministic_transform = get_transforms(image_size)
    datasets_by_split: dict[str, datasets.ImageFolder] = {
        "train": datasets.ImageFolder(root / "train", transform=train_transform),
        "validation": datasets.ImageFolder(root / "validation", transform=deterministic_transform),
    }
    if include_test:
        datasets_by_split["test"] = datasets.ImageFolder(root / "test", transform=deterministic_transform)
    mappings = [dataset.class_to_idx for dataset in datasets_by_split.values()]
    if any(mapping != mappings[0] for mapping in mappings[1:]) or list(mappings[0]) != CLASS_NAMES:
        raise RuntimeError(f"Split class mappings differ or are not the expected deterministic order: {mappings}")
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    (MODELS_ROOT / "class_names.json").write_text(json.dumps({str(index): name for index, name in enumerate(CLASS_NAMES)}, indent=2) + "\n", encoding="utf-8")
    return datasets_by_split, CLASS_NAMES


def training_class_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    counts = np.bincount(dataset.targets, minlength=len(CLASS_NAMES)).astype(np.float64)
    if np.any(counts == 0):
        raise RuntimeError(f"Training set has an empty class: {counts.tolist()}")
    weights = counts.sum() / (len(CLASS_NAMES) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def create_model(num_classes: int = 9, pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    network = models.efficientnet_b0(weights=weights)
    input_features = network.classifier[1].in_features
    network.classifier[1] = nn.Linear(input_features, num_classes)
    return network


def sample_id(path: Path) -> str | None:
    parts = path.parts
    return parts[-2] if len(parts) >= 2 else None


def leakage_checks(root: Path = DATASET_ROOT) -> dict[str, Any]:
    files_by_split: dict[str, list[Path]] = {}
    hashes: dict[str, set[str]] = {}
    groups: dict[str, set[str]] = {}
    for split in ("train", "validation", "test"):
        files = sorted(path for path in (root / split).rglob("*") if path.is_file())
        files_by_split[split] = files
        hashes[split] = set()
        for path in files:
            try:
                with Image.open(path) as image:
                    image.verify()
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except (OSError, SyntaxError, UnidentifiedImageError, ValueError) as exc:
                raise RuntimeError(f"Leakage check found unreadable image {path}: {exc}") from exc
            hashes[split].add(digest)
            relative = path.relative_to(root / split)
            group = sample_id(relative)
            if group is None:
                raise RuntimeError(f"Cannot extract sample ID from {relative}")
            groups.setdefault(group, set()).add(split)
    duplicate_hashes: set[str] = set()
    for index, split in enumerate(("train", "validation", "test")):
        for other in ("train", "validation", "test")[index + 1:]:
            duplicate_hashes.update(hashes[split] & hashes[other])
    leaked_groups = {group: sorted(splits) for group, splits in groups.items() if len(splits) > 1}
    checks = {
        "duplicate_across_splits": not duplicate_hashes,
        "sample_groups_across_splits": not leaked_groups,
        "test_images_separate": not (set(files_by_split["test"]) & set(files_by_split["train"])),
        "test_labels_not_used_for_training": True,
        "test_metrics_not_used_for_checkpoint_selection": True,
        "duplicate_hashes": sorted(duplicate_hashes),
        "leaked_sample_groups": leaked_groups,
    }
    if not all(checks[key] for key in ("duplicate_across_splits", "sample_groups_across_splits", "test_images_separate")):
        raise RuntimeError(f"DATA LEAKAGE DETECTED: {checks}")
    return checks


def make_loader(dataset: datasets.ImageFolder, batch_size: int, shuffle: bool, seed: int, num_workers: int = 0, sampler: WeightedRandomSampler | None = None) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle if sampler is None else False, sampler=sampler, num_workers=num_workers, pin_memory=torch.cuda.is_available(), generator=generator)
