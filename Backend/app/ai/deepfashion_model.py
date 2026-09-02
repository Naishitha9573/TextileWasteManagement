from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from torchvision import models, transforms

DEFAULT_INPUT_SIZE = 224


class DeepFashionGarmentClassifier(torch.nn.Module):
    def __init__(self, num_classes: int, input_size: int = DEFAULT_INPUT_SIZE):
        super().__init__()
        self.num_classes = int(num_classes)
        self.input_size = int(input_size)
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(in_features, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_inference_transform(input_size: int = DEFAULT_INPUT_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(input_size * 1.1)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def build_training_transform(input_size: int = DEFAULT_INPUT_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def preprocess_image(image: Image.Image | Path | str, input_size: int = DEFAULT_INPUT_SIZE, *, training: bool = False) -> torch.Tensor:
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    image = ImageOps.exif_transpose(image).convert("RGB")
    transform = build_training_transform(input_size) if training else build_inference_transform(input_size)
    return transform(image)


def load_model_from_checkpoint(checkpoint_path: str | Path, num_classes: int | None = None, input_size: int = DEFAULT_INPUT_SIZE) -> DeepFashionGarmentClassifier:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        num_classes = checkpoint.get("num_classes") or num_classes
        input_size = checkpoint.get("input_size") or input_size
        state_dict = checkpoint.get("model_state_dict", checkpoint)
    else:
        state_dict = checkpoint
    if num_classes is None:
        raise ValueError("num_classes required when loading checkpoint")
    model = DeepFashionGarmentClassifier(num_classes=num_classes, input_size=input_size)
    model.load_state_dict(state_dict)
    model.eval()
    return model
