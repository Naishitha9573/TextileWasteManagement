"""Predict one image with the best validation-selected checkpoint."""

from __future__ import annotations

import argparse
import json

import torch
from PIL import Image

from model import CLASS_NAMES, MODELS_ROOT, create_model, get_transforms


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict a fabric class for one image.")
    parser.add_argument("image_path")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODELS_ROOT / "best_efficientnet_b0.pth", map_location=device, weights_only=False)
    network = create_model(len(CLASS_NAMES), pretrained=False).to(device)
    network.load_state_dict(checkpoint["model_state_dict"]); network.eval()
    _, deterministic = get_transforms()
    with Image.open(args.image_path) as image:
        tensor = deterministic(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(network(tensor), 1)[0]
    values, indices = probabilities.topk(3)
    result = {"image": args.image_path, "predicted_class": CLASS_NAMES[indices[0].item()], "confidence": float(values[0]), "top_3": [{"class_name": CLASS_NAMES[index.item()], "probability": float(value)} for value, index in zip(values, indices)]}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())