"""Download deployment-only model artifacts that are intentionally not stored in Git."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen


MODEL_PATH = Path(os.getenv("FABRIC_MODEL_PATH", "/models/best_efficientnet_b0_class_aware_finetune_v1.pth"))
MODEL_URL = os.getenv("FABRIC_MODEL_URL", "").strip()


if not MODEL_PATH.is_file():
    if not MODEL_URL:
        raise SystemExit("FABRIC_MODEL_URL is required when the fabric checkpoint is not baked into the image")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=MODEL_PATH.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with urlopen(MODEL_URL, timeout=180) as response, temporary_path.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary_path.replace(MODEL_PATH)
    finally:
        temporary_path.unlink(missing_ok=True)

if not MODEL_PATH.is_file():
    raise SystemExit(f"Fabric checkpoint was not created at {MODEL_PATH}")

print(f"Fabric checkpoint ready: {MODEL_PATH}")
