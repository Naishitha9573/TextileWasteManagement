"""Shared helpers for the ml.datasets CLI pipeline.

All scripts operate on the real project data under Backend/data and
Backend/datasets. Nothing here deletes or modifies source images.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image

BACKEND_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = BACKEND_ROOT / "data"
PROCESSED_DIR = DATA_ROOT / "processed"
SPLITS_DIR = DATA_ROOT / "splits"
METADATA_DIR = DATA_ROOT / "metadata"
REPORTS_DIR = DATA_ROOT / "reports"
DATASETS_SRC = BACKEND_ROOT / "datasets"
PROJECT_ROOT = BACKEND_ROOT.parent

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HASH_CACHE_PATH = METADATA_DIR / "ml_dataset_hash_cache.json"

# Training datasets that feed supervised material models (final accepted sets).
TRAINING_DATASETS = {
    "ibug_material_v2": {
        "manifest": "ibug_material_v2_audit.json",
        "label_source": "manufacturer tag.txt composition metadata (iBUG)",
    },
    "material_deepfashion_verified_clean": {
        "manifest": "material_dataset_final_manifest.json",
        "label_source": "DeepFashion caption keyword mapping",
    },
}


def iter_images(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash64(path: Path, hash_size: int = 8) -> str:
    """64-bit difference hash (dHash) returned as 16 hex chars."""
    with Image.open(path) as img:
        gray = img.convert("L").resize((hash_size + 1, hash_size))
        pixels = list(gray.getdata())
    bits = []
    for row in range(hash_size):
        row_start = row * (hash_size + 1)
        for col in range(hash_size):
            bits.append("1" if pixels[row_start + col] > pixels[row_start + col + 1] else "0")
    return "".join("1" if b == "1" else "0" for b in bits)


def hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 2) ^ int(b, 2)).count("1")


def load_hash_cache() -> Dict[str, Dict[str, object]]:
    if HASH_CACHE_PATH.exists():
        try:
            return json.loads(HASH_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_hash_cache(cache: Dict[str, Dict[str, object]]) -> None:
    HASH_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    HASH_CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")


def compute_hashes(
    paths: List[Path], use_cache: bool = True
) -> Dict[str, Dict[str, object]]:
    """Return {abs_path: {"sha256":..., "dhash":...}} with an on-disk cache."""
    cache = load_hash_cache() if use_cache else {}
    result: Dict[str, Dict[str, object]] = {}
    pending: List[Path] = []
    for path in paths:
        key = str(path)
        stat = path.stat()
        stamp = f"{stat.st_size}:{int(stat.st_mtime)}"
        entry = cache.get(key)
        if use_cache and entry and entry.get("stamp") == stamp:
            result[key] = entry
        else:
            pending.append(path)
    total = len(paths)
    done = total - len(pending)
    for index, path in enumerate(pending):
        key = str(path)
        stat = path.stat()
        stamp = f"{stat.st_size}:{int(stat.st_mtime)}"
        try:
            entry = {"sha256": sha256_of(path), "dhash": dhash64(path), "stamp": stamp}
            result[key] = entry
            cache[key] = entry
        except Exception as exc:
            result[key] = {"error": str(exc), "stamp": stamp}
        if (index + 1) % 250 == 0:
            print(f"  hashed {done + index + 1}/{total}")
    if pending:
        save_hash_cache(cache)
    return result


def group_key_for_ibug(source_path: str) -> str:
    """'folder/sample' for iBUG source paths like .../ibug_fabrics/<folder>/<sample>/im_*.png"""
    parts = Path(source_path).parts
    return f"{parts[-3]}/{parts[-2]}"


def count_by_class(split_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for class_dir in sorted(d for d in split_dir.iterdir() if d.is_dir()):
        counts[class_dir.name] = sum(1 for _ in iter_images(class_dir))
    return counts


def json_load(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
