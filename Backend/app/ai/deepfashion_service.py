"""Measured DeepFashion dataset capability and optional garment inference."""
from __future__ import annotations

import re
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

DATASET_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "deepfashion"
MODEL_ROOT = Path(__file__).resolve().parents[2] / "models"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CATEGORY_PATTERN = re.compile(r"^(?:MEN|WOMEN)-(?P<category>[^-]+)-id_", re.IGNORECASE)


def _files_with_extensions(path: Path, extensions: set[str]) -> list[Path]:
    if not path.exists():
        return []
    return [item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in extensions]


def _annotation_files() -> list[str]:
    if not DATASET_ROOT.exists():
        return []
    files = [
        item.relative_to(DATASET_ROOT).as_posix()
        for item in DATASET_ROOT.rglob("*")
        if item.is_file() and item.suffix.lower() in {".json", ".txt"}
    ]
    for archive in DATASET_ROOT.parent.glob("*.zip"):
        try:
            with zipfile.ZipFile(archive) as bundle:
                files.extend(
                    f"{archive.name}:{entry.filename}"
                    for entry in bundle.infolist()
                    if not entry.is_dir() and entry.filename.lower().endswith(".txt")
                )
        except (OSError, zipfile.BadZipFile):
            continue
    return sorted(set(files))


def _model_artifacts() -> list[str]:
    if not MODEL_ROOT.exists():
        return []
    terms = ("deepfashion", "garment", "fashion")
    return sorted(
        item.relative_to(MODEL_ROOT).as_posix()
        for item in MODEL_ROOT.rglob("*")
        if item.is_file() and any(term in item.name.lower() for term in terms)
    )


@lru_cache(maxsize=1)
def get_deepfashion_status() -> Dict[str, Any]:
    """Return measured dataset facts without treating labels as model predictions."""
    image_files = _files_with_extensions(DATASET_ROOT / "images", IMAGE_EXTENSIONS)
    category_labels = sorted(
        {
            match.group("category")
            for image in image_files
            if (match := CATEGORY_PATTERN.match(image.name))
        }
    )
    annotations = _annotation_files()
    annotation_text = "\n".join(annotations).lower()
    model_artifacts = _model_artifacts()
    return {
        "dataset_available": bool(image_files),
        "dataset_path": "Backend/datasets/deepfashion",
        "subsets": [
            "Category and Attribute Prediction Benchmark (coarse/fine)",
            "Consumer-to-shop Clothes Retrieval Benchmark",
            "Extracted DeepFashion annotations and captions",
        ],
        "image_count": len(image_files),
        "annotation_files": annotations,
        "category_labels": category_labels,
        "attributes_available": "attr" in annotation_text or (DATASET_ROOT / "captions.json").exists(),
        "bounding_boxes_available": "bbox" in annotation_text,
        "landmarks_available": "landmark" in annotation_text or "keypoint" in annotation_text,
        "segmentation_available": bool(_files_with_extensions(DATASET_ROOT / "segm", {".png", ".jpg", ".jpeg"})),
        "retrieval_annotations_available": "retrieval" in annotation_text or "consumer2shop" in annotation_text,
        "model_available": bool(model_artifacts),
        "model_artifacts": model_artifacts,
        "inference_available": False,
        "model_status": "NOT AVAILABLE" if not model_artifacts else "UNSUPPORTED",
    }


def garment_analysis() -> Dict[str, Any]:
    status = get_deepfashion_status()
    if not status["inference_available"]:
        return {
            "status": "MODEL_NOT_READY",
            "category": None,
            "attributes": [],
            "confidence": None,
            "dataset": status,
            "message": "DeepFashion dataset available; compatible inference model not available.",
        }
    raise NotImplementedError("A compatible DeepFashion inference adapter is not configured.")
