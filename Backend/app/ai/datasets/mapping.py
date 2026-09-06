"""Load and apply transparent class mapping configuration."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from app.ai.datasets.paths import CLASS_MAPPING_PATH


def load_class_mapping(path: Optional[Path] = None) -> Dict[str, Any]:
    mapping_path = path or CLASS_MAPPING_PATH
    if not mapping_path.exists():
        raise FileNotFoundError(f"Class mapping config not found: {mapping_path}")
    with open(mapping_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def extract_material_from_caption(caption: str, mapping: Dict[str, Any]) -> Optional[str]:
    """Return target material class from caption keywords, or Mixed Fabrics if multiple."""
    if not caption:
        return None

    keywords = mapping.get("deepfashion_caption_keywords", {})
    caption_lower = caption.lower()
    matched: Set[str] = set()

    for keyword, target in keywords.items():
        if target and re.search(rf"\b{re.escape(keyword.lower())}\b", caption_lower):
            matched.add(target)

    if len(matched) == 0:
        return None
    if len(matched) == 1:
        return next(iter(matched))
    return "Mixed Fabrics"


def map_sustainable_material(value: str, mapping: Dict[str, Any]) -> Optional[str]:
    csv_map = mapping.get("sustainable_csv_materials", {})
    target = csv_map.get(value)
    if target is None and value in csv_map:
        return None
    return target
