"""Dataset validation helpers: image integrity, duplicates, dimensions."""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def collect_image_files(root: Path, recursive: bool = True) -> List[Path]:
    if not root.exists():
        return []
    if recursive:
        return sorted(
            p for p in root.rglob("*")
            if p.is_file() and is_image_file(p)
        )
    return sorted(
        p for p in root.iterdir()
        if p.is_file() and is_image_file(p)
    )


def validate_image(path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "path": str(path),
        "valid": False,
        "width": None,
        "height": None,
        "format": None,
        "error": None,
    }
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            result["width"] = img.width
            result["height"] = img.height
            result["format"] = img.format
            result["valid"] = True
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        result["error"] = str(exc)
    return result


def file_content_hash(path: Path, chunk_size: int = 65536) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def find_duplicate_hashes(
    paths: Iterable[Path],
    max_files: Optional[int] = None,
) -> Dict[str, List[str]]:
    """Return hash → list of paths for duplicates (full hash, capped for large sets)."""
    paths_list = list(paths)
    if max_files is not None:
        paths_list = paths_list[:max_files]

    hash_to_paths: Dict[str, List[str]] = defaultdict(list)
    for path in paths_list:
        try:
            digest = file_content_hash(path)
            hash_to_paths[digest].append(str(path))
        except OSError:
            continue

    return {h: ps for h, ps in hash_to_paths.items() if len(ps) > 1}


def dimension_summary(valid_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    widths = [r["width"] for r in valid_results if r.get("width")]
    heights = [r["height"] for r in valid_results if r.get("height")]
    if not widths:
        return {"count": 0}
    return {
        "count": len(widths),
        "min_width": min(widths),
        "max_width": max(widths),
        "min_height": min(heights),
        "max_height": max(heights),
        "avg_width": round(sum(widths) / len(widths), 1),
        "avg_height": round(sum(heights) / len(heights), 1),
    }


def class_distribution_from_paths(paths: List[Path], dataset_root: Path) -> Dict[str, int]:
    counts: Counter = Counter()
    for path in paths:
        try:
            rel = path.relative_to(dataset_root)
            label = rel.parts[0] if len(rel.parts) > 1 else "root"
        except ValueError:
            label = path.parent.name or "unknown"
        counts[label] += 1
    return dict(counts)
