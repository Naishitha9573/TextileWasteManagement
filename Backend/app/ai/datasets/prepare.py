"""Prepare processed datasets and splits without modifying originals."""
from __future__ import annotations

import csv
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sklearn.model_selection import train_test_split

from app.ai.datasets.mapping import extract_material_from_caption, load_class_mapping
from app.ai.datasets.paths import (
    DATASETS_ROOT,
    MANIFESTS_DIR,
    PROJECT_ROOT,
    PROCESSED_DIR,
    SPLITS_DIR,
    ensure_data_dirs,
)
from app.ai.datasets.validators import collect_image_files, is_image_file


RANDOM_SEED = 42


def _copy_or_link(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    shutil.copy2(source, dest)


def prepare_fabric_condition_dataset(
    datasets_root: Path,
    processed_root: Path,
) -> Dict[str, Any]:
    """Organize fabric_dataset defect folders → condition labels (not materials)."""
    mapping = load_class_mapping()
    fabric_cfg = mapping.get("fabric_dataset", {})
    source_folders = fabric_cfg.get("source_folders", {})

    dataset_path = datasets_root / "fabric_dataset"
    output_name = "fabric_condition"
    output_path = processed_root / output_name

    entries: List[Dict[str, Any]] = []
    class_counts: Dict[str, int] = {}

    for folder_name, target_label in source_folders.items():
        if not target_label:
            continue
        folder = dataset_path / folder_name
        if not folder.exists():
            continue
        for img in collect_image_files(folder, recursive=True):
            rel_dest = output_path / target_label / img.name
            _copy_or_link(img, rel_dest)
            entries.append({
                "image_path": str(rel_dest),
                "source_dataset": "fabric_dataset",
                "original_label": folder_name,
                "normalized_label": target_label,
                "label_source": "dataset_folder",
                "confidence": 1.0,
                "verification_status": "VERIFIED",
            })
            class_counts[target_label] = class_counts.get(target_label, 0) + 1

    manifest = {
        "dataset": output_name,
        "purpose": "condition_defect_classification",
        "maps_to_material_classes": False,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "class_counts": class_counts,
        "entries": entries,
    }
    manifest_path = MANIFESTS_DIR / f"{output_name}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return {
        "name": output_name,
        "output_dir": str(output_path),
        "class_counts": class_counts,
        "total_images": len(entries),
        "manifest": str(manifest_path),
    }


def prepare_deepfashion_material_manifest(
    datasets_root: Path,
    processed_root: Path,
    *,
    max_per_class: Optional[int] = None,
    copy_images: bool = True,
) -> Dict[str, Any]:
    """Build material-class folders from DeepFashion captions (keyword mapping)."""
    captions_path = datasets_root / "deepfashion" / "captions.json"
    images_root = datasets_root / "deepfashion" / "images"

    if not captions_path.exists():
        return {"name": "material_deepfashion", "error": "captions.json missing", "total_images": 0}

    with open(captions_path, "r", encoding="utf-8") as handle:
        captions: Dict[str, str] = json.load(handle)

    mapping = load_class_mapping()
    output_name = "material_deepfashion"
    output_path = processed_root / output_name

    image_index: Dict[str, Path] = {}
    if images_root.exists():
        for img in collect_image_files(images_root):
            image_index[img.name] = img

    class_buckets: Dict[str, List[Tuple[str, Path, str]]] = {}
    skipped_no_file = 0
    skipped_unmapped = 0

    for filename, caption in captions.items():
        material = extract_material_from_caption(caption, mapping)
        if not material:
            skipped_unmapped += 1
            continue
        source = image_index.get(filename)
        if not source:
            skipped_no_file += 1
            continue
        class_buckets.setdefault(material, []).append((filename, source, caption))

    random.seed(RANDOM_SEED)
    entries: List[Dict[str, Any]] = []
    class_counts: Dict[str, int] = {}

    for material, items in sorted(class_buckets.items()):
        if max_per_class and len(items) > max_per_class:
            items = random.sample(items, max_per_class)

        for filename, source, caption in items:
            dest = output_path / material / filename
            if copy_images:
                _copy_or_link(source, dest)
            entries.append({
                "image_path": str(dest) if copy_images else str(source),
                "source_dataset": "deepfashion",
                "original_label": caption[:150],
                "normalized_label": material,
                "label_source": "caption_keyword",
                "confidence": 0.70 if material != "Mixed Fabrics" else 0.50,
                "verification_status": "NOISY",
            })
            class_counts[material] = class_counts.get(material, 0) + 1

    manifest = {
        "dataset": output_name,
        "purpose": "material_classification",
        "maps_to_material_classes": True,
        "mapping_source": "deepfashion_caption_keywords",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "warning": "Labels inferred from caption keywords, not verified fabric composition.",
        "skipped_unmapped_captions": skipped_unmapped,
        "skipped_missing_files": skipped_no_file,
        "class_counts": class_counts,
        "entries": entries,
    }
    manifest_path = MANIFESTS_DIR / f"{output_name}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    # Export dataset_manifest.csv -> Backend/data/manifests/dataset_manifest.csv
    csv_manifest_path = MANIFESTS_DIR / "dataset_manifest.csv"
    with open(csv_manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "image_path",
            "source_dataset",
            "original_label",
            "normalized_label",
            "label_source",
            "confidence",
            "verification_status",
        ])
        for entry in entries:
            writer.writerow([
                entry["image_path"],
                entry["source_dataset"],
                entry["original_label"],
                entry["normalized_label"],
                entry["label_source"],
                entry["confidence"],
                entry["verification_status"],
            ])

    return {
        "name": output_name,
        "output_dir": str(output_path),
        "class_counts": class_counts,
        "total_images": len(entries),
        "manifest": str(manifest_path),
        "entries": entries,
    }


def create_splits_from_processed(
    processed_dataset_name: str,
    *,
    test_size: float = 0.15,
    validation_size: float = 0.15,
    random_state: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Create train/validation/test splits under data/splits/ and export CSVs."""
    source_root = PROCESSED_DIR / processed_dataset_name
    if not source_root.exists():
        raise FileNotFoundError(f"Processed dataset not found: {source_root}")

    image_files: List[Path] = []
    labels: List[str] = []
    for class_dir in sorted(d for d in source_root.iterdir() if d.is_dir()):
        for img in collect_image_files(class_dir, recursive=False):
            image_files.append(img)
            labels.append(class_dir.name)

    if not image_files:
        raise ValueError(f"No images in processed dataset: {source_root}")

    train_files, temp_files, train_labels, temp_labels = train_test_split(
        image_files,
        labels,
        test_size=test_size + validation_size,
        stratify=labels,
        random_state=random_state,
    )

    relative_val_ratio = validation_size / (test_size + validation_size)
    val_files, test_files, _, _ = train_test_split(
        temp_files,
        temp_labels,
        test_size=relative_val_ratio,
        stratify=temp_labels,
        random_state=random_state,
    )

    split_root = SPLITS_DIR / processed_dataset_name
    if split_root.exists():
        shutil.rmtree(split_root)

    split_counts = {"train": 0, "validation": 0, "test": 0}
    split_manifest: Dict[str, List[Dict[str, str]]] = {"train": [], "validation": [], "test": []}

    for split_name, files in {"train": train_files, "validation": val_files, "test": test_files}.items():
        # Also write split CSV to Backend/data/splits/<split_name>.csv
        csv_file_path = SPLITS_DIR / f"{split_name}.csv"
        with open(csv_file_path, "a", newline="", encoding="utf-8") as csv_handle:
            writer = csv.writer(csv_handle)
            if csv_handle.tell() == 0:
                writer.writerow(["image_path", "normalized_label", "split", "dataset"])

            for path in files:
                label = path.parent.name
                dest = split_root / split_name / label / path.name
                _copy_or_link(path, dest)
                split_manifest[split_name].append({
                    "source": str(path),
                    "split_path": str(dest),
                    "label": label,
                })
                split_counts[split_name] += 1
                writer.writerow([str(dest), label, split_name, processed_dataset_name])

    manifest = {
        "dataset": processed_dataset_name,
        "split_root": str(split_root),
        "random_state": random_state,
        "test_size": test_size,
        "validation_size": validation_size,
        "split_counts": split_counts,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = MANIFESTS_DIR / f"{processed_dataset_name}_splits.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump({**manifest, "entries": split_manifest}, handle, indent=2)

    return manifest


def generate_class_balance_report(datasets_results: List[Dict[str, Any]]) -> None:
    lines = [
        "# Class Balance Report — AI Textile Waste Intelligence Platform",
        "",
        "**Date:** 2026-08-18  ",
        "**Strategy:** Controlled sampling per class (max 500), class weighting during training, NO raw duplication.",
        "",
        "---",
        "",
        "## Class Distributions",
        "",
    ]
    for ds in datasets_results:
        lines.append(f"### Dataset: `{ds.get('name')}`")
        lines.append("")
        lines.append("| Material / Label | Prepared Images | Imbalance Status | Handling Strategy |")
        lines.append("|------------------|----------------:|------------------|-------------------|")
        counts = ds.get("class_counts", {})
        total = sum(counts.values())
        avg = total / max(len(counts), 1)
        for label, count in counts.items():
            ratio = count / avg if avg > 0 else 1.0
            status = "Balanced" if 0.8 <= ratio <= 1.2 else ("Under-represented" if ratio < 0.8 else "Over-represented")
            strategy = "Class weight adjustment (`compute_class_weight`)" if status != "Balanced" else "Standard sampling"
            lines.append(f"| **{label}** | {count:,} | {status} | {strategy} |")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## Balancing Directives",
        "",
        "1. **No Synthetic Duplication:** Training images are never artificially cloned to pad missing or minority classes.",
        "2. **Controlled Augmentation:** Autocontrast, random flip, and minor rotation applied dynamically during training steps on train split only.",
        "3. **Loss Weighting:** Compute balanced class weights via `sklearn.utils.class_weight.compute_class_weight` during Keras model training.",
        "",
        "---",
        "*Generated by `Backend/app/ai/datasets/prepare.py`*",
    ])

    report_path = PROJECT_ROOT / "CLASS_BALANCE_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_prepare(
    *,
    include_deepfashion: bool = True,
    deepfashion_max_per_class: Optional[int] = 500,
    create_splits: bool = True,
) -> Dict[str, Any]:
    ensure_data_dirs()

    # Clear existing split CSVs
    for split_name in ("train", "validation", "test"):
        csv_p = SPLITS_DIR / f"{split_name}.csv"
        if csv_p.exists():
            csv_p.unlink()

    results: Dict[str, Any] = {
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "datasets": [],
    }

    fabric_result = prepare_fabric_condition_dataset(DATASETS_ROOT, PROCESSED_DIR)
    results["datasets"].append(fabric_result)

    if include_deepfashion:
        df_result = prepare_deepfashion_material_manifest(
            DATASETS_ROOT,
            PROCESSED_DIR,
            max_per_class=deepfashion_max_per_class,
            copy_images=True,
        )
        results["datasets"].append(df_result)

    if create_splits:
        results["splits"] = []
        for item in results["datasets"]:
            if item.get("total_images", 0) > 0 and item.get("name"):
                try:
                    split_info = create_splits_from_processed(item["name"])
                    results["splits"].append(split_info)
                except Exception as exc:
                    results.setdefault("split_errors", []).append({
                        "dataset": item["name"],
                        "error": str(exc),
                    })

    generate_class_balance_report(results["datasets"])

    summary_path = MANIFESTS_DIR / "prepare_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    return results


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Prepare datasets for ML pipeline")
    parser.add_argument("--skip-deepfashion", action="store_true")
    parser.add_argument("--max-per-class", type=int, default=500)
    parser.add_argument("--no-splits", action="store_true")
    args = parser.parse_args()

    results = run_prepare(
        include_deepfashion=not args.skip_deepfashion,
        deepfashion_max_per_class=args.max_per_class,
        create_splits=not args.no_splits,
    )
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
