import json
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from sklearn.model_selection import train_test_split

from app.ai.datasets.paths import (
    DATASETS_ROOT,
    MANIFESTS_DIR,
    PROCESSED_DIR,
    SPLITS_DIR,
    ensure_data_dirs,
)
from app.ai.datasets.validators import collect_image_files, is_image_file

# iBUG derived files: sample-<weave>-<sampleid>-im_<view>.png
# All views of one physical sample must stay in the same split.
IBUG_SAMPLE_NAME_RE = re.compile(
    r"^sample-(?P<folder>.+)-(?P<sample>\d+)-(?P<view>im_\d+)\.(png|jpg)$",
    re.IGNORECASE,
)


class DatasetManager:

    def __init__(self, root_dir: Optional[str] = None):
        self.root_dir = Path(root_dir) if root_dir else DATASETS_ROOT
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir = PROCESSED_DIR
        self.splits_dir = SPLITS_DIR
        ensure_data_dirs()

    def discover_datasets(self) -> List[Dict[str, object]]:
        discovered = []
        if not self.root_dir.exists():
            return discovered

        for path in sorted(self.root_dir.iterdir()):
            if not path.is_dir() or path.name.startswith("__"):
                continue
            images = collect_image_files(path)
            discovered.append({
                "name": path.name,
                "path": str(path),
                "exists": True,
                "file_count": len(images),
            })
        return discovered

    def validate_dataset(self, dataset_name: str) -> Dict[str, object]:
        path = self.root_dir / dataset_name

        if not path.exists():
            return {
                "name": dataset_name,
                "exists": False,
                "issues": ["dataset directory missing"],
            }

        image_files = collect_image_files(path)
        issues: List[str] = []

        if not image_files:
            issues.append("no image files found")

        # Class = immediate subfolder under dataset root (skip processed/organized)
        class_names = sorted({
            img.parent.name
            for img in image_files
            if img.parent != path and img.parent.name not in {"processed", "organized", "__pycache__"}
        })

        return {
            "name": dataset_name,
            "exists": True,
            "image_count": len(image_files),
            "class_count": len(class_names),
            "classes": class_names,
            "issues": issues,
        }

    def get_processed_dataset_path(self, name: str) -> Path:
        return self.processed_dir / name

    def get_split_path(self, processed_name: str, split: str) -> Path:
        return self.splits_dir / processed_name / split

    def load_manifest(self, name: str) -> Dict[str, object]:
        manifest_path = MANIFESTS_DIR / f"{name}_manifest.json"
        if not manifest_path.exists():
            return {}
        with open(manifest_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def create_training_split(
        self,
        dataset_name: str,
        output_dir: Optional[str] = None,
        test_size: float = 0.15,
        validation_size: float = 0.15,
        random_state: int = 42,
        *,
        use_processed: bool = True,
    ) -> Dict[str, object]:
        """
        Create train/validation/test copies under data/splits/.
        Prefers data/processed/{dataset_name} when use_processed=True.
        Falls back to Backend/datasets/{dataset_name} class folders.
        """
        if use_processed:
            processed_path = self.processed_dir / dataset_name
            if processed_path.exists() and any(processed_path.iterdir()):
                dataset_path = processed_path
            else:
                dataset_path = self.root_dir / dataset_name
        else:
            dataset_path = self.root_dir / dataset_name

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        image_files: List[Path] = []
        labels: List[str] = []
        for class_dir in sorted(d for d in dataset_path.iterdir() if d.is_dir()):
            for img in collect_image_files(class_dir, recursive=True):
                image_files.append(img)
                labels.append(class_dir.name)

        if not image_files:
            raise ValueError(f"No class-folder images found in {dataset_path}")

        train_files, temp_files, train_labels, temp_labels = train_test_split(
            image_files,
            labels,
            test_size=test_size + validation_size,
            stratify=labels,
            random_state=random_state,
        )

        relative_test_ratio = validation_size / (test_size + validation_size)
        val_files, test_files, _, _ = train_test_split(
            temp_files,
            temp_labels,
            test_size=relative_test_ratio,
            stratify=temp_labels,
            random_state=random_state,
        )

        output_path = Path(output_dir) if output_dir else self.splits_dir / dataset_name
        if output_path.exists():
            shutil.rmtree(output_path)

        for split, files in {
            "train": train_files,
            "validation": val_files,
            "test": test_files,
        }.items():
            for image_path in files:
                class_name = image_path.parent.name
                destination = output_path / split / class_name / image_path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_path, destination)

        return {
            "dataset": dataset_name,
            "source": str(dataset_path),
            "train_count": len(train_files),
            "validation_count": len(val_files),
            "test_count": len(test_files),
            "output_dir": str(output_path),
            "status": "prepared",
        }

    @staticmethod
    def sample_group_of(image_path: Path) -> str:
        """Group id so all views of one physical sample share one split."""
        match = IBUG_SAMPLE_NAME_RE.match(image_path.name)
        if match:
            return f"{image_path.parent.name}/{match.group('folder')}/{match.group('sample')}"
        return f"{image_path.parent.name}/{image_path.stem}"

    def create_group_aware_training_split(
        self,
        dataset_name: str,
        output_dir: Optional[str] = None,
        test_size: float = 0.15,
        validation_size: float = 0.15,
        random_state: int = 42,
    ) -> Dict[str, object]:
        """
        Duplicate/leakage-safe split: whole sample groups (all views of a
        physical fabric sample) are allocated to exactly one split.
        Used instead of create_training_split for iBUG-derived datasets.
        """
        processed_path = self.processed_dir / dataset_name
        dataset_path = (
            processed_path if processed_path.exists() and any(processed_path.iterdir())
            else self.root_dir / dataset_name
        )
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        rng = random.Random(random_state)
        groups_by_class: Dict[str, Dict[str, List[Path]]] = {}
        for class_dir in sorted(d for d in dataset_path.iterdir() if d.is_dir()):
            for img in collect_image_files(class_dir, recursive=True):
                groups_by_class.setdefault(class_dir.name, {}).setdefault(
                    self.sample_group_of(img), []
                ).append(img)

        allocation: Dict[str, Dict[str, List[Path]]] = {
            split: {} for split in ("train", "validation", "test")
        }
        group_total = 0
        for class_name, groups in groups_by_class.items():
            names = sorted(groups)
            rng.shuffle(names)
            group_total += len(names)
            n = len(names)
            n_test = max(1, round(n * test_size)) if n >= 2 else (1 if n == 1 else 0)
            n_val = max(1, round(n * validation_size)) if n >= 3 else 0
            # With very few groups keep at least one in train when possible.
            if n >= 3 and n_test + n_val >= n:
                n_val = max(0, n - n_test - 1)
            test_names = names[:n_test]
            val_names = names[n_test:n_test + n_val]
            train_names = names[n_test + n_val:]
            for split, split_names in (
                ("train", train_names), ("validation", val_names), ("test", test_names)
            ):
                for group_name in split_names:
                    allocation[split].setdefault(class_name, []).extend(groups[group_name])

        output_path = Path(output_dir) if output_dir else self.splits_dir / dataset_name
        if output_path.exists():
            shutil.rmtree(output_path)

        counts: Dict[str, int] = {}
        for split, per_class in allocation.items():
            count = 0
            for class_name, files in sorted(per_class.items()):
                for image_path in sorted(files):
                    destination = output_path / split / class_name / image_path.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(image_path, destination)
                    count += 1
            counts[split] = count

        return {
            "dataset": dataset_name,
            "source": str(dataset_path),
            "group_aware": True,
            "sample_groups": group_total,
            "train_count": counts["train"],
            "validation_count": counts["validation"],
            "test_count": counts["test"],
            "output_dir": str(output_path),
            "status": "prepared",
        }

    def summarize(self) -> Dict[str, object]:
        return {
            "datasets_root": str(self.root_dir),
            "processed_dir": str(self.processed_dir),
            "splits_dir": str(self.splits_dir),
            "datasets": self.discover_datasets(),
            "processed": [
                d.name for d in self.processed_dir.iterdir() if d.is_dir()
            ] if self.processed_dir.exists() else [],
        }
