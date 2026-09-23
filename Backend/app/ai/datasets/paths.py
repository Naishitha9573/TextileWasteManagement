"""Central path resolution for dataset and data pipeline directories."""
from pathlib import Path
from typing import Optional

# Backend/ (parent of app/)
BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent.parent

DATA_ROOT = BACKEND_ROOT / "data"
RAW_DIR = DATA_ROOT / "raw"
INTERIM_DIR = DATA_ROOT / "interim"
PROCESSED_DIR = DATA_ROOT / "processed"
MANIFESTS_DIR = DATA_ROOT / "manifests"
SPLITS_DIR = DATA_ROOT / "splits"
METADATA_DIR = DATA_ROOT / "metadata"
REPORTS_DIR = DATA_ROOT / "reports"
REJECTED_DIR = DATA_ROOT / "rejected"
REJECTED_DUPLICATES_DIR = REJECTED_DIR / "duplicates"

DATASETS_ROOT = BACKEND_ROOT / "datasets"
CONFIG_DIR = BACKEND_ROOT / "config"
CLASS_MAPPING_PATH = CONFIG_DIR / "class_mapping.yaml"
PROJECT_ROOT = BACKEND_ROOT.parent


def ensure_data_dirs() -> None:
    for directory in (
        RAW_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        MANIFESTS_DIR,
        SPLITS_DIR,
        METADATA_DIR,
        REPORTS_DIR,
        REJECTED_DIR,
        REJECTED_DUPLICATES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def resolve_datasets_root(root: Optional[str] = None) -> Path:
    if root:
        return Path(root)
    return DATASETS_ROOT

