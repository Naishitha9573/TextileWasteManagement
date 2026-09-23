import json
import os
from pathlib import Path
from typing import Dict, Optional


class ModelRegistry:
    def __init__(self, root_dir: Optional[str] = None):
        self.root_dir = Path(root_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models"))
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.root_dir / "metadata.json"

    def save_metadata(self, payload: Dict[str, object]) -> None:
        with open(self.metadata_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def load_metadata(self) -> Dict[str, object]:
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        return {}

    def model_exists(self, model_name: str) -> bool:
        return (self.root_dir / f"{model_name}.json").exists()
