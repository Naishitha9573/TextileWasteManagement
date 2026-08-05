import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageOps
import numpy as np


class DatasetPreprocessor:
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size

    def preprocess_image(self, image_path: str) -> np.ndarray:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            img = ImageOps.resize(img, self.target_size, Image.Resampling.BILINEAR)
            img = ImageOps.autocontrast(img)
            arr = np.array(img, dtype=np.float32) / 255.0
        return arr

    def preprocess_directory(self, directory: str) -> List[np.ndarray]:
        items = []
        for path in Path(directory).rglob("*.png"):
            items.append(self.preprocess_image(str(path)))
        for path in Path(directory).rglob("*.jpg"):
            items.append(self.preprocess_image(str(path)))
        return items
