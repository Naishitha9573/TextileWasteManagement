import csv
import os
import tarfile
from functools import lru_cache
from typing import Dict, Any, List
from PIL import Image

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
DATASETS_ROOT = os.path.join(os.path.dirname(__file__), "tips")  # default root points to tips folder


def _resolve_dataset_root(root=None):
    base = root or os.path.dirname(__file__)
    if os.path.basename(os.path.normpath(base)) == "datasets":
        return base
    datasets_dir = os.path.join(base, "datasets")
    if os.path.isdir(datasets_dir):
        return datasets_dir
    return base


def dataset_dirs(root=None):
    root = _resolve_dataset_root(root)
    dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    return dirs


def _is_image_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def _collect_image_files(folder_path: str, recursive: bool = True) -> List[str]:
    if not os.path.isdir(folder_path):
        return []
    image_files = []
    if recursive:
        for root, _, files in os.walk(folder_path):
            for filename in files:
                if _is_image_file(filename):
                    image_files.append(os.path.join(root, filename))
    else:
        for filename in os.listdir(folder_path):
            path = os.path.join(folder_path, filename)
            if os.path.isfile(path) and _is_image_file(filename):
                image_files.append(path)
    return sorted(image_files)


def _count_csv_rows(csv_path: str) -> int:
    if not os.path.exists(csv_path):
        return 0
    try:
        with open(csv_path, newline='', encoding='utf-8') as handle:
            reader = csv.reader(handle)
            next(reader, None)
            return sum(1 for _ in reader)
    except Exception:
        return 0


def _split_tar_path(path: str) -> List[str]:
    return [segment for segment in path.split("/") if segment and segment != "."]


def _archive_signature(archive_path: str) -> str:
    try:
        stats = os.stat(archive_path)
        return f"{archive_path}|{int(stats.st_mtime)}|{stats.st_size}"
    except OSError:
        return archive_path


@lru_cache(maxsize=16)
def _scan_tar_metadata(archive_signature: str) -> Dict[str, Any]:
    archive_path = archive_signature.split("|", 1)[0]
    categories = set()
    image_count = 0

    try:
        with tarfile.open(archive_path, "r:*") as handle:
            for member in handle:
                if not member.name or member.isdir():
                    continue

                parts = _split_tar_path(member.name)
                if len(parts) >= 3:
                    categories.add(parts[1])

                _, ext = os.path.splitext(member.name)
                if ext.lower() in IMAGE_EXTENSIONS:
                    image_count += 1
    except Exception:
        return {"categories": [], "image_count": 0}

    return {"categories": sorted(categories), "image_count": image_count}


def _find_dataset_archive(dataset_path: str) -> str:
    if not os.path.isdir(dataset_path):
        return ""

    files = [
        os.path.join(dataset_path, name)
        for name in os.listdir(dataset_path)
        if os.path.isfile(os.path.join(dataset_path, name))
    ]

    if len(files) != 1:
        return ""

    candidate = files[0]
    try:
        if tarfile.is_tarfile(candidate):
            return candidate
    except Exception:
        return ""
    return ""


def _infer_categories_from_files(dataset_path: str) -> List[str]:
    if not os.path.isdir(dataset_path):
        return []

    entries = []
    for name in sorted(os.listdir(dataset_path)):
        full_path = os.path.join(dataset_path, name)
        if os.path.isdir(full_path):
            entries.append(name)
    if entries:
        return entries

    image_files = _collect_image_files(dataset_path)
    if image_files:
        return ["images"]
    return []


def _map_label_value_to_category(value: str) -> str:
    try:
        numeric_value = int(str(value).strip())
    except (TypeError, ValueError):
        return str(value)

    fashion_mnist_labels = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot",
    }
    return fashion_mnist_labels.get(numeric_value, str(value))


def _read_csv_categories(csv_path: str) -> List[str]:
    if not os.path.exists(csv_path):
        return []
    try:
        with open(csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return []
            field = next((name for name in ["Material_Type", "Sustainability_Rating", "Label", "Category", "label", "category", "class", "Class"] if name in reader.fieldnames), None)
            if not field:
                return []

            categories = []
            for row in reader:
                value = row.get(field)
                if not value:
                    continue
                if field.lower() in {"label", "category", "class"}:
                    categories.append(_map_label_value_to_category(value))
                else:
                    categories.append(str(value))
            return sorted({category for category in categories if category})
    except Exception:
        return []


def load_dataset_info(dataset_name: str, root=None) -> Dict[str, Any]:
    base = _resolve_dataset_root(root)
    path = os.path.join(base, dataset_name)
    info = {"name": dataset_name, "path": path, "exists": os.path.exists(path)}
    if not os.path.exists(path):
        return info

    classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    total_images = 0
    class_counts = {}
    for c in classes:
        cpath = os.path.join(path, c)
        image_paths = _collect_image_files(cpath)
        class_counts[c] = len(image_paths)
        total_images += len(image_paths)

    if total_images == 0:
        root_image_files = _collect_image_files(path)
        if root_image_files:
            total_images = len(root_image_files)

    csv_rows = 0
    for filename in sorted(os.listdir(path)):
        if filename.lower().endswith(".csv"):
            csv_rows += _count_csv_rows(os.path.join(path, filename))
    if total_images == 0 and csv_rows > 0:
        total_images = csv_rows

    archive_path = _find_dataset_archive(path)
    archive_metadata = {"categories": [], "image_count": 0}
    if archive_path:
        archive_metadata = _scan_tar_metadata(_archive_signature(archive_path))
        if total_images == 0:
            total_images = archive_metadata["image_count"]

    categories = _infer_categories_from_files(path)
    if not categories:
        for filename in sorted(os.listdir(path)):
            if filename.endswith(".csv"):
                categories = _read_csv_categories(os.path.join(path, filename))
                break
    if not categories and archive_metadata["categories"]:
        categories = archive_metadata["categories"]

    source_type = "unknown"
    if total_images > 0 and _collect_image_files(path):
        source_type = "image-folder"
    elif any(filename.lower().endswith(".csv") for filename in os.listdir(path)):
        source_type = "csv"
    elif archive_path:
        source_type = "archive"

    info.update({
        "classes": classes,
        "class_counts": class_counts,
        "total_images": total_images,
        "categories": categories,
        "source_type": source_type,
    })
    return info


def load_dataset_catalog(root=None) -> List[Dict[str, Any]]:
    base = _resolve_dataset_root(root)
    items = []
    for name in sorted(dataset_dirs(base)):
        if name.startswith("__"):
            continue
        info = load_dataset_info(name, root=base)
        items.append({
            "name": info["name"],
            "exists": info["exists"],
            "path": info["path"],
            "categories": info.get("categories", []),
            "class_counts": info.get("class_counts", {}),
            "total_images": info.get("total_images", 0),
            "source_type": info.get("source_type", "unknown"),
        })
    return items


def preview_image(dataset_name: str, class_name: str = None, index: int = 0, root=None):
    base = root or os.path.dirname(__file__)
    path = os.path.join(base, dataset_name)
    if not os.path.exists(path):
        return None
    classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    target_class = class_name or (classes[0] if classes else None)
    if not target_class:
        return None
    files = [f for f in os.listdir(os.path.join(path, target_class)) if os.path.isfile(os.path.join(path, target_class, f))]
    if not files:
        return None
    idx = index % len(files)
    img_path = os.path.join(path, target_class, files[idx])
    try:
        with open(img_path, 'rb') as f:
            return f.read()
    except Exception:
        return None
