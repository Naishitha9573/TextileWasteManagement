"""Central textile material taxonomy for the project.

This repository contains a broader intended taxonomy, while the active material
classifier is restricted to the five classes in MODEL_CLASSES.
"""

PROJECT_MATERIAL_TAXONOMY = [
    "Cotton",
    "Polyester",
    "Wool",
    "Silk",
    "Linen",
    "Denim",
    "Nylon",
    "Rayon",
    "Acrylic",
    "Mixed Fabrics",
]

MODEL_SUPPORTED_CLASSES = [
    "Cotton",
    "Nylon",
    "Polyester",
    "Silk",
    "Wool",
]
MODEL_CLASSES = tuple(MODEL_SUPPORTED_CLASSES)

VALID_MATERIAL_CLASSES = MODEL_SUPPORTED_CLASSES

MATERIAL_CLASSES = PROJECT_MATERIAL_TAXONOMY
MISSING_MATERIAL_CLASSES = [
    cls for cls in PROJECT_MATERIAL_TAXONOMY if cls not in VALID_MATERIAL_CLASSES
]

MATERIAL_CLASS_ALIASES = {
    "cotton": "Cotton",
    "polyester": "Polyester",
    "wool": "Wool",
    "silk": "Silk",
    "linen": "Linen",
    "denim": "Denim",
    "nylon": "Nylon",
    "rayon": "Rayon",
    "acrylic": "Acrylic",
    "mixed fabric": "Mixed Fabrics",
    "mixed fabrics": "Mixed Fabrics",
    "blend": "Mixed Fabrics",
    "blended": "Mixed Fabrics",
}


def normalize_material_class(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = text.replace("_", " ")
    key = normalized.lower()
    alias = MATERIAL_CLASS_ALIASES.get(key)
    if alias is not None:
        return alias
    for candidate in MATERIAL_CLASSES:
        if candidate.lower() == key:
            return candidate
    return text
