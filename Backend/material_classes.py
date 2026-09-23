"""Centralized material taxonomy for the Textile Waste Intelligence platform.

Two distinct scopes live here:

- MATERIAL_CLASSES: the ten target material categories supported by the
  platform's inventory, scoring rules and sustainability analytics.
- MODEL_SUPPORTED_CLASSES: the classes the currently trained MobileNetV3
  material classifier can actually predict (5 of the 10 target materials,
  manufacturer-tag-derived iBUG labels). Inference code must treat any
  prediction outside this list as unsupported rather than guessing.
"""

MATERIAL_CLASSES = [
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

# Classes the active CNN model (material-mobilenetv3-v0.2-5class-ibug,
# dataset ibug_material_v2) was trained on.
MODEL_SUPPORTED_CLASSES = [
    "Cotton",
    "Nylon",
    "Polyester",
    "Silk",
    "Wool",
]

# The single classifier contract shared by dataset preparation, training,
# inference, API responses, and reports.
MODEL_CLASSES = tuple(MODEL_SUPPORTED_CLASSES)

VALID_MATERIAL_CLASSES = MATERIAL_CLASSES


def normalize_material_name(name):
    """Map common aliases/spellings onto the canonical taxonomy."""
    if not isinstance(name, str):
        return None
    cleaned = name.strip()
    if not cleaned:
        return None
    lowered = cleaned.lower()
    aliases = {
        "cotton": "Cotton",
        "polyester": "Polyester",
        "wool": "Wool",
        "silk": "Silk",
        "linen": "Linen",
        "denim": "Denim",
        "nylon": "Nylon",
        "rayon": "Rayon",
        "viscose": "Rayon",
        "acrylic": "Acrylic",
        "mixed": "Mixed Fabrics",
        "mixed fabrics": "Mixed Fabrics",
        "mixed fabric": "Mixed Fabrics",
        "blend": "Mixed Fabrics",
        "blended": "Mixed Fabrics",
    }
    return aliases.get(lowered)
