from dataclasses import dataclass


@dataclass
class TrainingConfig:
    model_name: str = "material_classifier"
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 0.001
    input_shape: tuple = (224, 224, 3)
    classes: tuple = ("Cotton", "Polyester", "Silk", "Linen", "Denim", "Wool", "Rayon", "Acrylic", "Nylon", "Mixed Fabric")
