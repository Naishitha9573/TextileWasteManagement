"""Fix the trainer file by rewriting it cleanly."""
from pathlib import Path

trainer_code = '''"""Train the independent TensorFlow EfficientNet-B0 waste classifier."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import ClassLabel, load_dataset
from sklearn.metrics import f1_score

DATASET_ID = "wargoninnovation/clothingdatasetsecondhand"
IMAGE_COLUMN = "image"
LABEL_COLUMN = "usage"
LABEL_NORMALIZATION = {
    "Export": "Export",
    "export": "Export",
    "Reuse": "Reuse",
    "reuse": "Reuse",
    "Recycle": "Recycle",
    "recycle": "Recycle",
    "Rcycle": "Recycle",
    "Energy recovery": "Energy Recovery",
    "Remake": "Remake",
    "Repair": "Repair",
}
NORMALIZED_CLASSES = ["Export", "Reuse", "Recycle", "Energy Recovery", "Remake", "Repair"]
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "models" / "waste_classification"
IMAGE_SIZE = (224, 224)
SEED = 42
SPLIT_MANIFEST = OUTPUT / "waste_train_validation_split.json"


def _preprocess(image, training: bool):
    """Preprocess image for EfficientNet-B0 with pure TensorFlow operations."""
    # Convert grayscale to RGB if needed
    if len(image.shape) == 2:
        image = tf.image.grayscale_to_rgb(image)
    elif image.shape[-1] == 1:
        image = tf.squeeze(image, axis=-1)
        image = tf.image.grayscale_to_rgb(image)
    
    # Cast to float32 for TensorFlow operations
    image = tf.cast(image, tf.float32)
    
    if training:
        # Resize to 256, then random crop to 224
        image = tf.image.resize(image, (256, 256))
        image = tf.image.random_crop(image, size=(224, 224, 3), seed=SEED)
        # Horizontal flip (50% probability)
        image = tf.image.random_flip_left_right(image, seed=SEED)
        # Brightness variation
        image = tf.image.random_brightness(image, max_delta=0.08, seed=SEED)
        # Contrast variation  
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1, seed=SEED)
    else:
        # Deterministic resize for validation
        image = tf.image.resize(image, IMAGE_SIZE)
    
    # Return float32 (Keras EfficientNet handles normalization internally)
    return image


def _tf_dataset(dataset, image_column: str, label_column: str, label_to_id: dict[str, int], training: bool, batch_size: int, class_weights: dict[str, float] | None = None):
    def generator():
        for row in dataset:
            image = np.asarray(row[image_column].convert("RGB"), dtype=np.uint8)
            label = row[label_column]
            if isinstance(label, (int, np.integer)):
                label = dataset.features[label_column].int2str(int(label))
            label = LABEL_NORMALIZATION.get(str(label))
            if label is None:
                raise ValueError(f"Unexpected usage label: {row[label_column]!r}")
            label_id = label_to_id[str(label)]
            if training and class_weights is not None:
                yield image, label_id, class_weights[str(label)]
            else:
                yield image, label_id

    if training and class_weights is not None:
        output_signature = (tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8), tf.TensorSpec(shape=(), dtype=tf.int32), tf.TensorSpec(shape=(), dtype=tf.float32))
    else:
        output_signature = (tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8), tf.TensorSpec(shape=(), dtype=tf.int32))
    result = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    
    if training and class_weights is not None:
        result = result.map(lambda image, label, weight: (_preprocess(image, training), label, weight), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        result = result.map(lambda image, label: (_preprocess(image, training), label), num_parallel_calls=tf.data.AUTOTUNE)
    
    if training:
        return result.shuffle(512, seed=SEED, reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        return result.batch(batch_size).prefetch(tf.data.AUTOTUNE)


class StreamSubset:
    def __init__(self, dataset, indices: set[int], features) -> None:
        self.dataset = dataset
        self.indices = indices
        self.features = features

    def __iter__(self):
        for index, row in enumerate(self.dataset):
            if index in self.indices:
                yield row


class ValidationMacroF1(tf.keras.callbacks.Callback):
    def __init__(self, validation_dataset, names: list[str], batch_size: int) -> None:
        super().__init__()
        self.validation_dataset = validation_dataset
        self.names = names
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        actual = []
        predicted = []
        for images, labels in self.validation_dataset:
            actual.extend(labels.numpy().tolist())
            predicted.extend(np.argmax(self.model.predict(images, verbose=0), axis=1).tolist())
        logs["val_macro_f1"] = float(f1_score(actual, predicted, average="macro", labels=list(range(len(self.names))), zero_division=0))
        print(f" - val_macro_f1: {logs['val_macro_f1']:.4f}")


def _build_model(names: list[str]):
    backbone = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3))
    backbone.trainable = False
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    outputs = tf.keras.layers.Dense(len(names), activation="softmax")(
        tf.keras.layers.Dropout(0.3)(
            tf.keras.layers.GlobalAveragePooling2D()(
                backbone(inputs, training=False)
            )
        )
    )
    return tf.keras.Model(inputs, outputs), backbone


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-column", default=LABEL_COLUMN)
    parser.add_argument("--verified-end-of-life-label", action="store_true", help="Required acknowledgement that this column is an end-of-life target.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    
    if not args.verified_end_of_life_label:
        raise SystemExit("Training blocked: verify that the selected dataset column contains waste/end-of-life labels first.")
    
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
    OUTPUT.mkdir(parents=True, exist_ok=True)
    
    if not SPLIT_MANIFEST.is_file():
        raise SystemExit(f"Missing split manifest: {SPLIT_MANIFEST}; run prepare_waste_train_validation_split.py first.")
    
    manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("test_used_for_split") or manifest.get("original_test_split", {}).get("modified"):
        raise SystemExit("Training blocked: test split safety check failed.")
    
    ds = load_dataset(DATASET_ID, streaming=True)
    
    if args.label_column != LABEL_COLUMN:
        raise SystemExit(f"The waste trainer requires the verified target column {LABEL_COLUMN!r}.")
    if args.label_column not in ds["train"].column_names:
        raise SystemExit(f"Unknown label column {args.label_column!r}; run waste_dataset_inspection.py first.")
    
    names = NORMALIZED_CLASSES
    label_to_id = {name: index for index, name in enumerate(names)}
    
    if IMAGE_COLUMN not in ds["train"].column_names:
        raise SystemExit(f"Required image feature {IMAGE_COLUMN!r} was not found.")
    
    image_column = IMAGE_COLUMN
    (OUTPUT / "class_names.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    
    source_train = ds["train"]
    features = source_train.features
    train_indices = set(manifest["indices"]["train"])
    validation_indices = set(manifest["indices"]["validation"])
    
    train_subset = StreamSubset(source_train, train_indices, features)
    validation_subset = StreamSubset(source_train, validation_indices, features)
    
    train_distribution = manifest["train_distribution"]
    total_train = sum(train_distribution.values())
    class_weights = {name: float(total_train / (len(names) * train_distribution[name])) for name in names}
    
    print(f"TRAINING SAMPLES: {len(train_indices)}")
    print(f"VALIDATION SAMPLES: {len(validation_indices)}")
    print(f"TEST SAMPLES: {manifest['original_test_split']['samples']} (untouched, not loaded)")
    print(f"CLASS WEIGHTS: {class_weights}")
    
    classifier, backbone = _build_model(names)
    classifier.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    train = _tf_dataset(train_subset, image_column, args.label_column, label_to_id, True, args.batch_size, class_weights)
    validation = _tf_dataset(validation_subset, image_column, args.label_column, label_to_id, False, args.batch_size)
    
    stage1_path = OUTPUT / "stage1_best_waste_model.keras"
    callbacks = [
        ValidationMacroF1(validation, names, args.batch_size),
        tf.keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_macro_f1", mode="max", patience=2),
        tf.keras.callbacks.ModelCheckpoint(stage1_path, monitor="val_macro_f1", mode="max", save_best_only=True)
    ]
    
    print("\\n=== STAGE 1: Frozen Backbone Training ===")
    history = classifier.fit(train, validation_data=validation, epochs=args.epochs, callbacks=callbacks)
    
    # Stage 2: Fine-tuning
    classifier = tf.keras.models.load_model(stage1_path)
    backbone = next(layer for layer in classifier.layers if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower())
    
    # Unfreeze upper layers
    for layer in backbone.layers:
        layer.trainable = False
    for layer in backbone.layers[-30:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    
    classifier.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    stage2_path = OUTPUT / "best_waste_model.keras"
    callbacks = [
        ValidationMacroF1(validation, names, args.batch_size),
        tf.keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_macro_f1", mode="max", patience=2),
        tf.keras.callbacks.ModelCheckpoint(stage2_path, monitor="val_macro_f1", mode="max", save_best_only=True)
    ]
    
    print("\\n=== STAGE 2: Upper Layer Fine-tuning ===")
    history2 = classifier.fit(train, validation_data=validation, epochs=args.epochs, callbacks=callbacks)
    
    # Select best model across stages
    best_history = history.history.get("val_macro_f1", []) + history2.history.get("val_macro_f1", [])
    best_index = int(np.argmax(best_history))
    best_stage = "stage1" if best_index < len(history.history.get("val_macro_f1", [])) else "stage2"
    best_epoch = (best_index + 1) if best_stage == "stage1" else (best_index - len(history.history.get("val_macro_f1", [])) + 1)
    
    best_values = {
        "best_epoch": best_epoch,
        "best_validation_macro_f1": float(max(best_history)),
        "best_stage": best_stage,
        "test_evaluation": "NOT PERFORMED",
        "test_used": False,
    }
    
    (OUTPUT / "training_history.json").write_text(json.dumps({"stage1": history.history, "stage2": history2.history, "best": best_values}, indent=2), encoding="utf-8")
    
    print("\\n" + "=" * 50)
    print("WASTE CLASSIFIER TRAINING COMPLETED")
    print("=" * 50)
    print(f"Best Stage: {best_stage}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Validation Macro F1: {best_values['best_validation_macro_f1']:.6f}")
    print(f"Best Checkpoint: {stage1_path if best_stage == 'stage1' else stage2_path}")
    print(f"Test Set: UNTOUCHED ({manifest['original_test_split']['samples']} samples preserved)")
    print(f"Material Classifier: UNCHANGED")
    print("=" * 50)


if __name__ == "__main__":
    main()
'''

Path('c:/Users/sama/OneDrive/Desktop/naishtex/Textile/src/train_waste_classifier.py').write_text(trainer_code, encoding='utf-8')
print("✅ Trainer file rewritten successfully")
