"""[DEPRECATED] Keras/MobileNetV3 training pipeline.

DEPRECATION STATUS: This module is no longer used by the active EfficientNet-B0 inference pipeline.
It is preserved for historical reference and comparison only.

ACTIVE INFERENCE MODEL: EfficientNet-B0 (PyTorch) — see Backend/app/services/fabric_classifier.py
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
import numpy as np

from app.ai.dataset_manager import DatasetManager
from app.ai.dataset_preprocessor import DatasetPreprocessor
from app.ai.training_config import TrainingConfig
from app.ai.model_registry import ModelRegistry
from app.ai.evaluate import Evaluator

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from app.ai.mobilenet_preprocess import mobilenet_v3_preprocess
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None

from material_classes import MODEL_CLASSES


def build_material_classifier(
    num_classes: int,
    config: TrainingConfig,
):
    """Build transfer-learning classifier (MobileNetV3-Small) or fallback CNN."""
    inputs = keras.Input(shape=config.input_shape)

    x = inputs
    if config.use_augmentation:
        # Realistic textile transforms only: fabric texture must survive.
        x = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.05, 0.05),
            layers.RandomBrightness(0.12),
            layers.RandomContrast(0.15),
        ], name="textile_augmentation")(x)

    if config.architecture == "mobilenet_v3_small":
        # Preprocessor returns [0, 1]; MobileNet preprocess_input expects [0, 255]
        x = layers.Lambda(
            mobilenet_v3_preprocess,
            output_shape=lambda shape: shape,
            name="mobilenet_preprocess",
        )(x)
        base = keras.applications.MobileNetV3Small(
            include_top=False,
            weights="imagenet",
            input_tensor=x,
            pooling="avg",
        )
        base.trainable = False
        x = base.output
        x = layers.Dropout(config.dropout_rate)(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="material_output")(x)
        model = keras.Model(inputs, outputs)
    else:
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        model = keras.Model(inputs, outputs)

    loss = keras.losses.CategoricalCrossentropy(
        label_smoothing=config.label_smoothing
    ) if config.label_smoothing > 0 else "sparse_categorical_crossentropy"
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=loss,
        metrics=["accuracy"],
    )
    return model


class TrainingPipeline:
    def __init__(
        self,
        root_dir: Optional[str] = None,
        dataset_name: Optional[str] = None,
        config: Optional[TrainingConfig] = None,
    ):
        self.root_dir = Path(root_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "models",
        ))
        self.root_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or TrainingConfig()
        self.dataset_manager = DatasetManager()
        self.preprocessor = DatasetPreprocessor()
        self.registry = ModelRegistry(str(self.root_dir))
        self.evaluator = Evaluator()

        self.dataset_name = dataset_name or self.config.dataset_name
        self.metrics_path = self.root_dir / "model_metrics.json"
        safe_version = self.config.model_version.replace("/", "-")
        self.model_path = self.root_dir / f"{safe_version}.keras"
        self.label_map_path = self.root_dir / "label_map.json"
        self.baseline_sklearn_path = self.root_dir / "baseline_sklearn_v0.1_3class.joblib"

    def train(self) -> Dict[str, object]:
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError(
                "TensorFlow is required for MobileNetV3 training. "
                "Install with: pip install tensorflow>=2.16.0 "
                f"(~350MB+ free disk required). "
                "Sklearn baseline preserved at baseline_sklearn_v0.1_3class.joblib"
            )
        return self._train_keras()

    @staticmethod
    def _get_base_model(model):
        """Kept for compatibility; base layers are inlined in the outer model."""
        for layer in model.layers:
            if "mobilenet" in layer.name.lower() and hasattr(layer, "layers"):
                return layer
        return None

    def _train_keras(self) -> Dict[str, object]:
        import shutil

        sklearn_current = self.root_dir / "material_classifier.joblib"
        if sklearn_current.exists() and not self.baseline_sklearn_path.exists():
            shutil.copy2(sklearn_current, self.baseline_sklearn_path)
            print(f"Preserved sklearn baseline: {self.baseline_sklearn_path}")

        # Archive the currently active artifacts so rollback stays possible.
        try:
            previous = self.registry.load_metadata() or {}
            previous_version = previous.get("model_version") or previous.get(
                "metrics", {}
            ).get("model_version")
            if previous_version and previous_version != self.config.model_version:
                archive_dir = self.root_dir / "archive" / previous_version
                archive_dir.mkdir(parents=True, exist_ok=True)
                for name in ("metadata.json", "model_metrics.json", "label_map.json"):
                    src = self.root_dir / name
                    if src.exists():
                        shutil.copy2(src, archive_dir / name)
                print(f"Archived previous model metadata -> {archive_dir}")
        except Exception as exc:
            print(f"WARN: could not archive previous metadata ({exc})")

        tf.random.set_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

        processed_path = self.dataset_manager.get_processed_dataset_path(self.dataset_name)
        if processed_path.exists():
            split_path = self.dataset_manager.get_split_path(self.dataset_name, "train")
            if not split_path.exists():
                print(f"Creating GROUP-AWARE train/val/test split for {self.dataset_name}...")
                # Sample-group aware so no physical sample leaks across splits.
                self.dataset_manager.create_group_aware_training_split(
                    self.dataset_name,
                    test_size=0.15,
                    validation_size=0.15,
                    random_state=self.config.random_seed,
                )
            dataset_info = {
                "exists": True,
                "classes": sorted(d.name for d in processed_path.iterdir() if d.is_dir()),
                "image_count": sum(
                    len(list(d.rglob("*.jpg"))) + len(list(d.rglob("*.png")))
                    for d in processed_path.iterdir() if d.is_dir()
                ),
            }
        else:
            dataset_info = self.dataset_manager.validate_dataset(self.dataset_name)

        if not dataset_info.get("exists") or dataset_info.get("image_count", 0) == 0:
            raise RuntimeError(
                f"Dataset '{self.dataset_name}' not found. Run: python -m app.ai.datasets.prepare"
            )

        print("Loading and preprocessing images...")
        x_train, y_train, _ = self._load_split_data("train")
        x_val, y_val, _ = self._load_split_data("validation")
        x_test, y_test, class_names = self._load_split_data("test")

        if len(x_train) == 0:
            raise RuntimeError("No training images found after preprocessing")

        # Class index mapping is PINNED to the platform contract — the exact
        # order used by inference. Never rely on incidental folder order.
        if class_names != list(MODEL_CLASSES):
            raise RuntimeError(
                f"Split class order {class_names} does not match the required "
                f"class contract {list(MODEL_CLASSES)}"
            )
        num_classes = len(class_names)
        print(f"Training {self.config.architecture} with {num_classes} classes: {class_names}")

        model = build_material_classifier(num_classes, self.config)

        sample_weight_train = None
        if self.config.use_class_weights:
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y_train)
            weights = compute_class_weight("balanced", classes=classes, y=y_train)
            weight_by_class = {int(c): float(w) for c, w in zip(classes, weights)}
            sample_weight_train = np.array(
                [weight_by_class[int(label)] for label in y_train], dtype=np.float32
            )
            print(f"Class weights (from actual distribution): {weight_by_class}")
            class_weight = None

        def _targets(y):
            # Label-smoothed CategoricalCrossentropy expects one-hot targets.
            if self.config.label_smoothing > 0:
                return keras.utils.to_categorical(y, num_classes=num_classes)
            return y

        def make_callbacks(run_tag: str):
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True,
                ),
                keras.callbacks.ModelCheckpoint(
                    filepath=str(self.model_path),
                    monitor="val_accuracy",
                    save_best_only=True,
                    verbose=1,
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=self.config.reduce_lr_factor,
                    patience=self.config.reduce_lr_patience,
                    min_lr=self.config.reduce_lr_min_lr,
                    verbose=1,
                ),
                keras.callbacks.CSVLogger(
                    str(self.root_dir / f"training_log_{run_tag}.csv"), append=True
                ),
            ]
            return callbacks

        head_epochs = max(1, self.config.head_epochs)
        fine_tune_epochs = max(0, self.config.finetune_epochs)

        print(f"Phase A: training classification head (frozen base) for up to {head_epochs} epochs...")
        history_a = model.fit(
            x_train, _targets(y_train),
            validation_data=(x_val, _targets(y_val)),
            epochs=head_epochs,
            batch_size=self.config.batch_size,
            sample_weight=sample_weight_train,
            callbacks=make_callbacks("phase_a"),
            verbose=1,
        )

        histories = [history_a.history]
        # NOTE: build_material_classifier constructs MobileNetV3Small with
        # input_tensor=..., so its layers are INLINED into the outer model —
        # there is no nested base-model object. Unfreeze by layer index instead.
        if fine_tune_epochs > 0:
            print(f"Phase B: unfreezing layers from index {self.config.fine_tune_at} for up to {fine_tune_epochs} epochs (lr={self.config.finetune_learning_rate})...")
            for index, layer in enumerate(model.layers):
                layer.trainable = index >= self.config.fine_tune_at
            frozen_bn = 0
            for layer in model.layers:
                if isinstance(layer, layers.BatchNormalization):
                    layer.trainable = False
                    frozen_bn += 1
            trainable_count = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
            print(f"  trainable parameters after unfreeze: {trainable_count:,} (BatchNorm frozen: {frozen_bn})")
            if trainable_count == 0:
                raise RuntimeError("Fine-tuning produced zero trainable parameters — check fine_tune_at index.")
            loss_b = keras.losses.CategoricalCrossentropy(
                label_smoothing=self.config.label_smoothing
            ) if self.config.label_smoothing > 0 else "sparse_categorical_crossentropy"
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.config.finetune_learning_rate),
                loss=loss_b,
                metrics=["accuracy"],
            )
            history_b = model.fit(
                x_train, _targets(y_train),
                validation_data=(x_val, _targets(y_val)),
                epochs=head_epochs + fine_tune_epochs,
                initial_epoch=len(history_a.history["accuracy"]),
                batch_size=self.config.batch_size,
                sample_weight=sample_weight_train,
                callbacks=make_callbacks("phase_b"),
                verbose=1,
            )
            histories.append(history_b.history)
        else:
            print("Phase B skipped (fine-tuning disabled).")

        # The checkpoint on disk holds the best-val_accuracy weights; make sure
        # calibration + evaluation use exactly the promoted artifact.
        if self.model_path.exists():
            try:
                from app.ai.model_loader import load_keras_material_model
                model = load_keras_material_model(str(self.model_path))
                print("Reloaded best checkpoint for evaluation/calibration.")
            except Exception as exc:
                print(f"WARN: could not reload checkpoint ({exc}); using in-memory weights.")

        combined_history = {}
        for key in histories[0].keys():
            combined_history[key] = [v for h in histories for v in h.get(key, [])]
        print("Evaluating on held-out test set...")
        y_pred_proba = model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        eval_result = self.evaluator.evaluate(y_test, y_pred, class_names)

        # Confidence calibration on the VALIDATION set (never test).
        val_proba = model.predict(x_val, verbose=0)
        calibration = self._calibrate_thresholds(val_proba, y_val)
        print(f"Calibrated thresholds (validation): {calibration}")

        model.save(str(self.model_path))
        print(f"Model saved to: {self.model_path}")

        label_map = {str(i): name for i, name in enumerate(class_names)}
        with open(self.label_map_path, "w", encoding="utf-8") as handle:
            json.dump(label_map, handle, indent=2)
        safe_version_local = self.config.model_version.replace("/", "-")
        versioned_label_map = self.root_dir / f"label_map_{safe_version_local}.json"
        with open(versioned_label_map, "w", encoding="utf-8") as handle:
            json.dump(label_map, handle, indent=2)

        train_history = {
            "epochs": list(range(1, len(combined_history["accuracy"]) + 1)),
            "train_accuracy": [float(v) for v in combined_history["accuracy"]],
            "validation_accuracy": [float(v) for v in combined_history["val_accuracy"]],
            "train_loss": [float(v) for v in combined_history["loss"]],
            "validation_loss": [float(v) for v in combined_history["val_loss"]],
        }

        metrics = {
            "status": "evaluated",
            "model_name": self.config.model_name,
            "model_version": self.config.model_version,
            "architecture": self.config.architecture,
            "backend": "keras",
            "current_model_scope": f"{num_classes} CLASS ONLY",
            "model_path": str(self.model_path),
            "dataset": self.dataset_name,
            "dataset_version": "phase1_prepared",
            "class_names": class_names,
            "label_mapping_note": self._label_mapping_note(),
            "label_provenance": self.config.label_provenance,
            "accuracy": eval_result["accuracy"],
            "balanced_accuracy": eval_result["balanced_accuracy"],
            "precision": eval_result["precision"],
            "recall": eval_result["recall"],
            "f1_score": eval_result["f1_score"],
            "macro_precision": eval_result["macro_precision"],
            "macro_recall": eval_result["macro_recall"],
            "macro_f1": eval_result["macro_f1"],
            "confusion_matrix": eval_result["confusion_matrix"],
            "classification_report": eval_result["classification_report"],
            "calibration": calibration,
            "label_smoothing": self.config.label_smoothing,
            "dropout_rate": self.config.dropout_rate,
            "class_weighted_sampling": bool(self.config.use_class_weights),
            "train_samples": len(x_train),
            "validation_samples": len(x_val),
            "test_samples": len(x_test),
            "num_classes": num_classes,
            "epochs_run": len(combined_history["accuracy"]),
            "epochs_configured": self.config.epochs,
            "head_epochs_configured": head_epochs,
            "finetune_epochs_configured": fine_tune_epochs,
            "fine_tune_at_layer_index": self.config.fine_tune_at if fine_tune_epochs > 0 else None,
            "two_phase_training": bool(fine_tune_epochs > 0),
            "use_augmentation": self.config.use_augmentation,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "fine_tune_learning_rate": self.config.finetune_learning_rate if fine_tune_epochs > 0 else None,
            "random_seed": self.config.random_seed,
            "training_history": train_history,
            "final_training_accuracy": float(combined_history["accuracy"][-1]),
            "final_validation_accuracy": float(combined_history["val_accuracy"][-1]),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }

        self._save_metrics(metrics)

        payload = {
            "model_name": self.config.model_name,
            "model_version": self.config.model_version,
            "status": "trained",
            "backend": "keras",
            "current_model_scope": f"{num_classes} CLASS ONLY",
            "metrics": metrics,
            "classes": class_names,
        }
        self.registry.save_metadata(payload)

        # Reset inference singleton so API picks up new model
        try:
            from app.ai.inference_service import MaterialInferenceService
            MaterialInferenceService.reset_instance()
        except ImportError:
            pass

        return payload

    @staticmethod
    def _calibrate_thresholds(
        val_proba: np.ndarray, y_val: np.ndarray,
        target_high_precision: float = 0.90,
        target_medium_precision: float = 0.70,
    ) -> Dict[str, object]:
        """
        Derive confidence tiers from the validation confidence distribution.

        For each candidate threshold t (on max-softmax), compute the precision
        of all validation predictions with confidence >= t. HIGH threshold is
        the smallest t reaching the target precision; MEDIUM likewise at its
        lower target. Below MEDIUM -> manual review. Falls back to the
        best-achieving precision if targets are unreachable.
        """
        confidences = np.max(val_proba, axis=1)
        preds = np.argmax(val_proba, axis=1)
        correct = (preds == np.asarray(y_val)).astype(float)

        grid = np.round(np.arange(0.30, 0.995, 0.01), 3)
        best = {"high": None, "medium": None}
        best_prec = {"high": 0.0, "medium": 0.0}
        for t in grid:
            mask = confidences >= t
            covered = int(mask.sum())
            if covered < 5:
                continue
            precision = float(correct[mask].mean())
            if precision >= target_high_precision and best["high"] is None:
                best["high"] = float(t)
            if precision >= target_medium_precision and best["medium"] is None:
                best["medium"] = float(t)
            for tier in ("high", "medium"):
                # Track best achievable precision as fallback.
                if precision > best_prec[tier]:
                    best_prec[tier] = precision

        high_t = best["high"] if best["high"] is not None else round(min(0.95, max(0.5, best_prec["high"] + 0.05)), 2)
        medium_t = best["medium"]
        if medium_t is None or medium_t > high_t:
            medium_t = round(max(0.4, min(medium_t or high_t - 0.1, high_t)), 2)

        def stats(t):
            mask = confidences >= t
            return {
                "coverage": round(float(mask.mean()), 4),
                "precision_at_threshold": round(float(correct[mask].mean()), 4) if mask.any() else None,
            }

        return {
            "method": "validation cumulative precision on max-softmax",
            "target_high_precision": target_high_precision,
            "target_medium_precision": target_medium_precision,
            "high_confidence_threshold": round(high_t, 2),
            "medium_confidence_threshold": round(float(medium_t), 2),
            "manual_review_threshold": round(float(medium_t), 2),
            "validation_high_tier": stats(high_t),
            "validation_medium_tier": stats(medium_t),
            "validation_overall_accuracy": round(float(correct.mean()), 4),
        }

    def _load_split_data(self, split: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        dataset_path = self.dataset_manager.get_split_path(self.dataset_name, split)
        if not dataset_path.exists():
            return np.array([]), np.array([]), []

        # Skip class folders that contain no images (e.g. an accepted-taxonomy
        # class with zero derived samples). Indices stay consistent across splits
        # because the skip rule is deterministic.
        class_dirs = []
        for d in sorted(d for d in dataset_path.iterdir() if d.is_dir()):
            if any(d.glob("*.jpg")) or any(d.glob("*.jpeg")) or any(d.glob("*.png")):
                class_dirs.append(d)
        class_names = [d.name for d in class_dirs]
        images, labels = [], []

        for class_idx, class_name in enumerate(class_names):
            class_dir = dataset_path / class_name
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for image_file in class_dir.glob(ext):
                    try:
                        images.append(self.preprocessor.preprocess_image(str(image_file)))
                        labels.append(class_idx)
                    except Exception as exc:
                        print(f"Error loading {image_file}: {exc}")

        if not images:
            return np.array([]), np.array([]), class_names

        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32), class_names

    def _label_mapping_note(self) -> str:
        if "ibug" in self.dataset_name.lower():
            return (
                "Manufacturer tag.txt composition metadata (iBUG); "
                f"{len(self._accepted_class_names())} of 10 target materials."
            )
        return "Caption-inferred DeepFashion labels; 3 of 10 target materials."

    def _accepted_class_names(self) -> List[str]:
        dataset_path = self.dataset_manager.get_split_path(self.dataset_name, "test")
        if not dataset_path.exists():
            return []
        names = []
        for d in sorted(p for p in dataset_path.iterdir() if p.is_dir()):
            if any(d.glob("*.jpg")) or any(d.glob("*.jpeg")) or any(d.glob("*.png")):
                names.append(d.name)
        return names

    def _save_metrics(self, metrics: Dict) -> None:
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj

        with open(self.metrics_path, "w", encoding="utf-8") as handle:
            json.dump(convert(metrics), handle, indent=2)
        print(f"Metrics saved to: {self.metrics_path}")

    def load_metrics(self) -> Optional[Dict]:
        if self.metrics_path.exists():
            with open(self.metrics_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the material classifier")
    parser.add_argument("--dataset", default=None, help="Processed dataset name under data/processed/")
    parser.add_argument("--version", default=None, help="Model version tag (default from TrainingConfig)")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = TrainingConfig()
    if args.dataset:
        config.dataset_name = args.dataset
    if args.version:
        config.model_version = args.version
    if args.epochs:
        config.epochs = args.epochs

    pipeline = TrainingPipeline(config=config)
    result = pipeline.train()
    print(json.dumps({
        "status": result.get("status"),
        "classes": result.get("classes"),
        "accuracy": result.get("metrics", {}).get("accuracy"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
