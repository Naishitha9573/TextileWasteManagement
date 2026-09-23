"""Evaluate trained model and write artifacts/metrics/."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.ai.dataset_manager import DatasetManager
from app.ai.datasets.paths import BACKEND_ROOT
from app.ai.evaluate import Evaluator
from app.ai.feature_extractor import load_split_feature_matrix
from app.ai.model_registry import ModelRegistry
from app.ai.training_config import TrainingConfig

from app.ai.model_loader import load_keras_material_model

try:
    import joblib
    import tensorflow  # noqa: F401
    TF = True
except ImportError:
    TF = False
    joblib = None


ARTIFACTS_DIR = BACKEND_ROOT / "artifacts" / "metrics"


def evaluate_current_model() -> dict:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    registry = ModelRegistry()
    metadata = registry.load_metadata()
    if not metadata:
        raise RuntimeError("MODEL NOT TRAINED — no metadata found")

    metrics_saved = metadata.get("metrics", {})
    dataset_name = metrics_saved.get("dataset", TrainingConfig().dataset_name)
    backend = metadata.get("backend") or metrics_saved.get("backend")

    dm = DatasetManager()
    x_test, y_test, class_names = load_split_feature_matrix(
        dm.get_split_path(dataset_name, "test")
    )

    if backend == "sklearn":
        bundle = joblib.load(BACKEND_ROOT / "models" / "material_classifier.joblib")
        model = bundle["model"]
        y_pred = model.predict(x_test)
        y_proba = model.predict_proba(x_test)
    elif backend == "keras" and TF:
        from app.ai.dataset_preprocessor import DatasetPreprocessor
        preprocessor = DatasetPreprocessor()
        model_path = metrics_saved.get("model_path")
        model = load_keras_material_model(model_path)
        paths = []
        split_dir = dm.get_split_path(dataset_name, "test")
        labels = []
        for idx, cname in enumerate(class_names):
            for p in (split_dir / cname).glob("*.jpg"):
                paths.append(str(p))
                labels.append(idx)
            for p in (split_dir / cname).glob("*.png"):
                paths.append(str(p))
                labels.append(idx)
        xs = np.stack([preprocessor.preprocess_image(p) for p in paths])
        y_proba = model.predict(xs, verbose=0)
        y_pred = np.argmax(y_proba, axis=1)
        y_test = np.array(labels)
    else:
        raise RuntimeError(f"Cannot evaluate backend: {backend}")

    evaluator = Evaluator()
    result = evaluator.evaluate(y_test, y_pred, class_names)

    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": metadata.get("model_version"),
        "architecture": metrics_saved.get("architecture"),
        "backend": backend,
        "num_classes": len(class_names),
        "classes": class_names,
        "current_model_scope": "3 CLASS ONLY",
        "test_samples": len(y_test),
        **result,
    }

    for fname, content in [
        ("material_classification_metrics.json", report),
        ("classification_report.json", result.get("classification_report", {})),
        ("confusion_matrix.json", {"labels": class_names, "matrix": result.get("confusion_matrix", [])}),
    ]:
        with open(ARTIFACTS_DIR / fname, "w", encoding="utf-8") as handle:
            json.dump(content, handle, indent=2)

    return report


def render_evaluation_md(report: dict) -> str:
    cr = report.get("classification_report", {})
    per_class = {k: v for k, v in cr.items() if k not in {"accuracy", "macro avg", "weighted avg"}}
    lines = [
        "# Model Evaluation Report",
        "",
        f"**Evaluated:** {report.get('evaluated_at')}",
        f"**Model version:** {report.get('model_version')}",
        f"**Scope:** {report.get('current_model_scope')}",
        f"**Architecture:** {report.get('architecture')}",
        f"**Backend:** {report.get('backend')}",
        "",
        "## Test Set Metrics (Measured)",
        "",
        f"| Metric | Value |",
        f"|--------|------:|",
        f"| Accuracy | {report.get('accuracy', 0):.4f} |",
        f"| Precision (weighted) | {report.get('precision', 0):.4f} |",
        f"| Recall (weighted) | {report.get('recall', 0):.4f} |",
        f"| F1 (weighted) | {report.get('f1_score', 0):.4f} |",
        f"| Macro F1 | {cr.get('macro avg', {}).get('f1-score', 'N/A')} |",
        f"| Test samples | {report.get('test_samples')} |",
        "",
        "## Per-Class",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|-------|----------:|-------:|---:|--------:|",
    ]
    for cls, stats in per_class.items():
        if isinstance(stats, dict):
            lines.append(
                f"| {cls} | {stats.get('precision', 0):.3f} | {stats.get('recall', 0):.3f} | "
                f"{stats.get('f1-score', 0):.3f} | {int(stats.get('support', 0))} |"
            )

    lines.extend([
        "",
        "## Baseline Statement",
        "",
        f"Current MobileNetV3 test accuracy is **{report.get('accuracy', 0):.1%}** (measured on held-out test split). "
        "The prior sklearn feature baseline was approximately **42%**. Both models suffer significant label noise "
        "from caption-derived labels. Treat as **research/prototype baseline** — use confidence-aware / manual-review handling.",
        "",
        "**CURRENT MODEL: 3 CLASS ONLY** — Cotton, Denim, Mixed Fabrics.",
        "",
        "## Confusion Matrix",
        "",
        f"Labels: {report.get('classes')}",
        f"```\n{json.dumps(report.get('confusion_matrix'), indent=2)}\n```",
        "",
        f"*Artifacts: `{ARTIFACTS_DIR}`*",
    ])
    return "\n".join(lines)


def main() -> int:
    report = evaluate_current_model()
    md_path = BACKEND_ROOT.parent / "MODEL_EVALUATION_REPORT.md"
    md_path.write_text(render_evaluation_md(report), encoding="utf-8")
    print(json.dumps({"accuracy": report.get("accuracy"), "report": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
