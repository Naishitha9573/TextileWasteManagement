from typing import Dict


class Evaluator:
    def evaluate(self, training_metrics: Dict[str, object]) -> Dict[str, object]:
        return {
            "accuracy": training_metrics.get("metrics", {}).get("accuracy", 0.0),
            "precision": training_metrics.get("metrics", {}).get("precision", 0.0),
            "recall": training_metrics.get("metrics", {}).get("recall", 0.0),
            "f1_score": training_metrics.get("metrics", {}).get("f1_score", 0.0),
            "status": "ready",
        }
