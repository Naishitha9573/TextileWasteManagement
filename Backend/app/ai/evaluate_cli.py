import sys

from app.ai.train import TrainingPipeline


def main() -> int:
    pipeline = TrainingPipeline()
    metrics = pipeline.load_metrics()
    if not metrics:
        print("MODEL NOT TRAINED — no metrics file found.")
        return 1
    print(f"Model: {metrics.get('model_name')} v{metrics.get('model_version', '?')}")
    print(f"Dataset: {metrics.get('dataset')}")
    print(f"Classes: {metrics.get('class_names')}")
    print(f"Test accuracy: {metrics.get('accuracy')}")
    print(f"Precision: {metrics.get('precision')}")
    print(f"Recall: {metrics.get('recall')}")
    print(f"F1: {metrics.get('f1_score')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
