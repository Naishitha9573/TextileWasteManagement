#!/usr/bin/env python3
"""
ML Pipeline Validation Script

This script validates that the ML pipeline is working correctly with real data.

Usage:
    python test_ml_pipeline.py
    
From Backend directory:
    python -c "import sys; sys.path.insert(0, '.'); exec(open('test_ml_pipeline.py').read())"
"""

import os
import sys
from pathlib import Path

# Add Backend to path if running from root
sys.path.insert(0, str(Path(__file__).parent / "Backend"))

def test_imports():
    """Test that all required dependencies are installed."""
    print("=" * 60)
    print("STEP 1: Testing Dependencies")
    print("=" * 60)
    
    deps = {
        "tensorflow": "TensorFlow (Deep Learning)",
        "sklearn": "Scikit-Learn (Metrics)",
        "numpy": "NumPy (Numerical Computing)",
        "PIL": "Pillow (Image Processing)",
        "fastapi": "FastAPI (Web Framework)",
    }
    
    all_ok = True
    for pkg, desc in deps.items():
        try:
            __import__(pkg)
            print(f"✅ {desc} ({pkg})")
        except ImportError:
            print(f"❌ {desc} ({pkg}) - NOT INSTALLED")
            all_ok = False
    
    print()
    return all_ok


def test_dataset_structure():
    """Test dataset structure."""
    print("=" * 60)
    print("STEP 2: Checking Dataset Structure")
    print("=" * 60)
    
    backend_path = Path(__file__).parent / "Backend"
    dataset_path = backend_path / "datasets" / "fabric_dataset"
    
    print(f"Dataset path: {dataset_path}")
    print(f"Exists: {dataset_path.exists()}")
    print()
    
    if not dataset_path.exists():
        print("❌ Dataset directory not found!")
        return False
    
    # Check for class directories
    classes = [d for d in dataset_path.iterdir() 
              if d.is_dir() and not d.name.startswith('.') 
              and d.name not in ['processed', 'organized', '__pycache__']]
    
    print(f"Found {len(classes)} class directory(ies):")
    
    total_images = 0
    for class_dir in sorted(classes):
        images = [f for f in class_dir.rglob("*") 
                 if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        total_images += len(images)
        
        status = "✅" if len(images) > 0 else "⚠️ "
        print(f"  {status} {class_dir.name}: {len(images)} images")
    
    print()
    print(f"Total images found: {total_images}")
    
    if total_images == 0:
        print("⚠️  WARNING: No training images found!")
        print("   You need to add images to class subdirectories before training.")
        return False
    
    if len(classes) < 2:
        print("⚠️  WARNING: Less than 2 classes found!")
        print("   ML models need at least 2 classes to train.")
        return False
    
    print()
    return True


def test_models_directory():
    """Test models directory structure."""
    print("=" * 60)
    print("STEP 3: Checking Models Directory")
    print("=" * 60)
    
    backend_path = Path(__file__).parent / "Backend"
    models_path = backend_path / "models"
    
    print(f"Models path: {models_path}")
    print(f"Exists: {models_path.exists()}")
    
    if models_path.exists():
        files = list(models_path.glob("*"))
        print(f"Files in models directory: {len(files)}")
        for f in files:
            size = f.stat().st_size / (1024 * 1024)  # MB
            print(f"  - {f.name} ({size:.1f} MB)")
        
        # Check for metrics file
        metrics_path = models_path / "model_metrics.json"
        if metrics_path.exists():
            print("\n✅ Model metrics file found!")
            import json
            try:
                with open(metrics_path) as f:
                    metrics = json.load(f)
                print(f"   Accuracy: {metrics.get('accuracy', 'N/A')}")
                print(f"   Precision: {metrics.get('precision', 'N/A')}")
                print(f"   Recall: {metrics.get('recall', 'N/A')}")
                print(f"   F1 Score: {metrics.get('f1_score', 'N/A')}")
            except Exception as e:
                print(f"   ⚠️  Error reading metrics: {e}")
        else:
            print("\n⚠️  No model metrics found - model hasn't been trained yet")
    else:
        print("⚠️  Models directory doesn't exist - will be created on first training")
    
    print()
    return True


def test_training_pipeline():
    """Test the training pipeline class."""
    print("=" * 60)
    print("STEP 4: Testing Training Pipeline")
    print("=" * 60)
    
    try:
        from app.ai.train import TrainingPipeline, TENSORFLOW_AVAILABLE
        print("✅ TrainingPipeline imported successfully")
        
        if not TENSORFLOW_AVAILABLE:
            print("❌ TensorFlow not available - cannot train")
            return False
        
        print("✅ TensorFlow is available")
        
        # Try to instantiate
        pipeline = TrainingPipeline(dataset_name="fabric_dataset")
        print("✅ TrainingPipeline instantiated")
        
        # Load metrics if available
        metrics = pipeline.load_metrics()
        if metrics:
            print("✅ Metrics loaded successfully")
            print(f"   Model status: {metrics.get('status')}")
        else:
            print("⚠️  No metrics loaded (model not trained yet)")
        
        print()
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_predictor():
    """Test the predictor class."""
    print("=" * 60)
    print("STEP 5: Testing Predictor")
    print("=" * 60)
    
    try:
        from app.ai.predict import Predictor
        print("✅ Predictor imported successfully")
        
        predictor = Predictor()
        print("✅ Predictor instantiated")
        
        if predictor.model is not None:
            print("✅ Model loaded in predictor")
        else:
            print("⚠️  No model loaded (not trained yet)")
        
        print()
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def print_summary(results):
    """Print summary of all tests."""
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    status_emoji = {
        True: "✅",
        False: "❌"
    }
    
    test_names = [
        "Dependencies Installed",
        "Dataset Structure",
        "Models Directory",
        "Training Pipeline",
        "Predictor",
    ]
    
    for name, result in zip(test_names, results):
        print(f"{status_emoji[result]} {name}")
    
    print()
    
    if all(results):
        print("🎉 All checks passed! The ML pipeline is ready.")
    else:
        print("⚠️  Some checks failed. See details above.")
    
    print()
    print("NEXT STEPS:")
    print()
    
    if not results[0]:
        print("1. Install missing dependencies:")
        print("   pip install -r Backend/requirements.txt")
        print()
    
    if not results[1]:
        print("2. Prepare your dataset:")
        print("   - Create subdirectories in Backend/datasets/fabric_dataset/")
        print("   - Name them after fabric types (e.g., Cotton, Polyester)")
        print("   - Add at least 10 images per class")
        print()
    
    if all(results[:2]):
        print("3. Train the model:")
        print("   POST http://localhost:8000/api/train")
        print()
        print("4. Check metrics:")
        print("   GET http://localhost:8000/api/model-metrics")
        print()
        print("5. Make predictions:")
        print("   POST http://localhost:8000/api/predict (with image file)")
        print()


def main():
    """Run all validation tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  ML PIPELINE VALIDATION  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")
    
    results = [
        test_imports(),
        test_dataset_structure(),
        test_models_directory(),
        test_training_pipeline(),
        test_predictor(),
    ]
    
    print_summary(results)


if __name__ == "__main__":
    main()
