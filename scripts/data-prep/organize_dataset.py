#!/usr/bin/env python3
"""
Dataset Organization Script for Textile Fabric Classification

This script helps organize the existing fabric dataset into the proper
class-based directory structure required by the ML pipeline.

Usage:
    python organize_dataset.py --source /path/to/dataset --target /path/to/organized
    
Or from the Backend directory:
    python ../organize_dataset.py
"""

import os
import shutil
from pathlib import Path
import argparse


def organize_defect_dataset():
    """Organize the existing defect classification dataset."""
    dataset_path = Path(__file__).parent / "Backend" / "datasets" / "fabric_dataset"
    
    print("=" * 60)
    print("FABRIC DATASET ORGANIZATION SCRIPT")
    print("=" * 60)
    
    defect_path = dataset_path / "Defect_images"
    nodefect_path = dataset_path / "NODefect_images"
    
    if not defect_path.exists() or not nodefect_path.exists():
        print("❌ Error: Defect_images and NODefect_images directories not found")
        print(f"   Expected location: {dataset_path}")
        return False
    
    # Create organized structure
    organized_path = dataset_path / "organized"
    
    # Create binary classification structure
    defective_dir = organized_path / "Defective"
    non_defective_dir = organized_path / "Non_Defective"
    
    defective_dir.mkdir(parents=True, exist_ok=True)
    non_defective_dir.mkdir(parents=True, exist_ok=True)
    
    # Move defect images
    defect_count = 0
    for img_file in defect_path.rglob("*"):
        if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            try:
                dest = defective_dir / img_file.name
                if not dest.exists():
                    shutil.copy2(img_file, dest)
                    defect_count += 1
            except Exception as e:
                print(f"⚠️  Error copying {img_file.name}: {e}")
    
    # Move non-defect images
    non_defect_count = 0
    for img_file in nodefect_path.rglob("*"):
        if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            try:
                dest = non_defective_dir / img_file.name
                if not dest.exists():
                    shutil.copy2(img_file, dest)
                    non_defect_count += 1
            except Exception as e:
                print(f"⚠️  Error copying {img_file.name}: {e}")
    
    print()
    print("✅ Dataset organized successfully!")
    print(f"   Defective images: {defect_count}")
    print(f"   Non-defective images: {non_defect_count}")
    print()
    print(f"📁 Organized dataset location:")
    print(f"   {organized_path}")
    print()
    print("To use this organized dataset for training, update:")
    print(f"  pipeline = TrainingPipeline(dataset_name='fabric_dataset/organized')")
    print()
    
    return True


def show_dataset_structure():
    """Display the current dataset structure."""
    dataset_path = Path(__file__).parent / "Backend" / "datasets" / "fabric_dataset"
    
    print("\n📊 Current Dataset Structure:")
    print("=" * 60)
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found at: {dataset_path}")
        return
    
    for item in sorted(dataset_path.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            img_count = sum(1 for _ in item.rglob("*") 
                          if _.is_file() and _.suffix.lower() in ['.jpg', '.jpeg', '.png'])
            print(f"  📁 {item.name}/ ({img_count} images)")
            
            # Show sample images (first 3)
            samples = [f.name for f in sorted(item.iterdir()) 
                      if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']][:3]
            for sample in samples:
                print(f"     - {sample}")
            if len(samples) == 0:
                print(f"     (no images found)")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Organize textile fabric dataset for ML training"
    )
    parser.add_argument(
        "--organize-defect",
        action="store_true",
        help="Organize existing defect classification dataset"
    )
    parser.add_argument(
        "--show-structure",
        action="store_true",
        help="Show current dataset structure"
    )
    
    args = parser.parse_args()
    
    if not args.organize_defect and not args.show_structure:
        args.show_structure = True  # Default action
    
    if args.show_structure:
        show_dataset_structure()
    
    if args.organize_defect:
        organize_defect_dataset()


if __name__ == "__main__":
    main()
