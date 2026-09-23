# DeepFashion Dataset and Garment Analysis

## Audit Result

- Dataset path: `Backend/datasets/deepfashion`
- Detected images: 44,096
- Subsets: Category and Attribute Prediction Benchmark (coarse/fine) and Consumer-to-shop Clothes Retrieval Benchmark
- Category labels observed in image names: 17 garment categories, including `Dresses`, `Pants`, `Shorts`, `Skirts`, `Tees_Tanks`, and `Jackets_Coats`
- Annotation files: 40 detected across extracted files and the two benchmark archives
- Attributes: available
- Bounding boxes: available
- Landmarks/keypoints: available
- Segmentation: available (12,701 extracted images)
- Retrieval annotations: available in the Consumer-to-shop archive
- Captions: available in `captions.json`
- Trained garment model/checkpoint: not detected

## Purpose and Separation

DeepFashion contributes garment category, attribute, localization, segmentation, and retrieval data. It complements the iBUG/material dataset; it does not replace EfficientNet-B0 and its labels are not merged into the nine material classes. EfficientNet-B0 remains responsible for textile material classification, while a future compatible DeepFashion adapter will be responsible for garment analysis.

## Current Integration

The validated upload flow calls the existing material classifier and adds a `garment_analysis` object. Because no compatible garment model is present, it reports `MODEL_NOT_READY`, `category: null`, `attributes: []`, and `confidence: null`. The frontend displays dataset availability and does not fabricate a garment category. No garment statistics are generated without real inference data.

## Limitations and Future Improvement

The project has annotations but no registered garment inference checkpoint, documented garment preprocessing contract, or supported garment prediction classes at runtime. Future work is to train/evaluate a compatible category/attribute/localization model, register its checkpoint and preprocessing, and then enable real garment analytics in the dashboard and reports.