# DeepFashion Dataset & Garment Analysis

- **Purpose:** complementary garment/fashion analysis
- **Dataset/subset:** 44,096 images; Category and Attribute Prediction Benchmark plus Consumer-to-shop Clothes Retrieval Benchmark
- **Available annotations:** categories, attributes, bounding boxes, landmarks, segmentation, captions, and retrieval annotations
- **Model status:** dataset available; compatible trained garment model/checkpoint not available; inference `MODEL NOT READY`
- **Workflow:** upload -> validation -> optional garment adapter -> unified waste/sustainability recommendation
- **Relationship:** `iBUG -> EfficientNet-B0 -> Material Classification`; `DeepFashion -> Garment Analysis`
- **Boundary:** DeepFashion garment labels remain separate from the nine EfficientNet-B0 material classes