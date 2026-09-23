# Waste Classification Module Setup & Training Guide

## Overview
This document outlines the complete waste classification pipeline for the textile quality assessment system. The waste classifier operates independently from the material classification model to prevent data leakage and maintain model isolation.

## Dataset Schema
**Source:** Hugging Face Dataset `wargoninnovation/clothingdatasetsecondhand`

| Field | Type | Purpose | Notes |
|-------|------|---------|-------|
| `image` | PIL Image (224×224 RGB) | Input feature | Automatic resize & RGB conversion |
| `usage` | ClassLabel (string normalized) | Waste/End-of-Life Target | Contains 10 raw variants, normalized to 6 classes |
| `category` | String | Clothing audience (not used) | Ladies/Children/Unisex/Men — NOT waste target |

### Target Classes (Normalized)
| Class | Raw Variants | Meaning |
|-------|--------------|---------|
| Export | export, Export | Export for resale |
| Reuse | reuse, Reuse | Direct reuse/second-hand |
| Recycle | recycle, Recycle, Rcycle | Textile/material recycling |
| Energy Recovery | Energy recovery | Energy/thermal recovery |
| Remake | Remake | Remake/upcycling |
| Repair | Repair | Repair for resale |

### Dataset Split
- **Original Train:** 30,192 samples
- **Original Test:** 12,940 samples (untouched, not used for training)
- **Stratified Train/Val Split:** 25,663 / 4,529 (85% / 15%)

### Class Distribution (Training Set)
| Class | Count | Percentage |
|-------|-------|-----------|
| Reuse | 12,398 | 48.3% |
| Export | 10,454 | 40.7% |
| Recycle | 2,398 | 9.3% |
| Remake | 167 | 0.7% |
| Energy Recovery | 92 | 0.4% |
| Repair | 154 | 0.6% |

**Note:** Severe class imbalance is present. Training uses balanced class weights:
- Reuse: 0.99x
- Export: 1.22x
- Recycle: 5.36x
- Energy Recovery: 139.28x
- Remake: 77.01x
- Repair: 83.67x

## Pipeline Components

### 1. Dataset Inspection (`src/waste_dataset_inspection.py`)
**Purpose:** Verify dataset schema and identify label column without training.

**Command:**
```bash
py -3 src/waste_dataset_inspection.py
```

**Output:** `models/waste_classification/waste_dataset_inspection.json`
- Confirms `usage` field contains waste/end-of-life labels
- Reports class distribution
- Detects any unexpected label values

**Status:** ✅ COMPLETE — Report already generated

---

### 2. Split Preparation (`src/prepare_waste_train_validation_split.py`)
**Purpose:** Create stratified train/validation split while protecting test set.

**Command:**
```bash
py -3 src/prepare_waste_train_validation_split.py
```

**Output:** `models/waste_classification/waste_train_validation_split.json`
- Stratified 85/15 train/validation split from 30,192 original training samples
- Stores indices for reproducible split
- Validates test set (12,940) remains untouched
- Calculates per-class distribution and balanced weights

**Status:** ✅ COMPLETE — Split manifest already generated

**Safety Checks:**
- ✅ Test indices preserved (12,940 samples)
- ✅ Original split sizes validated (30,192 train, 12,940 test)
- ✅ All classes present in training set
- ✅ No label normalization errors

---

### 3. Model Training (`src/train_waste_classifier.py`)
**Purpose:** Two-stage EfficientNet-B0 training with macro F1 validation.

#### Stage 1: Frozen Backbone
- Freeze EfficientNet-B0 ImageNet-pretrained weights
- Train only: Global Average Pooling → Dense Classification Head
- Dropout: 0.3
- Loss: Sparse Categorical Crossentropy with class weights
- Optimizer: Adam (lr=1e-3)
- Early Stopping: Monitor `val_macro_f1`, patience=4
- Learning Rate Plateau: Reduce on plateau, patience=2
- Epochs: 20 (default) or configurable

#### Stage 2: Upper Layer Fine-tuning
- Unfreeze final ~30 layers of backbone
- Freeze batch normalization layers
- Lower learning rate: Adam (lr=1e-5)
- Continue training with same callbacks
- Epochs: 20 (default) or configurable

#### Image Preprocessing
**Training Augmentation:**
- Resize to 256×256 → Random crop to 224×224
- Random zoom (h/w: ±15%)
- Random translation (h/w: ±10%)
- Random horizontal flip (50%)
- Random brightness (±0.08)
- Random contrast (0.9–1.1)
- Random saturation (0.8–1.2)
- Random rotation (±3°)
- Grayscale → RGB conversion (if needed)

**Validation/Test Augmentation:**
- Deterministic resize to 224×224
- Center crop (implicit in resize)
- No other augmentation

**Normalization:**
- Cast to float32
- Keras EfficientNet handles ImageNet normalization internally (mean/std baked into weights)

#### Model Architecture
```
Input: (None, 224, 224, 3)
    ↓
EfficientNet-B0 (frozen) → Global Average Pooling 2D
    ↓
Dropout(0.3) → Dense(6, activation='softmax')
    ↓
Output: (None, 6) — Softmax probabilities over 6 waste classes
```

#### Command
```bash
# Recommended: 10 epochs for initial validation, 20 for full training
py -3 src/train_waste_classifier.py \
  --label-column usage \
  --verified-end-of-life-label \
  --epochs 20 \
  --batch-size 32
```

**Required Arguments:**
- `--verified-end-of-life-label`: Must be present (acknowledges correct label semantics)
- `--label-column`: Defaults to "usage" (verified waste column)

**Optional Arguments:**
- `--epochs`: Default 20 (set to 10-15 for faster iteration)
- `--batch-size`: Default 32 (reduce to 16 on low-memory systems)

#### Outputs
| File | Purpose |
|------|---------|
| `models/waste_classification/best_waste_model.keras` | Best checkpoint (TF SavedModel format) |
| `models/waste_classification/class_names.json` | Class label ordering ["Export", "Reuse", ...] |
| `models/waste_classification/training_history.json` | Stage 1 & Stage 2 loss/accuracy/F1 curves |
| `models/waste_classification/waste_train_validation_split.json` | Split indices (preserved/reloaded) |
| `models/waste_classification/stage1_best_waste_model.keras` | Stage 1 checkpoint (for debugging) |

#### Metrics Tracked
- **Training:** Accuracy, loss (sparse categorical crossentropy)
- **Validation:** Accuracy, macro F1 score, loss
- **Selection:** Best model based on validation macro F1

**Status:** ✅ READY TO TRAIN

---

### 4. Inference Service (`Backend/app/services/waste_classifier.py`)
**Purpose:** Lazy-loaded singleton model for prediction serving.

**Key Methods:**
```python
waste_classifier = get_waste_classifier()  # Load model once, reuse
prediction = waste_classifier.predict(image)  # PIL Image → dict
```

**Output Format:**
```json
{
  "predicted_category": "Reuse",
  "confidence": 0.945,
  "probabilities": {
    "Export": 0.034,
    "Reuse": 0.945,
    "Recycle": 0.015,
    "Energy Recovery": 0.003,
    "Remake": 0.002,
    "Repair": 0.001
  }
}
```

**Status:** ✅ READY TO SERVE (awaiting trained model)

---

### 5. API Endpoint (`Backend/main.py`)
**Route:** `POST /api/waste/predict`

**Authentication:** Required (Bearer token)

**Input:**
```
Content-Type: multipart/form-data
- file: Image (JPEG, PNG, etc.)
```

**Output:**
```json
{
  "predicted_category": "Reuse",
  "confidence": 0.945,
  "probabilities": {...}
}
```

**Error Handling:**
- `400 Bad Request`: Invalid image format/size
- `401 Unauthorized`: Missing/invalid token
- `503 Service Unavailable`: Model not ready/not found

**Status:** ✅ READY TO SERVE (awaiting trained model)

---

### 6. Frontend UI (`Frontend/src/pages/WasteCategorization.jsx`)
**Purpose:** Separate waste categorization interface.

**Features:**
- Drag-drop or file chooser for image upload
- Real-time prediction display
- Confidence percentage and per-class probability table
- Styling mirrors existing Fabric Classification view

**Navigation:** "Categorize Waste" button added to App.jsx

**Status:** ✅ READY TO SERVE (awaiting trained model)

---

## Execution Workflow

### Phase 1: Inspection & Validation ✅ COMPLETE
1. ✅ Dataset inspection completed
2. ✅ Split manifest generated (train: 25,663 / val: 4,529 / test: 12,940 untouched)
3. ✅ Class distribution verified
4. ✅ Label normalization validated

### Phase 2: Training → IN PROGRESS
1. Run training:
   ```bash
   py -3 src/train_waste_classifier.py --label-column usage --verified-end-of-life-label --epochs 20 --batch-size 32
   ```
2. Monitor outputs:
   - `best_waste_model.keras` saved
   - `training_history.json` logs loss/F1 progression
3. Validate macro F1 on validation set reaches >0.80 (depends on augmentation effectiveness)

### Phase 3: Inference & Testing
1. Backend loads model via `get_waste_classifier()`
2. Test endpoint: `POST /api/waste/predict` with sample images
3. Verify predictions align with expected waste categories
4. Frontend displays predictions in real-time

### Phase 4: Integration
1. Merge frontend/backend changes
2. Deploy waste endpoint alongside material classifier
3. Both models coexist without interference

---

## Critical Safety Constraints

| Constraint | Implementation | Verification |
|-----------|-----------------|--------------|
| **No test leakage** | Test indices (12,940) excluded from train/val split | ✅ Split manifest validates `test_used_for_split: False` |
| **Correct target column** | Hardcoded LABEL_COLUMN = "usage" (not "category") | ✅ Trainer checks `args.label_column == "usage"` |
| **Verified flag** | `--verified-end-of-life-label` required to start training | ✅ Trainer raises SystemExit if flag missing |
| **Original size preservation** | Split preparation validates 30,192 / 12,940 split | ✅ Validation throws if sizes mismatch |
| **No material model interference** | Separate service, separate API endpoint, separate frontend view | ✅ No route overlap, separate singleton instances |
| **Class balance** | Weighted loss and balanced sampling | ✅ Trainer computes and logs class weights |

---

## Troubleshooting

### Issue: "Missing split manifest"
**Solution:** Run `py -3 src/prepare_waste_train_validation_split.py`

### Issue: "Training blocked: test split safety check failed"
**Solution:** Verify split manifest `test_used_for_split: False` and `original_test_split.modified: False`

### Issue: "Unknown label column"
**Solution:** Ensure `LABEL_COLUMN = "usage"` is set; run dataset inspection first

### Issue: Model not loading in inference
**Solution:** Check that `models/waste_classification/best_waste_model.keras` exists after training completes

### Issue: Memory exhaustion during training
**Solution:** Reduce batch size: `--batch-size 16` or epochs: `--epochs 10`

---

## Model Evaluation Metrics

After training completes, review:
- **Macro F1:** Unweighted F1 across all 6 classes (accounts for class imbalance)
- **Per-Class Precision/Recall:** Check minority classes (Energy Recovery, Remake, Repair)
- **Confusion Matrix:** Identify class confusion patterns
- **Training Curves:** Verify convergence without overfitting

Expected baseline:
- Majority classes (Export, Reuse): 0.90+
- Minority classes: 0.40–0.70 (challenging due to imbalance)
- Overall macro F1: 0.65–0.75

---

## Next Steps

1. **Execute Training:**
   ```bash
   cd c:\Users\sama\OneDrive\Desktop\naishtex\Textile
   py -3 src/train_waste_classifier.py --label-column usage --verified-end-of-life-label --epochs 20
   ```

2. **Monitor Progress:**
   - Watch val_macro_f1 during training
   - Early stopping triggers if no improvement for 4 epochs

3. **Validate Model:**
   - Test with sample images via frontend
   - Verify predictions match expected waste categories

4. **Deploy:**
   - Backend & Frontend are ready to serve (no code changes needed)
   - Model will auto-load on first `/api/waste/predict` call

---

## Key Differences from Material Classifier

| Aspect | Material | Waste |
|--------|----------|-------|
| **Dataset** | Internal (25 fabric classes) | Hugging Face (6 waste classes) |
| **Target Column** | fabric_type | usage (end-of-life) |
| **Label Count** | 25 | 6 (normalized from 10 raw) |
| **Class Balance** | Relatively balanced | Severe imbalance (48% Reuse, 0.4% Energy) |
| **Augmentation** | Moderate (flip, rotation) | Aggressive (zoom, translate, rotation, color) |
| **Test Set** | Preserved separately | Preserved in split manifest |
| **API Endpoint** | /api/predict | /api/waste/predict |
| **Frontend View** | Classify Fabric | Categorize Waste |
| **Inference Service** | FabricClassifier | WasteClassifier |

---

## Files Summary

### Python Modules
- `src/waste_dataset_inspection.py` — Dataset validation
- `src/prepare_waste_train_validation_split.py` — Split creation
- `src/train_waste_classifier.py` — Training pipeline
- `Backend/app/services/waste_classifier.py` — Inference service

### Frontend
- `Frontend/src/pages/WasteCategorization.jsx` — UI component
- `Frontend/src/App.jsx` — Navigation integration

### Artifacts
- `models/waste_classification/waste_dataset_inspection.json` — Dataset report
- `models/waste_classification/waste_train_validation_split.json` — Split manifest
- `models/waste_classification/best_waste_model.keras` — Trained model (to be created)
- `models/waste_classification/class_names.json` — Class ordering
- `models/waste_classification/training_history.json` — Training logs

---

**Last Updated:** 2026-08-26  
**Status:** Ready for training execution  
**Estimated Training Time:** 4–8 hours (20 epochs, 32 batch size, ~43k samples across stages)
