# TEXTILE PROJECT — QUICK REFERENCE GUIDE

**Last Updated:** 2026-09-02  
**Active Training:** class_aware_finetune_320_v1 (PID 16400, Epoch 4/25)

---

## QUICK FACTS

| Item | Value |
|------|-------|
| **Primary Dataset** | iBUG (fabric classification) |
| **Training Dataset** | `dataset/processed/fabric_9class/` |
| **Model Type** | EfficientNet-B0 (PyTorch) |
| **Classes** | 9 (Cotton, Polyester, Denim, Wool, Nylon, Viscose, Silk, Fleece, Terrycloth) |
| **Input Size** | 320×320 pixels |
| **Batch Size** | 16 |
| **Max Epochs** | 25 |
| **Optimizer** | AdamW (dual LR: 1e-3 warmup, 1e-4 fine-tune) |
| **Loss** | CrossEntropyLoss (class-weighted, label-smoothing 0.05) |
| **Selection Metric** | Validation Macro F1 |
| **Early Stopping** | patience=5 epochs |
| **Seed** | 42 (deterministic) |
| **Active Checkpoint** | `models/experiments/class_aware_finetune_320_v1/resumed/best_checkpoint.pth` |
| **Champion Baseline** | `models/best_efficientnet_b0_class_aware_finetune_v1.pth` (Macro F1: 86.67%) |

---

## DIRECTORY QUICK REFERENCE

| Directory | Purpose | Status |
|-----------|---------|--------|
| `dataset/processed/fabric_9class/` | Active training data | LOCKED (no modifications) |
| `src/` | Training & evaluation scripts | READ-ONLY while training |
| `models/` | Model checkpoints | PROTECTED |
| `reports/` | Training history & results | Append-only |
| `Backend/` | Production application (LEGACY Keras) | READ-ONLY |
| `Frontend/` | Web UI | READ-ONLY |
| `Backend/data/processed/material_deepfashion*` | DeepFashion subset (separate) | DO NOT TOUCH |

---

## DATASET PATHS

```
dataset/processed/fabric_9class/
├── train/          3,450 images (70%)  ← Training set
├── validation/     740 images (15%)    ← Checkpoint selection
└── test/           740 images (15%)    ← HELD-OUT (evaluation only)
```

**Backup source:**
```
dataset/processed/fabric_16class/
└── all/            ~7000 images (16 classes, source for 9-class)
```

---

## KEY FILES BY FUNCTION

### Training

| File | Function |
|------|----------|
| `src/class_aware_finetune_320_v1.py` | **ACTIVE TRAINING SCRIPT** |
| `src/model.py` | Shared model utilities (CLASS_NAMES, transforms) |

### Dataset Construction (One-time, already complete)

| File | Function |
|------|----------|
| `build_ibug_material_v2.py` | Extract iBUG → 16-class |
| `src/build_fabric_16class.py` | Create 16-class structure |
| `src/build_fabric_9class.py` | Select 9 classes |
| `src/split_dataset.py` | Create train/val/test split |

### Evaluation

| File | Function |
|------|----------|
| `src/evaluate_class_aware_finetune_v1_test.py` | Test evaluator (for v1 baseline) |
| `src/predict.py` | Single-image prediction CLI |

### Legacy/Inactive

| File | Function | Status |
|------|----------|--------|
| `src/train.py` | Baseline training (224×224) | Deprecated |
| `src/class_aware_finetune_v1.py` | Previous fine-tuning (224×224) | Deprecated (but champion) |
| `Backend/app/ai/model_loader.py` | Keras model loading | Obsolete (not EfficientNet) |
| `Backend/app/ai/train.py` | Keras training | Obsolete |

---

## TRAINING COMMANDS

### Start Fresh Training

```bash
cd src
python class_aware_finetune_320_v1.py
```

**Arguments (all have defaults):**
```bash
python class_aware_finetune_320_v1.py \
  --epochs 25 \
  --warmup-epochs 2 \
  --batch-size 16 \
  --learning-rate 1e-3 \
  --fine-tune-learning-rate 1e-4 \
  --label-smoothing 0.05 \
  --patience 5 \
  --num-workers 0 \
  --image-size 320 \
  --seed 42
```

### Resume Training

```bash
cd src
python class_aware_finetune_320_v1.py --resume
```

**Behavior:** 
- Loads checkpoint from `models/experiments/class_aware_finetune_320_v1/best_checkpoint.pth`
- Resumes from best_epoch + 1
- Appends to history.csv
- Output goes to `resumed/` subdirectory

---

## CHECKPOINT LOCATIONS

**Current (active):**
```
models/experiments/class_aware_finetune_320_v1/resumed/best_checkpoint.pth
```

**Initial run:**
```
models/experiments/class_aware_finetune_320_v1/best_checkpoint.pth
```

**Champion baseline (v1, 224×224):**
```
models/best_efficientnet_b0_class_aware_finetune_v1.pth
```

**Baseline (no class weights, 224×224):**
```
models/best_efficientnet_b0_no_weights.pth
```

---

## HISTORY FILES

**Initial run (epochs 1–3):**
```
reports/experiments/class_aware_finetune_320_v1/history.csv
```

**Resumed run (epoch 4+):**
```
reports/experiments/class_aware_finetune_320_v1/resumed/history.csv
```

---

## PREDICTION

### CLI Single Image Prediction

```bash
cd src
python predict.py /path/to/image.jpg
```

**Output:**
```json
{
  "image": "/path/to/image.jpg",
  "predicted_class": "Cotton",
  "confidence": 0.92,
  "top_3": [
    {"class_name": "Cotton", "probability": 0.92},
    {"class_name": "Polyester", "probability": 0.06},
    {"class_name": "Silk", "probability": 0.02}
  ]
}
```

---

## CLASS INDEX REFERENCE

| Index | Class |
|-------|-------|
| 0 | Cotton |
| 1 | Denim |
| 2 | Fleece |
| 3 | Nylon |
| 4 | Polyester |
| 5 | Silk |
| 6 | Terrycloth |
| 7 | Viscose |
| 8 | Wool |

**Source:** `src/model.py` line 22

---

## NORMALIZATION STATS

**ImageNet:**
```python
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
```

**Source:** `src/model.py` lines 23–24

---

## AUGMENTATION (TRAINING ONLY)

**Applied to:** Training set ONLY

**Pipeline:**
1. RandomResizedCrop(320, scale=0.85–1.0)
2. RandomHorizontalFlip(p=0.5)
3. RandomRotation(degrees=8)
4. ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02)
5. ToTensor()
6. Normalize(MEAN, STD)

**Validation/Test:** Deterministic (no augmentation)
- Resize(360) → CenterCrop(320) → ToTensor() → Normalize()

---

## MONITORING TRAINING

### Watch History in Real-Time

```bash
tail -f reports/experiments/class_aware_finetune_320_v1/resumed/history.csv
```

### Check Process

```bash
# Windows PowerShell
Get-Process -Id 16400 | Select-Object Id, ProcessName, StartTime, Responding

# Unix/Linux
ps -p 16400 -o pid,cmd,etime,stime
```

### View Training Output

```bash
tail -f training_output.log
```

---

## COMPARING EXPERIMENTS

**Experiment progression:**

```
Baseline (224×224)
  └─ Accuracy: 85.41%, Macro F1: 83.33%

class_aware_finetune_v1 (224×224) ← CHAMPION
  └─ Accuracy: 88.51%, Macro F1: 86.67%

class_aware_finetune_320×320 (320×320) ← CURRENT (Epoch 4)
  └─ Accuracy: 81.89%, Macro F1: 75.96% (early)
```

**To beat:** Macro F1 > 86.67% (current champion)

---

## VALIDATION SPLIT

**Validation runs after every training epoch:**

1. Forward pass on validation set (740 images)
2. Calculate: Loss, Accuracy, Macro F1, Weighted F1
3. Compare to best Macro F1
4. If improved: save checkpoint, reset early-stop counter
5. If not improved: increment early-stop counter
6. If early-stop counter ≥ 5: STOP training

---

## HELD-OUT TEST EVALUATION

**Test set is NEVER used during training.**

**When:** After training converges (if Macro F1 > baseline)

**How:** Run evaluation script on checkpoint

**Result:** Final metrics reported; model approved or rejected

**Guarantees:**
- No test data leakage
- No overfitting to test set
- Fair final evaluation

---

## DO NOT TOUCH (WHILE TRAINING)

✗ DO NOT:
- Delete/modify `dataset/processed/fabric_9class/`
- Delete/modify `models/experiments/class_aware_finetune_320_v1/resumed/`
- Delete/modify `reports/experiments/class_aware_finetune_320_v1/resumed/`
- Modify `src/model.py` (CLASS_NAMES, transforms)
- Run other training processes
- Delete baseline checkpoints
- Move/rename any dataset folder
- Modify any train/val/test image

✓ SAFE:
- Read history files
- View checkpoint metadata
- Read past experiment results
- Modify Backend/Frontend code
- Modify documentation
- Modify legacy/unused scripts

---

## INTEGRATION STATUS

**Current implementation:** ✓ STANDALONE WORKING

- ✓ Training script works
- ✓ Prediction script works
- ✓ Checkpoint system works
- ✗ **NOT YET integrated** into Backend/app/ai
  - Backend still uses Keras (not PyTorch)
  - Requires adapter/wrapper for .pth checkpoint

---

## DEEPFASHION (SEPARATE — DO NOT TOUCH)

**Location:**
```
Backend/data/processed/material_deepfashion/
Backend/data/processed/material_deepfashion_verified_clean/
```

**Status:** Separate dataset; training unconfirmed

**Rule:** DO NOT MERGE with iBUG splits

**Files:** 
- `adjudicate_conflicting_garment_labels.py`
- `finalize_material_dataset.py`
- `Backend/config/class_mapping.yaml`

---

## CONFIGURATION SUMMARY

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Image Size | 320×320 | Capture finer fabric texture |
| Batch Size | 16 | Memory efficient for GPU |
| Learning Rate | 1e-3 → 1e-4 | Warm up then fine-tune |
| Optimizer | AdamW | Adaptive + L2 regularization |
| Loss | CE + weights + smoothing | Balance classes + prevent overconfidence |
| Augmentation | Moderate (not aggressive) | Robust without overfitting |
| Early Stopping | patience=5 | Prevent overfitting |
| Seed | 42 | Reproducibility |
| Selection Metric | Macro F1 | Fair across imbalanced classes |

---

## VALIDATION METRICS EXPLAINED

| Metric | Range | Purpose | Good Value |
|--------|-------|---------|-----------|
| **Accuracy** | 0–1 | Fraction correct | High (depends on class dist) |
| **Macro F1** | 0–1 | Unweighted per-class F1 | ≥ 0.8 (balanced across classes) |
| **Weighted F1** | 0–1 | Per-class F1, weighted by count | ≥ 0.85 (accounts for imbalance) |
| **Per-class F1** | 0–1 | F1 for single class | ≥ 0.7 for minority classes |

**Why Macro F1 for selection:** Ensures minority class performance (Silk, Terrycloth, Fleece)

---

## WHEN TO STOP TRAINING

Training automatically stops if:
1. Macro F1 doesn't improve for 5 consecutive epochs (early stopping)
2. 25 epochs reached (max)

**Current status (Epoch 4):**
- Stale counter: 0 or 1 (likely improving)
- Estimated time to completion: 15–20 hours (if runs full 25 epochs)

---

## NEXT STEPS AFTER TRAINING

1. **Training converges** → Macro F1 plateaus, early stopping triggers
2. **Run test evaluation** → Create evaluator, evaluate on held-out test
3. **Compare to baseline** → If Macro F1 > 86.67%, candidate is better
4. **Deploy (optional)** → Export to ONNX/Keras, integrate into Backend
5. **Production inference** → User images → Prediction API

---

## SUPPORT FILES

**Full detailed documentation:**
```
PROJECT_COMPLETE_MAP.md  (this file is a summary)
```

**Original experiment reports:**
```
reports/experiments/class_aware_finetune_320_v1/resumed/history.csv
reports/class_aware_finetune_v1_history.csv
reports/targeted_augmentation_v2_history.csv
```

**Dataset reports:**
```
reports/fabric_9class_summary.json
reports/fabric_9class_cleaning.md
reports/data_leakage_check.md
```

---

**Document Version:** 1.0  
**Last Updated:** 2026-09-02  
**Status:** ACTIVE TRAINING ONGOING

