# TEXTILE FABRIC CLASSIFICATION PROJECT — COMPLETE SYSTEM MAP

**Generated:** 2026-09-02  
**Status:** Active training in progress (PID 16400)  
**Experiment:** class_aware_finetune_320_v1  
**Model:** EfficientNet-B0 (320×320 input)  
**Current Epoch:** 4 (of 25 max)

---

## PART 1: CURRENT TRAINING STATUS

### ✓ Process Status

| Item | Status |
|------|--------|
| **PID** | 16400 |
| **Process** | python |
| **Running** | YES ✓ |
| **Start Time** | 2026-09-02 21:57:45 |
| **Responding** | YES ✓ |

### Training Checkpoint

| Item | Path |
|------|------|
| **Script** | `src/class_aware_finetune_320_v1.py` |
| **Model Checkpoint** | `models/experiments/class_aware_finetune_320_v1/resumed/best_checkpoint.pth` |
| **History (Original)** | `reports/experiments/class_aware_finetune_320_v1/history.csv` |
| **History (Resumed)** | `reports/experiments/class_aware_finetune_320_v1/resumed/history.csv` |
| **Current Epoch** | 4 |
| **Maximum Epochs** | 25 |
| **Warmup Epochs** | 2 |

### Latest Training Metrics (Epoch 4 — Resumed)

```
Epoch:                  4
Train Loss:             0.8853
Train Accuracy:         0.8169
Validation Loss:        0.9919
Validation Accuracy:    0.8189
Validation Macro F1:    0.7596
Validation Weighted F1: 0.8239
Learning Rate:          0.0009953895432879838
Elapsed Time:           725.78 seconds (12.1 minutes)
```

---

## PART 2: COMPLETE PROJECT STRUCTURE

### Root Directory Organization

```
Textile/
├── dataset/                          # Core dataset storage
│   └── processed/
│       ├── fabric_9class/            # Active 9-class training dataset
│       │   ├── all/                  # Unified class folders
│       │   ├── train/                # Training split (70%)
│       │   ├── validation/           # Validation split (15%)
│       │   └── test/                 # Test split (15%)
│       └── fabric_16class/           # Source 16-class (read-only backup)
│           └── all/
│
├── src/                              # Active training scripts
│   ├── class_aware_finetune_320_v1.py ◄─ CURRENTLY RUNNING
│   ├── class_aware_finetune_v1.py
│   ├── targeted_augmentation_v2.py
│   ├── model.py                      # Shared utilities (CLASS_NAMES, transforms)
│   ├── build_fabric_9class.py        # Dataset construction
│   ├── build_fabric_16class.py
│   ├── split_dataset.py
│   ├── evaluate_class_aware_finetune_v1_test.py
│   ├── evaluate_targeted_augmentation_v2.py
│   ├── predict.py                    # Single-image prediction
│   └── [legacy training variants]
│
├── models/                           # Trained model checkpoints
│   ├── best_efficientnet_b0.pth      # Baseline 224×224
│   ├── best_efficientnet_b0_class_aware_finetune_v1.pth  # Champion
│   ├── best_efficientnet_b0_targeted_augmentation_v2.pth
│   ├── class_names.json
│   ├── training_config.json
│   └── experiments/
│       ├── class_aware_finetune_320_v1/
│       │   ├── best_checkpoint.pth
│       │   └── resumed/
│       │       └── best_checkpoint.pth  ◄─ ACTIVE CHECKPOINT
│       └── [legacy experiments]
│
├── reports/                          # Training history & evaluation
│   ├── experiments/
│   │   └── class_aware_finetune_320_v1/
│   │       ├── history.csv
│   │       └── resumed/
│   │           ├── history.csv        ◄─ ACTIVE HISTORY
│   │           └── validation_confusion_matrix.png
│   ├── fabric_9class_*.json/csv       # Dataset reports
│   ├── class_aware_finetune_v1_history.csv
│   ├── targeted_augmentation_v2_history.csv
│   ├── data_leakage_check.md
│   └── [evaluation results]
│
├── Backend/                          # Production application
│   ├── app/
│   │   ├── ai/                       # Inference & model loading
│   │   │   ├── model_loader.py       # Keras model (LEGACY)
│   │   │   ├── inference_service.py  # Keras/sklearn service
│   │   │   ├── model_registry.py     # Model metadata
│   │   │   ├── predict.py
│   │   │   ├── dataset_manager.py
│   │   │   ├── dataset_preprocessor.py
│   │   │   ├── evaluate.py
│   │   │   ├── evaluate_cli.py
│   │   │   ├── feature_extractor.py
│   │   │   ├── confidence.py
│   │   │   └── [other ML utilities]
│   │   ├── api/                      # REST endpoints
│   │   ├── services/                 # Business logic
│   │   └── repositories/
│   ├── main.py
│   ├── datasets_router.py
│   ├── material_classes.py           # MODEL_CLASSES definition
│   ├── requirements.txt
│   ├── Dockerfile
│   └── data/                         # Data storage
│       ├── manifests/
│       └── processed/
│           ├── material_deepfashion/ [DEEPFASHION — DO NOT TOUCH]
│           └── material_deepfashion_verified_clean/ [DEEPFASHION — DO NOT TOUCH]
│
├── Frontend/                         # Web UI
│   ├── src/
│   ├── public/
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── Dockerfile
│
├── tests/
├── docs/
└── .env, .gitignore, .git, package.json, README.md
```

---

## PART 3: IBUG DATASET PIPELINE

### 3.1 Original IBUG Dataset Location

**CONFIRMED:** iBUG dataset is the **primary source**.

Original iBUG files are NOT in this repository. They were processed into:
- `Backend/data/processed/material_deepfashion/` (caption-extracted subset — separate from iBUG)
- Extracted fabric labels stored in dataset manifests

### 3.2 Dataset Construction Pipeline

| Step | Script | Input | Output | Purpose |
|------|--------|-------|--------|---------|
| 1 | `build_ibug_material_v2.py` | iBUG raw | 16-class dataset | Extract iBUG fabric images, create 16 classes, resolve duplicates |
| 2 | `build_fabric_16class.py` | iBUG extracted | `fabric_16class/all/` | Create standardized 16-class structure |
| 3 | `build_fabric_9class.py` | `fabric_16class/all/` | `fabric_9class/all/` | Select 9 strongest classes, verify integrity |
| 4 | `split_dataset.py` | `fabric_9class/all/` | `fabric_9class/{train,val,test}/` | Stratified split with leakage checks |

### 3.3 9-Class Definition

**CONFIRMED:** Exactly 9 classes (sorted alphabetically)

```python
CLASS_NAMES = sorted([
    "Cotton",      # Index 0
    "Polyester",   # Index 1
    "Denim",       # Index 2
    "Wool",        # Index 3
    "Nylon",       # Index 4
    "Viscose",     # Index 5
    "Silk",        # Index 6
    "Fleece",      # Index 7
    "Terrycloth"   # Index 8
])
```

**Source:** `src/model.py` line 22

### 3.4 Class-to-Label Mapping

**CONFIRMED:** Automatically generated from folder structure in `fabric_9class/`

```python
expected_mapping = {
    "Cotton": 0,
    "Denim": 1,
    "Fleece": 2,
    "Nylon": 3,
    "Polyester": 4,
    "Silk": 5,
    "Terrycloth": 6,
    "Viscose": 7,
    "Wool": 8
}
```

**Verification:** Each training script verifies: `train_dataset.class_to_idx == expected_mapping`

### 3.5 Train/Validation/Test Split

**CONFIRMED:** Stratified split by sample ID, NO data leakage

| Split | Ratio | Count | Purpose |
|-------|-------|-------|---------|
| **Train** | 70% | ~3,450 images | Model training |
| **Validation** | 15% | ~740 images | Checkpoint selection (Macro F1) |
| **Test** | 15% | ~740 images | **Held-out** final evaluation (not used during training) |

**Source:** `src/split_dataset.py` lines 17-18  
**Ratios:** `RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}`

### 3.6 Split Ratios & Method

- **Seed:** 42 (deterministic)
- **Method:** Stratified by sample ID (groups all views of same garment together)
- **Stratification:** Per-class distribution balanced across splits
- **Prevents:** Sample leakage (same garment not split across train/val/test)

### 3.7 Duplicate Detection

**CONFIRMED:** SHA-256 hash comparison

- Every image compared with source using `hashlib.sha256()`
- Copied file hash matches source file hash
- Cross-split duplicates detected and rejected
- **Result:** NO duplicate hashes across splits

**Source:** `src/split_dataset.py` lines 115-126

### 3.8 Data Leakage Checks

**CONFIRMED:** Multiple leakage checks performed

```python
checks = {
    "duplicate_across_splits": not duplicate_hashes,           # ✓ PASS
    "sample_groups_across_splits": not leaked_groups,         # ✓ PASS
    "test_images_separate": not (test ∩ train),              # ✓ PASS
    "test_labels_not_used_for_training": True,               # ✓ PASS
    "test_metrics_not_used_for_checkpoint_selection": True,  # ✓ PASS
}
```

**Result:** ALL CHECKS PASS

### 3.9 Final Dataset Directory

**CONFIRMED:** Active dataset location

```
dataset/processed/fabric_9class/
├── train/        ← 70% images, training only
├── validation/   ← 15% images, checkpoint selection
└── test/         ← 15% images, held-out evaluation
```

**Path in code:** `src/model.py` line 20  
`DATASET_ROOT = ROOT / "dataset" / "processed" / "fabric_9class"`

---

## PART 4: CURRENT TRAINING PIPELINE (class_aware_finetune_320_v1)

### 4.1 Dataset Loading

```
Source:        fabric_9class/{train,validation,test}
Loader:        torch.utils.data.DataLoader
Load:          src/class_aware_finetune_320_v1.py (lines 182-183)
Train Dataset: datasets.ImageFolder(DATASET_ROOT / "train", transform=train_transform)
Val Dataset:   datasets.ImageFolder(DATASET_ROOT / "validation", transform=validation_transform)
```

### 4.2 Dataset Path

**CONFIRMED:** `dataset/processed/fabric_9class`

**Source:** `src/model.py` line 20

### 4.3 Image Loading

**Library:** torchvision.datasets.ImageFolder

**Process:**
1. Read image files from class folders
2. Verify file is readable (PIL.Image.open)
3. Convert to RGB (if needed)
4. Apply transforms
5. Convert to tensor
6. Normalize

**Source:** `src/class_aware_finetune_320_v1.py` (lines 182-183)

### 4.4 Image Resizing

| Component | Value | Source |
|-----------|-------|--------|
| **Input Resolution** | 320×320 | Argument parser (line 50: `--image-size 320`) |
| **Resize Method** | RandomResizedCrop | Augmentation pipeline |
| **Crop Scale** | 0.85–1.0 | `build_train_transform()` line 135 |

**Exact:** `transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0))`

### 4.5 Input Resolution

**CONFIRMED:** 320×320 pixels (model-specific)

**Requirement:** This experiment enforces 320×320 only:  
`if args.image_size != 320: raise SystemExit("This dedicated experiment requires --image-size 320")`

**Source:** `src/class_aware_finetune_320_v1.py` line 170

### 4.6 Normalization

**ImageNet statistics (hardcoded):**

```python
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
```

**Applied via:** `transforms.Normalize(MEAN, STD)`

**Source:** `src/model.py` lines 23-24

### 4.7 Augmentation (Training Only)

**Applied to:** Training set ONLY

**Disabled for:** Validation & test sets (deterministic transforms only)

**Training augmentation pipeline:**

```python
transforms.Compose([
    transforms.RandomResizedCrop(320, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=8),
    transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
```

**Breakdown:**
| Augmentation | Probability/Parameter | Purpose |
|------------------|----------------------|---------|
| RandomResizedCrop | scale=(0.85, 1.0) | Vary crop region |
| RandomHorizontalFlip | p=0.5 | Mirror 50% of images |
| RandomRotation | degrees=8 | Rotate ±8° |
| ColorJitter | brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02 | Color variation |

**Source:** `src/class_aware_finetune_320_v1.py` (lines 134-141)

**Validation/test transforms (deterministic):**

```python
transforms.Compose([
    transforms.Resize(round(320 * 256 / 224)),
    transforms.CenterCrop(320),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
```

### 4.8 Batch Size

**CONFIRMED:** 16 (default)

**Source:** `src/class_aware_finetune_320_v1.py` line 49  
`parser.add_argument("--batch-size", type=int, default=16)`

### 4.9 Number of Classes

**CONFIRMED:** 9 (hardcoded)

**Validation:**
```python
if len(CLASS_NAMES) != 9:
    raise RuntimeError("Expected 9 classes")
```

**Source:** `src/class_aware_finetune_320_v1.py` (line 195)

### 4.10 Class-Aware Logic

**Implementation:** Class weighting via inverse frequency

```python
train_counts = np.bincount(train_dataset.targets, minlength=9)
inverse_frequency = train_counts.sum() / (9 * train_counts)
class_weights = np.sqrt(inverse_frequency)
class_weights /= class_weights.mean()  # Normalize
```

**Purpose:** Balance gradient magnitudes across imbalanced classes

**Applied in:** CrossEntropyLoss  
`loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)`

**Source:** `src/class_aware_finetune_320_v1.py` (lines 207-210, 221-222)

### 4.11 Model Architecture

**Type:** EfficientNet-B0

**Base:** torchvision.models.efficientnet_b0

**Pretrained:** YES (ImageNet weights)

**Source:** `src/model.py` (lines 104-109)

```python
def create_model(num_classes: int = 9, pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    network = models.efficientnet_b0(weights=weights)
    input_features = network.classifier[1].in_features
    network.classifier[1] = nn.Linear(input_features, num_classes)  # Replace head
    return network
```

### 4.12 Pretrained Weights

**CONFIRMED:** ImageNet pretrained

**Load:** `models.EfficientNet_B0_Weights.DEFAULT`

**Source:** `src/model.py` line 106

### 4.13 Frozen Layers

**Warmup phase (epochs 1–2):**
- Feature extractor FROZEN
- Classifier head TRAINED
- Learning rate: 1e-3 (high)

**Code:**
```python
for parameter in model.features.parameters():
    parameter.requires_grad = False
```

**Source:** `src/class_aware_finetune_320_v1.py` (line 218)

### 4.14 Fine-Tuned Layers

**After warmup (epoch 3+):**
- Last 3 blocks of features UNFROZEN
- All of classifier TRAINED
- Two LR groups:
  - Classifier: 1e-3
  - Features[-3:]: 1e-4 (lower)

**Code:**
```python
for parameter in model.features[-3:].parameters():
    parameter.requires_grad = True
```

**Source:** `src/class_aware_finetune_320_v1.py` (line 233)

### 4.15 Classification Head

**Original:** Linear(1280 → 1000) [ImageNet]

**Replaced with:** Linear(1280 → 9) [Fabric classes]

**Source:** `src/model.py` (line 108)

### 4.16 Loss Function

**Type:** Cross-Entropy with class weights and label smoothing

```python
loss_fn = nn.CrossEntropyLoss(
    weight=torch.tensor(class_weights),
    label_smoothing=0.05
)
```

**Components:**
- **Class weights:** Inverse frequency normalization
- **Label smoothing:** 0.05 (prevents overconfidence)

**Source:** `src/class_aware_finetune_320_v1.py` (line 221-222)

### 4.17 Class Weighting

**Formula:**
```
inverse_frequency = total_samples / (num_classes * per_class_count)
class_weights = sqrt(inverse_frequency)
class_weights /= class_weights.mean()  # Normalize
```

**Purpose:** Penalize misclassification of underrepresented classes more

**Source:** `src/class_aware_finetune_320_v1.py` (lines 207-210)

### 4.18 Label Smoothing

**Value:** 0.05

**Effect:** Soft targets instead of one-hot; prevents extreme confidence

**Source:** `src/class_aware_finetune_320_v1.py` (line 222)

### 4.19 Optimizer

**Type:** AdamW (Adam with weight decay)

**LR group 1 (warmup):**
- Params: All trainable
- LR: 1e-3 (high)

**LR group 2 (post-warmup):**
- Params: Classifier (always) + features[-3:] (post-warmup)
- LR: 1e-4 (low)

**Weight decay:** 1e-4 (L2 regularization)

**Source:** `src/class_aware_finetune_320_v1.py` (lines 219-222, 233-235)

### 4.20 Learning Rate (Warmup)

**Warmup LR:** 1e-3

**Duration:** 2 epochs (first 2)

**Purpose:** Stabilize classifier training before fine-tuning features

**Source:** `src/class_aware_finetune_320_v1.py` (lines 48-49, 219)

### 4.21 Learning Rate Scheduler

**Type:** CosineAnnealingLR

**Configuration:**
- **T_max:** `max(1, epochs - warmup_epochs)` = 23 (for 25 epochs)
- **eta_min:** Fine-tune LR / 10 = 1e-5 (after warmup)
- **Behavior:** Cosine decay from current LR to eta_min

**Source:** `src/class_aware_finetune_320_v1.py` (line 225)

### 4.22 Maximum Epochs

**CONFIRMED:** 25 epochs (default)

**Argument:** `parser.add_argument("--epochs", type=int, default=25)`

**Source:** `src/class_aware_finetune_320_v1.py` (line 48)

### 4.23 Early Stopping

**Metric:** Validation Macro F1

**Patience:** 5 epochs without improvement

**Behavior:** Stop training if no F1 increase for 5 consecutive epochs

**Code:**
```python
if validation_macro_f1 > best["macro_f1"]:
    best = {...}
    stale = 0
else:
    stale += 1
    if stale >= args.patience:
        print(f"Early stopping at epoch {epoch}; patience reached.")
        break
```

**Source:** `src/class_aware_finetune_320_v1.py` (lines 291-298)

### 4.24 Checkpoint Saving

**When:** When validation Macro F1 improves

**Path:** 
- Initial: `models/experiments/class_aware_finetune_320_v1/best_checkpoint.pth`
- Resumed: `models/experiments/class_aware_finetune_320_v1/resumed/best_checkpoint.pth`

**Saved state:**
```python
torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": CLASS_NAMES,
    "experiment_name": EXPERIMENT_NAME,
    "best_validation_macro_f1": validation_macro_f1,
    "best_validation_accuracy": validation_accuracy,
    "best_validation_weighted_f1": validation_weighted_f1,
    "best_epoch": epoch
}, CHECKPOINT_PATH)
```

**Source:** `src/class_aware_finetune_320_v1.py` (lines 289-297)

### 4.25 Resume Functionality

**Flag:** `--resume`

**Behavior:**
1. Load saved checkpoint
2. Resume from next epoch
3. Unfreeze last 3 feature blocks (aggressive fine-tuning)
4. Reconstruct optimizers with dual learning rates
5. Append to existing history.csv

**Code:**
```python
if args.resume:
    checkpoint = torch.load(resume_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    start_epoch = int(checkpoint.get("best_epoch", 0)) + 1
    # Unfreeze features[-3:]
    for parameter in model.features[-3:].parameters():
        parameter.requires_grad = True
```

**Source:** `src/class_aware_finetune_320_v1.py` (lines 227-235)

### 4.26 Validation Process

**When:** After every training epoch

**Dataset:** Validation split (15%, ~740 images)

**Metrics calculated:**
1. Validation loss
2. Validation accuracy
3. Validation Macro F1 (per-class F1 average)
4. Validation Weighted F1 (per-class F1, weighted by support)
5. Confusion matrix

**Source:**
```python
validation_pass(model, validation_loader, loss_fn, device)
→ src/class_aware_finetune_320_v1.py (lines 144-160)
```

### 4.27 Macro F1 Calculation

**Formula:** Average F1 score across all classes (unweighted)

```python
precision, recall, f1_values, support = precision_recall_fscore_support(
    actual, predicted, labels=range(9), zero_division=0
)
macro_f1 = float(f1_values.mean())
```

**Purpose:** Balanced metric (doesn't favor majority class)

**Source:** `src/class_aware_finetune_320_v1.py` (lines 156-157)

### 4.28 Weighted F1 Calculation

**Formula:** Per-class F1, weighted by class support

```python
weighted_f1 = float(np.average(f1_values, weights=support))
```

**Purpose:** Reflects actual class distribution in dataset

**Source:** `src/class_aware_finetune_320_v1.py` (line 157)

### 4.29 Model Selection Criterion

**Primary Metric:** Validation Macro F1 (maximized)

**Secondary Metrics:** Accuracy, Weighted F1 (for reference)

**Selection Logic:**
```python
if validation_macro_f1 > best["macro_f1"]:
    save_checkpoint()
    stale = 0
else:
    stale += 1
```

**Rationale:** Macro F1 handles class imbalance; accounts for minority class performance

**Source:** `src/class_aware_finetune_320_v1.py` (lines 289-298)

---

## PART 5: MODEL ARCHITECTURE

### 5.1 EfficientNet-B0 Architecture Flow

```
Input Image (320×320)
         ↓
Resize & Preprocessing
(CenterCrop 320, Normalize)
         ↓
EfficientNet-B0 Features
┌─────────────────────────────────────────────────────────┐
│ • Stem (7×7 conv, 32 channels)                          │
│ • MBConv blocks 1–7 (various depths, ratios, dilations) │
│ • Progressive reduction 320→160→80→40→20→10→5          │
│ • Final conv (1280 channels)                            │
│ • Spatial dimensions: 5×5 at end of features            │
└─────────────────────────────────────────────────────────┘
         ↓
Global Average Pooling (5×5 → 1×1)
→ 1280-d vector
         ↓
Classification Head
Linear(1280 → 9)
         ↓
Softmax Activation
         ↓
9 Fabric Class Probabilities
         ↓
argmax → Predicted Class
```

### 5.2 Exact Code Responsible for Each Stage

| Stage | File | Function/Line | Code |
|-------|------|---------------|------|
| Input image | `src/class_aware_finetune_320_v1.py` | lines 182-183 | `ImageFolder(..., transform=train_transform)` |
| Resize | `src/class_aware_finetune_320_v1.py` | line 135 | `RandomResizedCrop(320, scale=(0.85, 1.0))` |
| Preprocessing | `src/class_aware_finetune_320_v1.py` | lines 138-140 | `ToTensor(), Normalize(MEAN, STD)` |
| EfficientNet-B0 | `src/model.py` | lines 104-109 | `models.efficientnet_b0(weights=weights)` |
| Pooling | torchvision source | – | Global average pool (built-in EfficientNet) |
| Classification head | `src/model.py` | line 108 | `Linear(1280, 9)` |
| 9 fabric classes | `src/model.py` | line 22 | `CLASS_NAMES = sorted([...])` |
| Probabilities | `src/class_aware_finetune_320_v1.py` | line 283 | `torch.softmax(logits, dim=1)` |
| Predicted class | `src/class_aware_finetune_320_v1.py` | line 283 | `logits.argmax(1)` |

### 5.3 Architecture Summary

| Component | Value |
|-----------|-------|
| **Model Type** | EfficientNet-B0 |
| **Pretrained** | YES (ImageNet) |
| **Input Channels** | 3 (RGB) |
| **Input Size** | 320×320 |
| **Feature Channels** | 1280 (final) |
| **Output Classes** | 9 |
| **Activation** | ReLU (features), Softmax (output) |
| **Dropout** | Built-in (MBConv blocks) |
| **Batch Normalization** | Built-in (MBConv blocks) |

### 5.4 Frozen & Unfrozen Layers

**Warmup (epochs 1–2):**
- Frozen: `model.features` (all blocks 0–N)
- Trainable: `model.classifier` only

**Post-warmup (epochs 3+):**
- Frozen: `model.features[:-3]` (first N-3 blocks)
- Trainable: `model.features[-3:]` (last 3 blocks) + `model.classifier`

**Rationale:** Gradual unfreezing prevents catastrophic forgetting

**Source:** `src/class_aware_finetune_320_v1.py` (lines 218, 233)

---

## PART 6: AUGMENTATION (TRAINING ONLY)

### 6.1 Active Augmentations in class_aware_finetune_320_v1

**Applied to:** Training set ONLY

**Disabled for:** Validation & test (deterministic transforms only)

| Augmentation | Parameter | Probability | Purpose |
|--------------|-----------|-------------|---------|
| RandomResizedCrop | scale=(0.85, 1.0) | 100% | Vary crop region, maintain 320×320 |
| RandomHorizontalFlip | p=0.5 | 50% | Mirror images (garments symmetric) |
| RandomRotation | degrees=8 | 100% | Rotate ±8 degrees |
| ColorJitter | brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02 | 100% | Lighting/color variation |

### 6.2 Augmentation Code

**Source:** `src/class_aware_finetune_320_v1.py` (lines 134-141)

```python
def build_train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
```

### 6.3 Validation/Test Augmentation

**Deterministic only (no randomness):**

```python
deterministic = transforms.Compose([
    transforms.Resize(round(320 * 256 / 224)),  # Resize to 360×360
    transforms.CenterCrop(320),                  # Center crop to 320×320
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
```

**Purpose:** Reproducible evaluation

**Source:** `src/model.py` (lines 54-58)

### 6.4 Confirmation

✓ Augmentation applied to **training only**  
✓ Validation/test are **NOT augmented**  

---

## PART 7: TRAINING CONFIGURATION

### 7.1 Hyperparameters Summary

| Parameter | Value | Source |
|-----------|-------|--------|
| **Max epochs** | 25 | Default argument |
| **Warmup epochs** | 2 | Default argument |
| **Batch size** | 16 | Default argument |
| **Learning rate (warmup)** | 1e-3 | Default argument |
| **Learning rate (fine-tune)** | 1e-4 | Default argument |
| **Label smoothing** | 0.05 | Default argument |
| **Early stopping patience** | 5 | Default argument |
| **Random seed** | 42 | Default argument |
| **Optimizer** | AdamW | Code line 220 |
| **Weight decay** | 1e-4 | Code line 220 |
| **Scheduler** | CosineAnnealingLR | Code line 225 |
| **Loss function** | CrossEntropyLoss | Code line 221 |
| **Class weighting** | sqrt(inverse frequency) | Code lines 207–210 |

### 7.2 Configuration File

**Saved config:** `models/experiments/class_aware_finetune_320_v1/resumed/final.json`

**Contains:**
- Experiment name
- Best epoch, accuracy, Macro F1, Weighted F1
- Checkpoint path
- Resumed from status
- Test set used: False
- Training configuration (args)
- Class weights
- Safety checks

---

## PART 8: EXPERIMENT HISTORY

### 8.1 Complete Experiment Timeline

| Experiment | Champion | Best Epoch | Val Accuracy | Macro F1 | Weighted F1 | Status |
|------------|----------|-----------|--------------|----------|------------|--------|
| Baseline | ✓ | 24 | 85.41% | 83.33% | 85.60% | Complete |
| Targeted Augmentation | – | 20 | 85.00% | 80.32% | – | Complete (worse) |
| Targeted Mixup | – | 16 | 82.70% | 73.81% | – | Complete (worse) |
| Targeted Balanced Augmentation | – | 16 | 78.92% | 74.11% | – | Complete (worse) |
| Targeted Augmentation V2 | – | 20 | 87.84% | 84.95% | 87.97% | Complete |
| **class_aware_finetune_v1** | ✓✓ | 13 | 88.51% | 86.67% | 88.78% | Complete ← Previous Champion |
| **class_aware_finetune_320_v1** | ? | ? | ? | ? | ? | **RUNNING (Epoch 4/25)** ← Current |

### 8.2 class_aware_finetune_320_v1 Training History (So Far)

**History file:** `reports/experiments/class_aware_finetune_320_v1/history.csv` (initial 3 epochs)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Macro F1 | Weighted F1 | LR | Time (s) |
|-------|-----------|-----------|----------|---------|----------|------------|-----|----------|
| 1 | 1.4919 | 0.6266 | 1.3485 | 0.6743 | 0.5891 | 0.6816 | 9.96e-4 | 542.2 |
| 2 | 1.2397 | 0.6671 | 1.3231 | 0.6770 | 0.6139 | 0.6933 | 9.83e-4 | 1133.5 |
| 3 | 1.1003 | 0.7390 | 1.0812 | 0.8122 | 0.7481 | 0.8198 | 9.95e-4 | 2682.33 |

**Resumed history:** `reports/experiments/class_aware_finetune_320_v1/resumed/history.csv` (current)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Macro F1 | Weighted F1 | LR | Time (s) |
|-------|-----------|-----------|----------|---------|----------|------------|-----|----------|
| 4 | 0.8853 | 0.8169 | 0.9919 | 0.8189 | 0.7596 | 0.8239 | 9.95e-4 | 725.78 |

**Observation:** Significant improvement from epoch 3→4 after resume; features[-3:] unlocked

### 8.3 Baseline Comparison

**Champion (class_aware_finetune_v1) baseline values:**

```
Validation Accuracy: 88.51%
Validation Macro F1: 86.67%
Validation Weighted F1: 88.78%
```

**Current experiment (320×320) target:**
- Beat Macro F1: > 86.67%
- Current (epoch 4): 75.96% ← Still behind, but early

**Status:** Training in progress; will evaluate on held-out test set if Macro F1 improves

---

## PART 9: CHECKPOINT SYSTEM

### 9.1 Checkpoint Directory Structure

**Initial run:**
```
models/experiments/class_aware_finetune_320_v1/
└── best_checkpoint.pth
```

**Resumed run:**
```
models/experiments/class_aware_finetune_320_v1/
├── best_checkpoint.pth          (original from initial run)
└── resumed/
    ├── best_checkpoint.pth      ◄─ ACTIVE (current training)
    └── best_validation_confusion_matrix.png
```

### 9.2 Checkpoint Filename

**Pattern:** `best_checkpoint.pth`

**Reason:** Always overwrites when Macro F1 improves (single "best" per experiment run)

### 9.3 Saved Model State

```python
torch.save({
    "model_state_dict": model.state_dict(),      # ← Model weights only
    "class_names": CLASS_NAMES,                  # ← 9 classes
    "experiment_name": EXPERIMENT_NAME,          # ← "class_aware_finetune_320_v1"
    "best_validation_macro_f1": float,
    "best_validation_accuracy": float,
    "best_validation_weighted_f1": float,
    "best_epoch": int,
}, CHECKPOINT_PATH)
```

### 9.4 Optimizer State

**NOT saved in checkpoint** (reconstructed on resume)

**Reason:** Allows for aggressive fine-tuning parameters on resume

### 9.5 Scheduler State

**NOT saved in checkpoint** (reconstructed on resume)

**Reason:** Scheduler reset to new T_max for remaining epochs

### 9.6 Epoch Information

**Saved:** `best_epoch` (best validation Macro F1 epoch)

**Loaded on resume:** `start_epoch = best_epoch + 1`

### 9.7 Best Macro F1 Information

**Saved fields:**
- `best_validation_macro_f1` (float)
- `best_validation_accuracy` (float)
- `best_validation_weighted_f1` (float)

### 9.8 Resume Mechanism

**Command:** `python src/class_aware_finetune_320_v1.py --resume`

**Flow:**
1. Load checkpoint
2. Verify class mapping matches
3. Load model state
4. Get best_epoch from checkpoint
5. Set start_epoch = best_epoch + 1
6. Unfreeze features[-3:]
7. Create new optimizer with dual LR
8. Append to history.csv (append mode)

**Source:** `src/class_aware_finetune_320_v1.py` (lines 171-235)

### 9.9 Separate Checkpoints

**Baseline:** `models/best_efficientnet_b0.pth`  
**Champion (v1):** `models/best_efficientnet_b0_class_aware_finetune_v1.pth`  
**Current (320×320):** `models/experiments/class_aware_finetune_320_v1/resumed/best_checkpoint.pth`

Each is independent; current training will NOT overwrite previous champions

---

## PART 10: VALIDATION

### 10.1 Training → Validation Flow

```
TRAINING SET (70%, ~3,450 images)
         ↓
Model Training
(epoch N)
         ↓
VALIDATION SET (15%, ~740 images)
         ↓
Model Evaluation
(NO GRADIENT)
         ↓
Metrics Calculated
(Accuracy, Macro F1, Weighted F1)
         ↓
Comparison vs. best Macro F1
         ↓
IF Macro F1 improved:
  Save checkpoint
  stale = 0
ELSE:
  stale += 1
  IF stale == 5: STOP
```

### 10.2 Validation Accuracy

**Calculation:** `accuracy_score(actual_labels, predicted_labels)`

**Scale:** 0–1 (or 0–100%)

### 10.3 Validation Macro F1

**Calculation:** 
```python
precision, recall, f1_values, _ = precision_recall_fscore_support(
    actual, predicted, labels=range(9), zero_division=0
)
macro_f1 = f1_values.mean()  # Unweighted average
```

**Purpose:** Balanced metric across all classes (ignores class imbalance)

### 10.4 Validation Weighted F1

**Calculation:**
```python
weighted_f1 = np.average(f1_values, weights=support)
```

**Purpose:** Real-world metric reflecting actual class distribution

### 10.5 Best Epoch Selection

**Criterion:** Validation Macro F1 (maximized)

**Logic:**
```python
if validation_macro_f1 > best["macro_f1"]:
    best_epoch = current_epoch
    save_checkpoint()
    stale = 0
```

### 10.6 Checkpoint Selection

**Condition:** Highest validation Macro F1 across all epochs

**Guarantee:** Best checkpoint always corresponds to best Macro F1

### 10.7 Validation Data Isolation

✓ Validation data NOT used for training  
✓ Validation gradients NOT computed  
✓ Validation metrics NOT used for loss  
✓ Validation split is independent

---

## PART 11: TEST EVALUATION

### 11.1 Test Evaluator for EfficientNet-B0 IBUG Models

**For class_aware_finetune_v1:**
- **Evaluator:** `src/evaluate_class_aware_finetune_v1_test.py`
- **Checkpoint:** `models/best_efficientnet_b0_class_aware_finetune_v1.pth`
- **Status:** Complete (test results exist)

**For class_aware_finetune_320_v1:**
- **Evaluator:** None yet (in development or awaiting completion)
- **Checkpoint:** `models/experiments/class_aware_finetune_320_v1/resumed/best_checkpoint.pth`
- **Status:** Training in progress; evaluator will be needed after convergence

**For targeted_augmentation_v2:**
- **Evaluator:** `src/evaluate_targeted_augmentation_v2.py`
- **Checkpoint:** `models/best_efficientnet_b0_targeted_augmentation_v2.pth`
- **Status:** Complete

### 11.2 Dedicated 320×320 Evaluator

**Status:** NOT CONFIRMED — no dedicated evaluator found yet

**Need:** When class_aware_finetune_320_v1 converges, need to:
1. Create evaluator (or adapt existing)
2. Ensure 320×320 preprocessing
3. Load checkpoint from `models/experiments/class_aware_finetune_320_v1/resumed/best_checkpoint.pth`
4. Evaluate on held-out TEST split

### 11.3 Checkpoint for Evaluation

**Current:** `models/experiments/class_aware_finetune_320_v1/resumed/best_checkpoint.pth`

**Will update:** When training converges (Macro F1 no longer improves for 5 epochs)

### 11.4 Test Dataset

**Location:** `dataset/processed/fabric_9class/test/`

**Count:** ~740 images (15% of 4,932 total)

**Guarantee:** Held-out; never seen during training or validation

### 11.5 Held-Out Test Split

✓ TEST split is COMPLETELY held-out  
✓ NO test data used during training  
✓ NO test data used during validation  
✓ NO test metrics used for checkpoint selection  

### 11.6 Evaluation Metrics

**Calculated on test set:**

1. **Accuracy:** `accuracy_score(actual, predicted)`
2. **Precision (per-class & macro):** `precision_recall_fscore_support()[0]`
3. **Recall (per-class & macro):** `precision_recall_fscore_support()[1]`
4. **F1 (per-class, Macro, Weighted):** `precision_recall_fscore_support()[2]`
5. **Support (per-class count):** `precision_recall_fscore_support()[3]`
6. **Top-3 Accuracy:** `mean([label in top_3 for label, probs in zip(...)])`

### 11.7 Confusion Matrix

**Generated:** After all test predictions made

**Content:** 9×9 matrix of actual vs. predicted classes

**Visualization:** Heatmap (PNG)

**File:** `reports/test_confusion_matrix.png` (when evaluator runs)

### 11.8 Classification Report

**Format:** sklearn text report

**Content:**
- Per-class: precision, recall, F1-score, support
- Macro averages
- Weighted averages

**File:** `reports/test_classification_report.txt` (when evaluator runs)

**Source (template):** `src/evaluate_class_aware_finetune_v1_test.py` (lines 72-75)

---

## PART 12: PREDICTION PIPELINE

### 12.1 User Image → Prediction Flow

```
User provides image file path
         ↓
src/predict.py
         ↓
1. Load checkpoint
   └─ models/best_efficientnet_b0.pth
         ↓
2. Preprocess image
   └─ Open, convert to RGB
   └─ Apply deterministic transform
   └─ Normalize with ImageNet stats
         ↓
3. Load model
   └─ Create EfficientNet-B0 (9 classes)
   └─ Load saved weights
   └─ Set to eval mode
         ↓
4. Forward pass
   └─ Input: 1×3×320×320 tensor
   └─ Output: 1×9 logits
         ↓
5. Softmax → probabilities
   └─ torch.softmax(logits, dim=1)
   └─ Sum to 1.0
         ↓
6. Select top class
   └─ argmax(probabilities)
         ↓
7. Get confidence
   └─ max(probabilities)
         ↓
8. Get top-3
   └─ probabilities.topk(3)
         ↓
Output JSON:
{
  "predicted_class": "Cotton",
  "confidence": 0.92,
  "top_3": [
    {"class_name": "Cotton", "probability": 0.92},
    {"class_name": "Polyester", "probability": 0.06},
    {"class_name": "Silk", "probability": 0.02}
  ]
}
```

### 12.2 Source File: Prediction Entrypoint

**File:** `src/predict.py`

**Function:** `main()`

**Arguments:** 
```python
parser.add_argument("image_path")
```

### 12.3 Image Preprocessing

**Steps:**
1. Open with PIL.Image
2. Convert to RGB (if not already)
3. Apply deterministic transform

**Code:**
```python
with Image.open(args.image_path) as image:
    tensor = deterministic(image.convert("RGB")).unsqueeze(0).to(device)
```

**Source:** `src/predict.py` (lines 28-30)

### 12.4 Model Loading

**Checkpoint:** `models/best_efficientnet_b0.pth`

**Loading:**
```python
checkpoint = torch.load(MODELS_ROOT / "best_efficientnet_b0.pth", 
                        map_location=device, weights_only=False)
network = create_model(len(CLASS_NAMES), pretrained=False)
network.load_state_dict(checkpoint["model_state_dict"])
network.eval()
```

**Source:** `src/predict.py` (lines 23-26)

### 12.5 Forward Pass

**Input:** Single image (1×3×320×320)

**Output:** Logits (1×9)

**Code:**
```python
with torch.no_grad():
    probabilities = torch.softmax(network(tensor), 1)[0]
```

**Source:** `src/predict.py` (lines 31-32)

### 12.6 Prediction Probabilities

**Method:** `torch.softmax(logits, dim=1)`

**Output:** Probabilities for each of 9 classes (sum = 1.0)

**Source:** `src/predict.py` (line 32)

### 12.7 Final Class Selection

**Method:** `argmax(probabilities)`

**Output:** Index (0–8) → class name via CLASS_NAMES

**Code:**
```python
values, indices = probabilities.topk(3)
result = {
    "predicted_class": CLASS_NAMES[indices[0].item()],
    "confidence": float(values[0]),
    "top_3": [...]
}
```

**Source:** `src/predict.py` (lines 33-35)

### 12.8 Confidence Score

**Definition:** Probability of predicted class

**Calculation:** `max(probabilities)`

**Range:** 0–1 (or 0–100%)

**Source:** `src/predict.py` (line 34)

### 12.9 Output Format

**JSON:**
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

**Source:** `src/predict.py` (lines 33-37)

---

## PART 13: APPLICATION PIPELINE

### 13.1 Backend/app/ai Directory Overview

| File | Purpose | Connection to IBUG EfficientNet-B0 |
|------|---------|----------------------------------|
| model_loader.py | Load Keras .keras models | LEGACY (Keras, not EfficientNet) |
| inference_service.py | Unified inference dispatcher | Attempts Keras first; backend fallback |
| model_registry.py | Metadata I/O | Tracks model version |
| predict.py | Single inference call | Not used for EfficientNet-B0 |
| model_loader.py | Load Keras material classifier | LEGACY (not connected) |
| dataset_manager.py | Dataset I/O | For training; not for inference |
| dataset_preprocessor.py | Image preprocessing | Generic; not EfficientNet-specific |
| evaluate.py | Evaluation entrypoint | LEGACY (evaluation) |
| evaluate_cli.py | CLI evaluation | LEGACY |
| evaluate_artifacts.py | Artifact evaluation | LEGACY |
| feature_extractor.py | Feature extraction | Not used for classification inference |
| confidence.py | Confidence calibration | Generic confidence scoring |
| training_config.py | Training configuration | LEGACY (Keras) |
| train.py | Training script | LEGACY (Keras/sklearn, not EfficientNet) |
| mobilenet_preprocess.py | MobileNetV3 preprocessing | LEGACY (MobileNetV3, not EfficientNet) |
| sklearn_train.py | Scikit-learn training | LEGACY (sklearn, not neural) |

### 13.2 Current IBUG EfficientNet-B0 Connection Status

**Direct inference via `src/predict.py`:**
- ✓ Works standalone
- ✓ Loads checkpoint
- ✓ Predicts class

**Backend integration:**
- ✗ NO direct connection found
- ✗ Backend uses Keras model (not EfficientNet PyTorch)
- ✗ Backend uses 5-class contract (not 9-class)

**Conclusion:** EfficientNet-B0 trained model exists but is NOT integrated into production Backend/app

### 13.3 Legacy Components Identified

**MobileNetV3:**
- `Backend/app/ai/mobilenet_preprocess.py` — Custom preprocessing
- `model_loader.py` — Loads MobileNetV3 Keras model
- **Status:** LEGACY

**3-Class Model:**
- `Backend/app/ai/inference_service.py` — References 3-class contract initially
- `Backend/material_classes.py` — Defines MODEL_CLASSES (appears to be 3–5 classes)
- **Status:** LEGACY

**Sklearn:**
- `Backend/app/ai/sklearn_train.py` — Sklearn training
- `Backend/app/ai/training_config.py` — Keras config
- **Status:** LEGACY

### 13.4 Active Files Possibly Connected

**None reliably connected to EfficientNet-B0 at application level**

**Inference service signature:**
```python
class MaterialInferenceService:
    def _load_model(self) -> None:
        # Attempts to load Keras .keras model
        # Falls back to sklearn bundle
        # No PyTorch EfficientNet support found
```

**Source:** `Backend/app/ai/inference_service.py` (lines 66–100)

---

## PART 14: FRONTEND / BACKEND CONNECTION

### 14.1 Complete Communication Flow

```
FRONTEND (Vue.js / React)
    ↓
Upload image via form
    ↓
HTTP POST /api/classify
    ↓
BACKEND (FastAPI / Flask in Backend/main.py)
    ↓
datasets_router.py
    or
inference_service.py
    ↓
Load model (Keras or sklearn)
    ├─ NOT PyTorch EfficientNet-B0
    ├─ NOT from models/ checkpoint
    └─ Uses Backend/data/processed/material_deepfashion
    ↓
Preprocess image
    (MobileNetV3 or generic)
    ↓
Inference
    ↓
Get prediction (class name, confidence)
    ↓
HTTP 200 JSON response
    ↓
FRONTEND displays result
```

### 14.2 Exact Files for Each Stage

| Stage | File | Function |
|-------|------|----------|
| Frontend upload | `Frontend/src/` | Form submission |
| API endpoint | `Backend/main.py` | FastAPI route |
| Router logic | `Backend/datasets_router.py` | Route handler |
| Inference dispatch | `Backend/app/ai/inference_service.py` | `MaterialInferenceService.classify()` |
| Model loading | `Backend/app/ai/model_loader.py` | `load_keras_material_model()` |
| Preprocessing | `Backend/app/ai/dataset_preprocessor.py` or `mobilenet_preprocess.py` | Image normalization |
| Prediction | `Backend/app/ai/inference_service.py` | `predict()` call |
| Response | `Backend/main.py` | JSON serialization |
| Frontend display | `Frontend/src/` | Render result |

### 14.3 Missing Connection

**Problem:** Production backend does NOT use trained EfficientNet-B0 model

**Reason:** 
1. Backend expects Keras .keras model artifact
2. EfficientNet-B0 is PyTorch (not Keras)
3. Model saved as .pth (PyTorch state dict)
4. No PyTorch inference service in Backend/app/ai

**Resolution required:** 
- Export EfficientNet-B0 to ONNX or Keras format, OR
- Add PyTorch inference service to Backend, OR
- Create bridge/wrapper for .pth checkpoint

---

## PART 15: DEEPFASHION FILES

### [DEEPFASHION — DO NOT TOUCH]

**Important:** DeepFashion is a SEPARATE DATASET; training is ongoing (unconfirmed if active)

**Primary DeepFashion data:**
```
Backend/data/processed/material_deepfashion/
  ├── Cotton/
  ├── Polyester/
  ├── [other classes]/
  └── [~1500 images from DeepFashion captions]

Backend/data/processed/material_deepfashion_verified_clean/
  └── [cleaned DeepFashion subset]
```

**DeepFashion manifest:**
```
Backend/data/manifests/material_deepfashion_manifest.json
```

**DeepFashion-related scripts:**
```
adjudicate_conflicting_garment_labels.py
  └─ Resolves label conflicts in DeepFashion
  
finalize_material_dataset.py
  └─ Finalizes DeepFashion dataset

audit_ibug_fabrics.py
  └─ Compares iBUG vs DeepFashion domain
```

**DeepFashion in documentation:**
```
docs/DATASET_REPORT.md
  → Describes material_deepfashion_verified_clean dataset
```

**Configuration:**
```
Backend/config/class_mapping.yaml
  → Keyword extraction for DeepFashion captions
```

**DO NOT:**
- Delete files in `Backend/data/processed/material_deepfashion*`
- Modify `adjudicate_conflicting_garment_labels.py`
- Modify `finalize_material_dataset.py`
- Modify DeepFashion training scripts (if running)
- Alter manifests
- Merge DeepFashion with iBUG splits

---

## PART 16: LEGACY FILES

### Legacy/Obsolete Components

| File | Purpose | Why Legacy | Current Status |
|------|---------|-----------|-----------------|
| `Backend/app/ai/model_loader.py` | Load Keras models | EfficientNet is PyTorch, not Keras | Obsolete (not used for EfficientNet) |
| `Backend/app/ai/train.py` | Keras training | EfficientNet not trained via this | Obsolete |
| `Backend/app/ai/sklearn_train.py` | Sklearn training | Low-capacity model | Obsolete |
| `Backend/app/ai/training_config.py` | Keras config | Not for EfficientNet | Obsolete |
| `Backend/app/ai/mobilenet_preprocess.py` | MobileNetV3 preprocessing | EfficientNet has different inputs | Obsolete (not for EfficientNet) |
| `src/train.py` | Baseline EfficientNet training (224×224) | Superseded by class_aware_finetune | Deprecated |
| `src/class_aware_finetune_v1.py` | Previous fine-tuning (224×224) | Superseded by 320×320 variant | Deprecated (but champion) |
| `src/targeted_augmentation_v2.py` | Old augmentation strategy | Superseded by class-aware | Deprecated (experiment) |
| `src/targeted_balanced_augmentation.py` | Old balancing strategy | Inferior performance | Obsolete (experiment) |
| `src/targeted_mixup.py` | Mixup augmentation trial | Poor results | Obsolete (experiment) |
| `src/evaluate.py` | Generic evaluator | Superseded by experiment-specific evaluators | Deprecated |
| `src/evaluate_targeted_augmentation_v2.py` | Old evaluator | For old experiment | Deprecated |
| `Backend/app/ai/evaluate_cli.py` | CLI evaluation | Not for current model | Deprecated |
| `Backend/app/ai/feature_extractor.py` | Feature extraction | Not needed for classification inference | Unused |
| `Backend/app/ai/dataset_manager.py` | Dataset I/O for training | Not for inference | Unused (training-only) |
| `Backend/app/ai/predict.py` | Backend prediction wrapper | Unclear if used | Possibly unused |

### Does Current Training Depend on Legacy Files?

| Legacy File | Current Training Depends? | Reason |
|-------------|--------------------------|--------|
| `Backend/app/ai/model_loader.py` | NO | EfficientNet-B0 is PyTorch |
| `Backend/app/ai/train.py` | NO | Using src/class_aware_finetune_320_v1.py |
| `Backend/app/ai/sklearn_train.py` | NO | Neural network training only |
| `Backend/app/ai/mobilenet_preprocess.py` | NO | Using torchvision transforms |
| `src/train.py` | NO | Superseded by current script |
| `src/evaluate.py` | NO | Using experiment-specific evaluators |

**Conclusion:** NO legacy files are required for current training.

---

## PART 17: COMPLETE END-TO-END FLOW

### 17.1 Full Pipeline: IBUG → Prediction

```
IBUG RAW DATA
(original iBUG dataset, external source)
    ↓
    ↓ build_ibug_material_v2.py
    ↓ (extract fabric labels, resolve duplicates)
    ↓
EXTRACTED IBUG LABELS + IMAGES
    ↓
    ↓ build_fabric_16class.py
    ↓ (organize into 16 class folders)
    ↓
16-CLASS DATASET
dataset/processed/fabric_16class/all/
├── Cotton/          [source images]
├── Polyester/
├── Denim/
├── [16 classes total]
└── [~7000 images]
    ↓
    ↓ build_fabric_9class.py
    ↓ (select 9 strongest classes)
    ↓
9-CLASS DATASET
dataset/processed/fabric_9class/all/
├── Cotton/          [4932 images total]
├── Polyester/
├── Denim/
├── Wool/
├── Nylon/
├── Viscose/
├── Silk/
├── Fleece/
└── Terrycloth/
    ↓
    ↓ split_dataset.py (seed=42, stratified, sample-grouped)
    ↓ (verify: no duplicates, no leakage, no sample-group split)
    ↓
TRAIN / VALIDATION / TEST SPLIT
    ├─ train/           [70%, ~3450 images]
    ├─ validation/      [15%, ~740 images]
    └─ test/            [15%, ~740 images, HELD-OUT]
    
    HELD-OUT TEST IS NEVER USED FOR TRAINING OR VALIDATION CHECKPOINT SELECTION
    ↓
    ↓ class_aware_finetune_320_v1.py (CURRENTLY RUNNING — PID 16400)
    ↓
LOAD TRAINING CONFIGURATION
├─ Model: EfficientNet-B0 (pretrained, ImageNet)
├─ Input: 320×320 RGB images
├─ Augmentation:
│  └─ RandomResizedCrop(320, 0.85–1.0)
│  └─ RandomHorizontalFlip(p=0.5)
│  └─ RandomRotation(±8°)
│  └─ ColorJitter(0.10, 0.10, 0.05, 0.02)
├─ Batch size: 16
├─ Optimizer: AdamW (LR 1e-3 warmup, 1e-4 fine-tune)
├─ Loss: CrossEntropyLoss (class weights, label smoothing 0.05)
├─ Scheduler: CosineAnnealingLR
├─ Max epochs: 25
├─ Warmup epochs: 2 (classifier only)
├─ Early stopping: patience 5 epochs
└─ Seed: 42 (deterministic)
    ↓
    ↓ EPOCH LOOP (1 to 25)
    ↓
FOR EACH EPOCH:
    ├─ TRAINING PHASE
    │  └─ Forward pass on train/ (augmented)
    │  └─ Compute loss (class-weighted CE + label smoothing)
    │  └─ Backward pass
    │  └─ Update weights (frozen features during warmup)
    │  └─ Log train loss, accuracy
    │
    ├─ VALIDATION PHASE
    │  └─ Forward pass on validation/ (no augmentation)
    │  └─ Compute validation loss
    │  └─ Calculate accuracy, Macro F1, Weighted F1
    │  └─ Compare to best Macro F1
    │  │
    │  └─ IF validation_macro_f1 > best_macro_f1:
    │     ├─ Save checkpoint
    │     └─ Save confusion matrix
    │  │
    │  └─ ELSE:
    │     ├─ Increment stale counter
    │     └─ IF stale >= 5: STOP (early stopping)
    │
    ├─ SCHEDULER STEP
    │  └─ Cosine annealing decay
    │
    └─ HISTORY LOGGING
       └─ Append row to history.csv
       └─ Format: epoch, train_loss, train_acc, val_loss, val_acc, macro_f1, weighted_f1, lr, elapsed_s

    POST-WARMUP (after epoch 2):
    └─ Unfreeze features[-3:] (last 3 blocks)
    └─ Update optimizer with two LR groups
    └─ Continue training
    ↓
END OF TRAINING (convergence or epoch 25)
    ↓
SAVE FINAL METRICS
    ├─ Best epoch
    ├─ Best validation accuracy
    ├─ Best validation Macro F1
    ├─ Best validation Weighted F1
    ├─ Final confusion matrix (PNG)
    └─ final.json (metadata)
    ↓
BEST CHECKPOINT
models/experiments/class_aware_finetune_320_v1/resumed/best_checkpoint.pth
    ├─ model_state_dict (weights)
    ├─ class_names (9 classes)
    ├─ experiment_name
    ├─ best_validation_macro_f1
    ├─ best_validation_accuracy
    ├─ best_validation_weighted_f1
    └─ best_epoch
    ↓
    ↓ (CONDITIONAL: if Macro F1 > baseline)
    ↓
HELD-OUT TEST EVALUATION (when ready)
    ├─ Load checkpoint
    ├─ Forward pass on test/ (no augmentation, ~740 images)
    ├─ Compute test Accuracy, Macro F1, Weighted F1, Top-3
    ├─ Generate test confusion matrix
    ├─ Generate classification report
    └─ Compare to baseline
    ↓
TEST RESULTS (report)
    ├─ Test accuracy
    ├─ Test Macro F1
    ├─ Test Weighted F1
    ├─ Per-class precision, recall, F1
    └─ Confusion matrix (PNG)
    ↓
    ↓ Deploy to Backend (if approved)
    ↓
MODEL DEPLOYMENT (NOT YET CONNECTED)
    ├─ Export checkpoint to ONNX or Keras (needed)
    ├─ OR integrate PyTorch inference service (needed)
    ├─ Update Backend/app/ai/inference_service.py
    └─ Register model in Backend/app/ai/model_registry.py
    ↓
PRODUCTION INFERENCE
    ├─ User uploads image to Frontend
    ├─ Frontend POSTs to Backend /api/classify
    ├─ Backend calls MaterialInferenceService.classify()
    ├─ Service loads EfficientNet-B0 checkpoint
    ├─ Preprocess: Load image, Normalize (320×320)
    ├─ Forward pass
    ├─ Softmax → probabilities
    ├─ argmax → predicted class
    ├─ top_3 probabilities
    ├─ Return JSON:
    │  {
    │    "predicted_class": "Cotton",
    │    "confidence": 0.92,
    │    "top_3": [...]
    │  }
    └─ Frontend displays result
```

### 17.2 File Responsibility by Pipeline Stage

| Pipeline Stage | File | Function/Line | Action |
|---|---|---|---|
| 1. iBUG dataset input | (external) | – | Raw images + fabric labels |
| 2. Extract & organize | `build_ibug_material_v2.py` | `main()` | Build 16-class dataset |
| 3. Select 9 classes | `src/build_fabric_9class.py` | `main()` | Copy best 9 classes |
| 4. Stratified split | `src/split_dataset.py` | `main()` | Create train/val/test |
| 5. Train data load | `src/model.py` | `make_datasets()` | ImageFolder loaders |
| 6. Image augmentation | `src/class_aware_finetune_320_v1.py` | `build_train_transform()` | RandomResizedCrop, etc. |
| 7. Normalization | `src/class_aware_finetune_320_v1.py` | line 140 | Normalize(MEAN, STD) |
| 8. Model creation | `src/model.py` | `create_model()` | efficientnet_b0 + head |
| 9. Training loop | `src/class_aware_finetune_320_v1.py` | `main()` | Epoch loop, forward/backward |
| 10. Validation | `src/class_aware_finetune_320_v1.py` | `validation_pass()` | Evaluate on val set |
| 11. Checkpoint save | `src/class_aware_finetune_320_v1.py` | line 289-297 | torch.save() |
| 12. History logging | `src/class_aware_finetune_320_v1.py` | `save_history()` | Append to CSV |
| 13. Test evaluation | `src/evaluate_class_aware_finetune_v1_test.py` | `main()` | Test on held-out set |
| 14. Model loading | `Backend/app/ai/model_loader.py` | `load_keras_material_model()` | (LEGACY — needs replacement) |
| 15. Inference | `src/predict.py` (CLI) or Backend (TODO) | `main()` or TBD | Single image prediction |
| 16. Response | `Backend/main.py` | API route handler | JSON response |

---

## PART 18: REVIEWER EXPLANATION

### Why These Experiments Were Run Sequentially

**Experiment progression:**

1. **Baseline** (224×224, no class weights, no augmentation)
   - Result: Accuracy 85.41%, Macro F1 83.33%
   - Established benchmark
   - Identified: Model generalizes; class imbalance issue

2. **Targeted Augmentation** (224×224, ColorJitter + Rotation)
   - Goal: Improve robustness via augmentation
   - Result: Accuracy 85.00%, Macro F1 80.32% ✗
   - Finding: Aggressive augmentation hurt performance
   - Rejected: F1 worse than baseline

3. **Targeted Mixup** (224×224, Mixup regularization)
   - Goal: Smooth decision boundaries via data interpolation
   - Result: Accuracy 82.70%, Macro F1 73.81% ✗
   - Finding: Mixup too disruptive for fabric classification
   - Rejected: Significant F1 drop

4. **Targeted Balanced Augmentation** (224×224, weighted sampler)
   - Goal: Class balance via weighted sampling
   - Result: Accuracy 78.92%, Macro F1 74.11% ✗
   - Finding: Oversampling minority classes created overfitting
   - Rejected: F1 collapsed

5. **Targeted Augmentation V2** (224×224, tuned augmentation)
   - Goal: Moderate augmentation with better hyperparameters
   - Result: Accuracy 87.84%, Macro F1 84.95%, Weighted F1 87.97% ✓
   - Finding: Improved over baseline; more robust
   - Status: Competitive, new champion

6. **class_aware_finetune_v1** (224×224, class-aware + gradual unfreezing)
   - Goal: Fine-tune pretrained features + balance classes
   - Result: Accuracy 88.51%, Macro F1 86.67%, Weighted F1 88.78% ✓✓
   - Finding: Best on 224×224; two-stage training effective
   - Status: **CHAMPION** (current baseline)

7. **class_aware_finetune_320×320** (320×320, class-aware + higher resolution)
   - Goal: Capture more fabric detail at higher resolution
   - Result: Training in progress (Epoch 4/25)
   - Current: Macro F1 75.96% (early, after resume)
   - Hypothesis: Higher resolution allows finer texture discrimination
   - Status: **RUNNING** (may beat champion)

### Why Failed Experiments Were Rejected

| Experiment | Macro F1 | Baseline | Reject Reason |
|---|---|---|---|
| Targeted Augmentation | 80.32% | 83.33% | -3.01pp below baseline; worse generalization |
| Targeted Mixup | 73.81% | 83.33% | -9.52pp; Mixup too disruptive for discrete classes |
| Targeted Balanced Augmentation | 74.11% | 83.33% | -9.22pp; Oversampling harmed metric learning |

**General pattern:** Introduced regularization hurt more than helped; model needs stability, not aggression.

### Why Macro F1 Is the Primary Selection Metric

**Class imbalance in fabric_9class:**

| Class | Count | % of Total |
|-------|-------|-----------|
| Cotton | 2320 | 47.0% |
| Polyester | 852 | 17.3% |
| Denim | 644 | 13.1% |
| Wool | 344 | 7.0% |
| Nylon | 228 | 4.6% |
| Viscose | 148 | 3.0% |
| Silk | 144 | 2.9% |
| Fleece | 132 | 2.7% |
| Terrycloth | 120 | 2.4% |

**Why accuracy alone is insufficient:**

If model predicts Cotton for everything:
- Accuracy: 47.0% ✓ (misleading; looks okay)
- Macro F1: 9.3% ✗ (exposes bias; fails for all other classes)

**Macro F1 enforces:**
- Per-class fairness (minority classes count equally)
- Balanced generalization across all 9 classes
- Honest performance metric for imbalanced data

**Therefore:**
- Selection criterion: **Maximize Macro F1** (not accuracy)
- This ensures model works for rare classes (Terrycloth, Fleece, Silk)
- Prevents majority-class overfitting

---

## PART 19: FILES THAT MUST NOT BE TOUCHED WHILE TRAINING

### Critical Active Files (DO NOT MODIFY)

✗ **NEVER MODIFY WHILE PID 16400 IS RUNNING:**

```
CHECKPOINT STATE
└─ models/experiments/class_aware_finetune_320_v1/resumed/best_checkpoint.pth
   ├─ BEING UPDATED in real-time
   └─ Corruption will halt training

TRAINING HISTORY
└─ reports/experiments/class_aware_finetune_320_v1/resumed/history.csv
   ├─ BEING APPENDED every epoch
   └─ Corruption will halt training

TRAINING SCRIPT
└─ src/class_aware_finetune_320_v1.py
   ├─ Running in memory (modifications won't affect current run)
   └─ DO NOT MODIFY anyway (will break resume)

DATASET
├─ dataset/processed/fabric_9class/train/
├─ dataset/processed/fabric_9class/validation/
├─ dataset/processed/fabric_9class/test/
│  └─ MUST REMAIN UNCHANGED
│  └─ Any image addition/deletion breaks training

DATASET REFERENCES
├─ dataset/processed/fabric_16class/
│  └─ Source backup (should not be touched)

CLASS MAPPING
└─ src/model.py (CLASS_NAMES, mapping functions)
   └─ MUST remain consistent
```

### Critical Reference Files (READ-ONLY)

| File | Status | Why Protected |
|------|--------|---------------|
| `src/model.py` | Read-only | CLASS_NAMES must match checkpoint |
| `models/best_efficientnet_b0_class_aware_finetune_v1.pth` | Read-only | Baseline reference |
| `models/best_efficientnet_b0_targeted_augmentation_v2.pth` | Read-only | V2 checkpoint reference |
| `models/best_efficientnet_b0_no_weights.pth` | Read-only | Baseline checkpoint validation |
| `dataset/processed/fabric_9class/` | Read-only | Train/val/test splits locked |

### Safe-to-Modify Files (Non-Critical)

✓ **Safe while training:**
- Documentation (README.md, comments)
- Evaluation scripts (not currently running)
- Backend application code (doesn't affect training)
- Frontend code (doesn't affect training)
- Legacy scripts (not in use)
- Reports (past experiment results)

### Cleanup Forbidden

✗ **DO NOT:**
- Delete any checkpoint
- Rename any checkpoint
- Delete any image from train/val/test
- Modify dataset CSV/JSON metadata
- Move dataset folders
- Rename dataset folders
- Delete history files
- Refactor unrelated files
- Perform project cleanup
- Run other training processes

---

## SUMMARY

This textile fabric classification project uses a **9-class EfficientNet-B0 model** trained on the **iBUG dataset** with careful attention to:

1. **No data leakage** (stratified split, duplicate detection, sample-group preservation)
2. **Class balance** (weighted loss, class-aware training)
3. **Gradual fine-tuning** (warmup then unfreezing)
4. **Held-out test evaluation** (never used during training)
5. **Macro F1 optimization** (fair across imbalanced classes)

**Current status:** Class-aware fine-tuning at 320×320 resolution is running (Epoch 4/25), attempting to beat the champion model (Macro F1 86.67%) on the same held-out test set.

**Key insight:** The sequential experiments were needed to identify that moderate class-aware fine-tuning with pretrained features works best; aggressive regularization or aggressive sampling hurts performance.

---

**Generated:** 2026-09-02 — Do not modify while PID 16400 is active.

