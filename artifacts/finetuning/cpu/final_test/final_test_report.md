# EfficientNet-B0 Baseline Report

## Dataset Description

The validated leakage-safe 9-class fabric dataset was used without modifying the original iBUG dataset or the cleaned source dataset.

## Classes

Cotton, Denim, Fleece, Nylon, Polyester, Silk, Terrycloth, Viscose, Wool

## Split Counts

- Train: **3,452**
- Validation: **740**
- Test: **740**

## Model and Transfer Learning

EfficientNet-B0 initialized with ImageNet weights. The classifier head was warmed up with the backbone frozen, then the upper feature layers were fine-tuned. AdamW and validation macro F1 checkpoint selection were used.

## Evaluation

The test set was evaluated once after training using only the best validation-macro-F1 checkpoint.

- Test accuracy: **0.8324**
- Macro precision: **0.7798**
- Macro recall: **0.7735**
- Macro F1: **0.7672**
- Weighted F1: **0.8329**
- Top-3 accuracy: **0.9703**

## Per-Class Results

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Cotton | 0.8856 | 0.8678 | 0.8766 | 348 |
| Denim | 0.9778 | 0.9167 | 0.9462 | 96 |
| Fleece | 0.7857 | 0.5500 | 0.6471 | 20 |
| Nylon | 0.5472 | 0.8056 | 0.6517 | 36 |
| Polyester | 0.7576 | 0.7812 | 0.7692 | 128 |
| Silk | 0.6154 | 0.8000 | 0.6957 | 20 |
| Terrycloth | 1.0000 | 1.0000 | 1.0000 | 16 |
| Viscose | 0.6000 | 0.3750 | 0.4615 | 24 |
| Wool | 0.8491 | 0.8654 | 0.8571 | 52 |

## Confusion Matrix Interpretation

The confusion matrix is saved as `reports/confusion_matrix.png`; off-diagonal counts indicate which fabric classes the baseline confuses most.

## Weakest Classes

Viscose (F1=0.4615), Fleece (F1=0.6471), Nylon (F1=0.6517)

## Most Confused Class Pairs

Cotton -> Polyester (22), Cotton -> Nylon (12), Polyester -> Nylon (11), Viscose -> Cotton (9), Denim -> Cotton (8)

## Limitations and Next Experiments

This is a baseline trained on imbalanced fabric data. Results may be limited by class scarcity, visual similarity, and residual label ambiguity. Recommended next experiments include calibrated sampling, targeted augmentation, and error review without changing the held-out test set.
