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

- Test accuracy: **0.8365**
- Macro precision: **0.7981**
- Macro recall: **0.8207**
- Macro F1: **0.7955**
- Weighted F1: **0.8409**
- Top-3 accuracy: **0.9743**

## Per-Class Results

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Cotton | 0.9233 | 0.8649 | 0.8932 | 348 |
| Denim | 0.9565 | 0.9167 | 0.9362 | 96 |
| Fleece | 0.7778 | 0.7000 | 0.7368 | 20 |
| Nylon | 0.5660 | 0.8333 | 0.6742 | 36 |
| Polyester | 0.7164 | 0.7500 | 0.7328 | 128 |
| Silk | 0.5135 | 0.9500 | 0.6667 | 20 |
| Terrycloth | 1.0000 | 1.0000 | 1.0000 | 16 |
| Viscose | 0.8750 | 0.5833 | 0.7000 | 24 |
| Wool | 0.8542 | 0.7885 | 0.8200 | 52 |

## Confusion Matrix Interpretation

The confusion matrix is saved as `reports/confusion_matrix.png`; off-diagonal counts indicate which fabric classes the baseline confuses most.

## Weakest Classes

Silk (F1=0.6667), Nylon (F1=0.6742), Viscose (F1=0.7000)

## Most Confused Class Pairs

Cotton -> Polyester (25), Polyester -> Silk (15), Polyester -> Nylon (11), Cotton -> Nylon (11), Wool -> Cotton (8)

## Limitations and Next Experiments

This is a baseline trained on imbalanced fabric data. Results may be limited by class scarcity, visual similarity, and residual label ambiguity. Recommended next experiments include calibrated sampling, targeted augmentation, and error review without changing the held-out test set.
