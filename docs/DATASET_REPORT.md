# Dataset Report

_Generated 2026-08-22T15:52:51 by `python -m ml.datasets.dataset_report`. All numbers below are measured from this repository._

## 1. Datasets in use (final training sets)

### `ibug_material_v2`

- Location: `Backend/data/processed/ibug_material_v2`
- Images: **4164**
- Classes: Cotton: 2748, Nylon: 184, Polyester: 756, Silk: 140, Wool: 336
- Label provenance: manufacturer tag.txt composition metadata

### `material_deepfashion_verified_clean`

- Location: `Backend/data/processed/material_deepfashion_verified_clean`
- Images: **708**
- Classes: Cotton: 469, Denim: 163, Mixed Fabrics: 76
- Label provenance: DeepFashion caption keyword mapping

## 2. Group-aware splits

### `fabric_condition` — total 247

| Split | train | validation | test |
|---|---:|---:|---:|
| Total | 172 | 37 | 38 |
| Defective | 74 | 16 | 16 |
| Non_Defective | 98 | 21 | 22 |

### `ibug_material_v2` — total 4164

| Split | train | validation | test |
|---|---:|---:|---:|
| Total | 2916 | 624 | 624 |
| Cotton | 1924 | 412 | 412 |
| Denim | 0 | 0 | 0 |
| Nylon | 128 | 28 | 28 |
| Polyester | 528 | 116 | 112 |
| Silk | 100 | 20 | 20 |
| Wool | 236 | 48 | 52 |

### `material_deepfashion` — total 1500

| Split | train | validation | test |
|---|---:|---:|---:|
| Total | 1050 | 225 | 225 |
| Cotton | 350 | 75 | 75 |
| Denim | 350 | 75 | 75 |
| Mixed Fabrics | 350 | 75 | 75 |

### `material_deepfashion_verified` — total 1500

| Split | train | validation | test |
|---|---:|---:|---:|
| Total | 1050 | 225 | 225 |
| Cotton | 336 | 85 | 79 |
| Denim | 357 | 75 | 68 |
| Mixed Fabrics | 357 | 65 | 78 |

### `material_deepfashion_verified_clean` — total 708

| Split | train | validation | test |
|---|---:|---:|---:|
| Total | 496 | 106 | 106 |
| Cotton | 329 | 70 | 70 |
| Denim | 114 | 24 | 25 |
| Mixed Fabrics | 53 | 12 | 11 |

## 3. Duplicate & leakage review

- **ibug_material_v2**: cross-split duplicate hashes: **0**, leakage_detected: **False**, unreadable: 0
- **material_deepfashion_verified_clean**: cross-split duplicate hashes: **0**, leakage_detected: **False**, unreadable: 0
- **ibug_material_v2**: exact-duplicate groups in processed set: **0**; cross-sample dHash pairs ≤4: **6026** (texture-similarity dominated — see note); cross-split exact duplicates: **0**
- **material_deepfashion_verified_clean**: exact-duplicate groups in processed set: **0**; cross-sample dHash pairs ≤4: **25** (texture-similarity dominated — see note); cross-split exact duplicates: **0**

Interpretation note: on plain fabric textures, 64-bit dHash distance ≤8 flags visual similarity far more often than duplication (cross-class pairs dominate). Hard evidence of duplication is sha256-identical content: there is none within or across the final training splits. The authoritative source-level iBUG review excluded all cross-sample exact/near-duplicate-flagged groups during derivation.
- **fabric_condition**: not_applicable (independent captures, no multi-view sample grouping)
- **ibug_material_v2**: group-aware split valid: **True** (1041 sample groups; leaked groups: 0)
- **material_deepfashion**: group-aware split valid: **False** (1354 sample groups; leaked groups: 56)
- **material_deepfashion_verified**: group-aware split valid: **True** (1354 sample groups; leaked groups: 0)
- **material_deepfashion_verified_clean**: group-aware split valid: **True** (674 sample groups; leaked groups: 0)

Source-level iBUG duplicate review: 213 exact + 223 near-duplicate groups were identified in the original iBUG download; all groups flagged for cross-sample duplication are excluded from the accepted derived dataset (see `IBUG_MATERIAL_V2_DUPLICATE_REVIEW.md`). No source image was deleted.

## 4. Exclusions

Final manifest rows: **10093** (included: **4872**, excluded: **5221**) — see `Backend/data/metadata/final_dataset_manifest.csv`.

## 5. Preprocessing & augmentation (training)

- Resize to 224×224, RGB, MobileNetV3 `preprocess_input` scaling.
- Augmentation at train time: none beyond resize/autocontrast applied historically; class weighting used for imbalance.
- Splits are sample-group aware (all views of one fabric sample stay together).

## 6. Known limitations

- Labels are manufacturer composition claims (iBUG) or caption-derived mentions (DeepFashion); neither is laboratory-verified fiber content.
- iBUG v2 accepted set has **no Denim class**: no sample declares denim as its sole composition component (denim samples declare cotton).
- Class imbalance: Cotton dominates the accepted iBUG set (~66%).
- Silk (140 images) and Nylon (184 images) are small classes; their metrics will be noisy.
- Waste-category labels (Recyclable/Reusable/Repairable/Upcyclable/Compostable/Hazardous) do not exist in any image dataset here; waste classification remains rule-based by design.
