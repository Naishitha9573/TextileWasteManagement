# Fabric 16-Class Dataset Cleaning

The original iBUG dataset was preserved as read-only. Images were copied into a derived dataset; no source image was deleted, moved, renamed, or modified.

## Class Policy

The provisional 16 classes focus on clearly defined fabric/textile categories for the first experiment. Artificial fur, artificial leather, blended, felt, leather, Lut, suede, unclassified, and utilities were not used because they are small, ambiguous, general-material, or unclassified categories. These classes remain in the original dataset.

## Duplicate and Conflict Policy

- Same-class exact SHA-256 duplicates keep one deterministic representative; every redundant copy is recorded in `excluded_images.csv`.
- All confirmed cross-class conflict sample IDs are excluded from every approved class, including all available views. This is a provisional conflict policy, not a decision about the correct label.
- Near duplicates are not removed. They are counted as review flags from `near_duplicates.csv`.

## Final Counts

| Class | Original | Same-class exact excluded | Conflict excluded | Near flagged | Final |
|---|---:|---:|---:|---:|---:|
| Acrylic | 48 | 0 | 0 | 0 | 48 |
| Chenille | 52 | 0 | 0 | 4 | 52 |
| Corduroy | 96 | 4 | 0 | 0 | 92 |
| Cotton | 2352 | 32 | 0 | 204 | 2320 |
| Crepe | 104 | 0 | 48 | 0 | 56 |
| Denim | 648 | 4 | 0 | 42 | 644 |
| Fleece | 132 | 0 | 0 | 0 | 132 |
| Linen | 76 | 0 | 0 | 8 | 76 |
| Nylon | 228 | 0 | 0 | 4 | 228 |
| Polyester | 904 | 12 | 40 | 38 | 852 |
| Satin | 96 | 0 | 96 | 0 | 0 |
| Silk | 200 | 0 | 56 | 4 | 144 |
| Terrycloth | 120 | 0 | 0 | 0 | 120 |
| Velvet | 44 | 0 | 0 | 0 | 44 |
| Viscose | 148 | 0 | 0 | 20 | 148 |
| Wool | 360 | 16 | 0 | 4 | 344 |

- Total original approved-class images: **5608**
- Total final images: **5300**
- Total excluded images: **308**

No train/validation/test folders were created. No model was trained.
