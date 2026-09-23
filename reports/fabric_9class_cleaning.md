# Fabric 9-Class Dataset

The original iBUG dataset remains untouched. The cleaned 16-class dataset also remains untouched; this dataset was copied from it into a separate derived directory.

## Selection

This is a focused first experiment using the nine classes with the strongest remaining final counts: Cotton, Polyester, Denim, Wool, Nylon, Viscose, Silk, Fleece, and Terrycloth. Acrylic, Chenille, Corduroy, Crepe, Linen, Satin, and Velvet were excluded because their final usable counts were too small for the first robust experiment. Satin was specifically excluded because it has zero usable images after cross-class conflict handling.

## Copy and Verification

Images were copied without additional duplicate removal. Every copied file was checked for readability and compared with its source using SHA-256. No train/validation/test folders were created and no model was trained.

## Final Counts

| Class | Source | Copied | Missing |
|---|---:|---:|---:|
| Cotton | 2320 | 2320 | 0 |
| Polyester | 852 | 852 | 0 |
| Denim | 644 | 644 | 0 |
| Wool | 344 | 344 | 0 |
| Nylon | 228 | 228 | 0 |
| Viscose | 148 | 148 | 0 |
| Silk | 144 | 144 | 0 |
| Fleece | 132 | 132 | 0 |
| Terrycloth | 120 | 120 | 0 |

- Total images: **4932**
- Expected total: **4932**
- Count validation: **PASS**
