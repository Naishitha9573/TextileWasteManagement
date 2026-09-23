# Validation Error Analysis

MODEL: EfficientNet-B0
CHECKPOINT: C:\Users\sama\OneDrive\Desktop\naishtex\Textile\models\best_efficientnet_b0_no_weights.pth
VALIDATION IMAGE COUNT: 740
VALIDATION ACCURACY: 0.8541
VALIDATION MACRO F1: 0.8333
VALIDATION WEIGHTED F1: 0.8560

## TOP 10 CONFUSION PAIRS

- Cotton -> Polyester: 14 errors (4.02% of true class)
- Denim -> Cotton: 11 errors (11.46% of true class)
- Polyester -> Nylon: 11 errors (8.59% of true class)
- Cotton -> Wool: 9 errors (2.59% of true class)
- Wool -> Cotton: 9 errors (17.31% of true class)
- Cotton -> Nylon: 8 errors (2.30% of true class)
- Polyester -> Silk: 6 errors (4.69% of true class)
- Cotton -> Denim: 5 errors (1.44% of true class)
- Polyester -> Cotton: 5 errors (3.91% of true class)
- Cotton -> Viscose: 4 errors (1.15% of true class)

## WEAKEST 5 CLASSES

- Nylon: F1=0.7089, accuracy=0.8750, support=32, most common wrong prediction=Polyester
- Silk: F1=0.7500, accuracy=0.7500, support=24, most common wrong prediction=Cotton
- Wool: F1=0.7573, accuracy=0.7500, support=52, most common wrong prediction=Cotton
- Polyester: F1=0.8078, accuracy=0.8047, support=128, most common wrong prediction=Nylon
- Viscose: F1=0.8182, accuracy=0.9000, support=20, most common wrong prediction=Cotton

## MOST COMMON WRONG PREDICTIONS

- Cotton: 34 times
- Polyester: 24 times
- Nylon: 19 times
- Wool: 12 times
- Denim: 6 times
- Viscose: 6 times
- Silk: 6 times
- Fleece: 1 times

## HIGH-CONFIDENCE WRONG PREDICTIONS

- Fleece -> Cotton (confidence=0.9981)
- Cotton -> Denim (confidence=0.9978)
- Polyester -> Nylon (confidence=0.9951)
- Cotton -> Denim (confidence=0.9950)
- Cotton -> Polyester (confidence=0.9940)
- Viscose -> Cotton (confidence=0.9874)
- Polyester -> Silk (confidence=0.9855)
- Denim -> Cotton (confidence=0.9844)
- Cotton -> Polyester (confidence=0.9773)
- Polyester -> Nylon (confidence=0.9751)

## LOW-CONFIDENCE CORRECT PREDICTIONS

- dataset\processed\fabric_9class\validation\Cotton\105\im_4.png (Cotton, confidence=0.3546)
- dataset\processed\fabric_9class\validation\Cotton\105\im_4.png (Cotton, confidence=0.3669)
- dataset\processed\fabric_9class\validation\Cotton\1123\im_3.png (Silk, confidence=0.3748)
- dataset\processed\fabric_9class\validation\Cotton\1127\im_4.png (Cotton, confidence=0.3919)
- dataset\processed\fabric_9class\validation\Cotton\1108\im_3.png (Denim, confidence=0.4408)
- dataset\processed\fabric_9class\validation\Cotton\1108\im_4.png (Viscose, confidence=0.4417)
- dataset\processed\fabric_9class\validation\Cotton\1127\im_4.png (Polyester, confidence=0.4667)
- dataset\processed\fabric_9class\validation\Cotton\1122\im_4.png (Cotton, confidence=0.4875)
- dataset\processed\fabric_9class\validation\Cotton\11\im_2.png (Cotton, confidence=0.4884)
- dataset\processed\fabric_9class\validation\Cotton\1146\im_1.png (Silk, confidence=0.4923)

## CANDIDATE FOR MANUAL LABEL REVIEW

- Cotton -> Polyester
- Denim -> Cotton
- Polyester -> Nylon
- Cotton -> Wool
- Cotton -> Nylon
- Polyester -> Silk
- Cotton -> Viscose
- Silk -> Cotton
- Wool -> Polyester
- Fleece -> Cotton
