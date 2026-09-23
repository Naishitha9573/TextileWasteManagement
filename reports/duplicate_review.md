# Duplicate Detection Review

This report is informational only. The original dataset was read-only: no image was deleted, moved, renamed, or modified.

## Method

- Scanned **7,885** supported images recursively across **25** class folders.
- Exact duplicates use SHA-256 of the original file bytes.
- Near duplicates use pHash with a configurable Hamming-distance threshold of **6**.
- Near results are pair-groups, so each reported distance remains directly interpretable; they are review flags, not removal decisions.

## Results

- Exact duplicate groups: **213** (217 duplicate images beyond one representative per group).
- Near duplicate groups: **384** (420 unique images involved).
- Cross-class exact groups: **120**.
- Cross-class near groups: **0**.
- Images with hashing errors: **0**.

## Classes Involved

- Exact duplicate classes: Blended, Corduroy, Cotton, Crepe, Denim, Leather, Polyester, Satin, Silk, Unclassified, Wool
- Near duplicate classes: Artificial_fur, Blended, Chenille, Cotton, Denim, Linen, Nylon, Polyester, Silk, Unclassified, Viscose, Wool

## Cross-Class Groups

- `EXACT-0001` (Crepe, Polyester): Crepe/1000/im_1.png; Polyester/1000/im_1.png
- `EXACT-0002` (Crepe, Polyester): Crepe/1000/im_2.png; Polyester/1000/im_2.png
- `EXACT-0003` (Crepe, Polyester): Crepe/1000/im_3.png; Polyester/1000/im_3.png
- `EXACT-0004` (Crepe, Polyester): Crepe/1000/im_4.png; Polyester/1000/im_4.png
- `EXACT-0005` (Crepe, Polyester): Crepe/1001/im_1.png; Polyester/1001/im_1.png
- `EXACT-0006` (Crepe, Polyester): Crepe/1001/im_2.png; Polyester/1001/im_2.png
- `EXACT-0007` (Crepe, Polyester): Crepe/1001/im_3.png; Polyester/1001/im_3.png
- `EXACT-0008` (Crepe, Polyester): Crepe/1001/im_4.png; Polyester/1001/im_4.png
- `EXACT-0009` (Crepe, Polyester): Crepe/1002/im_1.png; Polyester/1002/im_1.png
- `EXACT-0010` (Crepe, Polyester): Crepe/1002/im_2.png; Polyester/1002/im_2.png
- `EXACT-0011` (Crepe, Polyester): Crepe/1002/im_3.png; Polyester/1002/im_3.png
- `EXACT-0012` (Crepe, Polyester): Crepe/1002/im_4.png; Polyester/1002/im_4.png
- `EXACT-0013` (Crepe, Polyester): Crepe/1003/im_1.png; Polyester/1003/im_1.png
- `EXACT-0014` (Crepe, Polyester): Crepe/1003/im_2.png; Polyester/1003/im_2.png
- `EXACT-0015` (Crepe, Polyester): Crepe/1003/im_3.png; Polyester/1003/im_3.png
- `EXACT-0016` (Crepe, Polyester): Crepe/1003/im_4.png; Polyester/1003/im_4.png
- `EXACT-0017` (Crepe, Satin): Crepe/1028/im_1.png; Satin/1028/im_1.png
- `EXACT-0018` (Crepe, Satin): Crepe/1028/im_2.png; Satin/1028/im_2.png
- `EXACT-0019` (Crepe, Satin): Crepe/1028/im_3.png; Satin/1028/im_3.png
- `EXACT-0020` (Crepe, Satin): Crepe/1028/im_4.png; Satin/1028/im_4.png
- `EXACT-0021` (Crepe, Satin): Crepe/1029/im_1.png; Satin/1029/im_1.png
- `EXACT-0022` (Crepe, Satin): Crepe/1029/im_2.png; Satin/1029/im_2.png
- `EXACT-0023` (Crepe, Satin): Crepe/1029/im_3.png; Satin/1029/im_3.png
- `EXACT-0024` (Crepe, Satin): Crepe/1029/im_4.png; Satin/1029/im_4.png
- `EXACT-0025` (Crepe, Satin): Crepe/1030/im_1.png; Satin/1030/im_1.png
- `EXACT-0026` (Crepe, Satin): Crepe/1030/im_2.png; Satin/1030/im_2.png
- `EXACT-0027` (Crepe, Satin): Crepe/1030/im_3.png; Satin/1030/im_3.png
- `EXACT-0028` (Crepe, Satin): Crepe/1030/im_4.png; Satin/1030/im_4.png
- `EXACT-0029` (Crepe, Satin): Crepe/1031/im_1.png; Satin/1031/im_1.png
- `EXACT-0030` (Crepe, Satin): Crepe/1031/im_2.png; Satin/1031/im_2.png
- `EXACT-0031` (Crepe, Satin): Crepe/1031/im_3.png; Satin/1031/im_3.png
- `EXACT-0032` (Crepe, Satin): Crepe/1031/im_4.png; Satin/1031/im_4.png
- `EXACT-0033` (Crepe, Satin): Crepe/1032/im_1.png; Satin/1032/im_1.png
- `EXACT-0034` (Crepe, Satin): Crepe/1032/im_2.png; Satin/1032/im_2.png
- `EXACT-0035` (Crepe, Satin): Crepe/1032/im_3.png; Satin/1032/im_3.png
- `EXACT-0036` (Crepe, Satin): Crepe/1032/im_4.png; Satin/1032/im_4.png
- `EXACT-0037` (Crepe, Satin): Crepe/1033/im_1.png; Satin/1033/im_1.png
- `EXACT-0038` (Crepe, Satin): Crepe/1033/im_2.png; Satin/1033/im_2.png
- `EXACT-0039` (Crepe, Satin): Crepe/1033/im_3.png; Satin/1033/im_3.png
- `EXACT-0040` (Crepe, Satin): Crepe/1033/im_4.png; Satin/1033/im_4.png
- `EXACT-0041` (Crepe, Polyester): Crepe/998/im_1.png; Polyester/998/im_1.png
- `EXACT-0042` (Crepe, Polyester): Crepe/998/im_2.png; Polyester/998/im_2.png
- `EXACT-0043` (Crepe, Polyester): Crepe/998/im_3.png; Polyester/998/im_3.png
- `EXACT-0044` (Crepe, Polyester): Crepe/998/im_4.png; Polyester/998/im_4.png
- `EXACT-0045` (Crepe, Polyester): Crepe/999/im_1.png; Polyester/999/im_1.png
- `EXACT-0046` (Crepe, Polyester): Crepe/999/im_2.png; Polyester/999/im_2.png
- `EXACT-0047` (Crepe, Polyester): Crepe/999/im_3.png; Polyester/999/im_3.png
- `EXACT-0048` (Crepe, Polyester): Crepe/999/im_4.png; Polyester/999/im_4.png
- `EXACT-0049` (Polyester, Satin): Polyester/770/im_1.png; Satin/770/im_1.png
- `EXACT-0050` (Polyester, Satin): Polyester/770/im_2.png; Satin/770/im_2.png
- `EXACT-0051` (Polyester, Satin): Polyester/770/im_3.png; Satin/770/im_3.png
- `EXACT-0052` (Polyester, Satin): Polyester/770/im_4.png; Satin/770/im_4.png
- `EXACT-0053` (Polyester, Satin): Polyester/771/im_1.png; Satin/771/im_1.png
- `EXACT-0054` (Polyester, Satin): Polyester/771/im_2.png; Satin/771/im_2.png
- `EXACT-0055` (Polyester, Satin): Polyester/771/im_3.png; Satin/771/im_3.png
- `EXACT-0056` (Polyester, Satin): Polyester/771/im_4.png; Satin/771/im_4.png
- `EXACT-0057` (Polyester, Satin): Polyester/772/im_1.png; Satin/772/im_1.png
- `EXACT-0058` (Polyester, Satin): Polyester/772/im_2.png; Satin/772/im_2.png
- `EXACT-0059` (Polyester, Satin): Polyester/772/im_3.png; Satin/772/im_3.png
- `EXACT-0060` (Polyester, Satin): Polyester/772/im_4.png; Satin/772/im_4.png
- `EXACT-0061` (Polyester, Satin): Polyester/773/im_1.png; Satin/773/im_1.png
- `EXACT-0062` (Polyester, Satin): Polyester/773/im_2.png; Satin/773/im_2.png
- `EXACT-0063` (Polyester, Satin): Polyester/773/im_3.png; Satin/773/im_3.png
- `EXACT-0064` (Polyester, Satin): Polyester/773/im_4.png; Satin/773/im_4.png
- `EXACT-0065` (Satin, Silk): Satin/1004/im_1.png; Silk/1004/im_1.png
- `EXACT-0066` (Satin, Silk): Satin/1004/im_2.png; Silk/1004/im_2.png
- `EXACT-0067` (Satin, Silk): Satin/1004/im_3.png; Silk/1004/im_3.png
- `EXACT-0068` (Satin, Silk): Satin/1004/im_4.png; Silk/1004/im_4.png
- `EXACT-0069` (Satin, Silk): Satin/1005/im_1.png; Silk/1005/im_1.png
- `EXACT-0070` (Satin, Silk): Satin/1005/im_2.png; Silk/1005/im_2.png
- `EXACT-0071` (Satin, Silk): Satin/1005/im_3.png; Silk/1005/im_3.png
- `EXACT-0072` (Satin, Silk): Satin/1005/im_4.png; Silk/1005/im_4.png
- `EXACT-0073` (Satin, Silk): Satin/1006/im_1.png; Silk/1006/im_1.png
- `EXACT-0074` (Satin, Silk): Satin/1006/im_2.png; Silk/1006/im_2.png
- `EXACT-0075` (Satin, Silk): Satin/1006/im_3.png; Silk/1006/im_3.png
- `EXACT-0076` (Satin, Silk): Satin/1006/im_4.png; Silk/1006/im_4.png
- `EXACT-0077` (Satin, Silk): Satin/1007/im_1.png; Silk/1007/im_1.png
- `EXACT-0078` (Satin, Silk): Satin/1007/im_2.png; Silk/1007/im_2.png
- `EXACT-0079` (Satin, Silk): Satin/1007/im_3.png; Silk/1007/im_3.png
- `EXACT-0080` (Satin, Silk): Satin/1007/im_4.png; Silk/1007/im_4.png
- `EXACT-0081` (Satin, Silk): Satin/1008/im_1.png; Silk/1008/im_1.png
- `EXACT-0082` (Satin, Silk): Satin/1008/im_2.png; Silk/1008/im_2.png
- `EXACT-0083` (Satin, Silk): Satin/1008/im_3.png; Silk/1008/im_3.png
- `EXACT-0084` (Satin, Silk): Satin/1008/im_4.png; Silk/1008/im_4.png
- `EXACT-0085` (Satin, Silk): Satin/1009/im_1.png; Silk/1009/im_1.png
- `EXACT-0086` (Satin, Silk): Satin/1009/im_2.png; Silk/1009/im_2.png
- `EXACT-0087` (Satin, Silk): Satin/1009/im_3.png; Silk/1009/im_3.png
- `EXACT-0088` (Satin, Silk): Satin/1009/im_4.png; Silk/1009/im_4.png
- `EXACT-0089` (Satin, Silk): Satin/1010/im_1.png; Silk/1010/im_1.png
- `EXACT-0090` (Satin, Silk): Satin/1010/im_2.png; Silk/1010/im_2.png
- `EXACT-0091` (Satin, Silk): Satin/1010/im_3.png; Silk/1010/im_3.png
- `EXACT-0092` (Satin, Silk): Satin/1010/im_4.png; Silk/1010/im_4.png
- `EXACT-0093` (Satin, Silk): Satin/1011/im_1.png; Silk/1011/im_1.png
- `EXACT-0094` (Satin, Silk): Satin/1011/im_2.png; Silk/1011/im_2.png
- `EXACT-0095` (Satin, Silk): Satin/1011/im_3.png; Silk/1011/im_3.png
- `EXACT-0096` (Satin, Silk): Satin/1011/im_4.png; Silk/1011/im_4.png
- `EXACT-0097` (Satin, Silk): Satin/1012/im_1.png; Silk/1012/im_1.png
- `EXACT-0098` (Satin, Silk): Satin/1012/im_2.png; Silk/1012/im_2.png
- `EXACT-0099` (Satin, Silk): Satin/1012/im_3.png; Silk/1012/im_3.png
- `EXACT-0100` (Satin, Silk): Satin/1012/im_4.png; Silk/1012/im_4.png
- `EXACT-0101` (Satin, Silk): Satin/1013/im_1.png; Silk/1013/im_1.png
- `EXACT-0102` (Satin, Silk): Satin/1013/im_2.png; Silk/1013/im_2.png
- `EXACT-0103` (Satin, Silk): Satin/1013/im_3.png; Silk/1013/im_3.png
- `EXACT-0104` (Satin, Silk): Satin/1013/im_4.png; Silk/1013/im_4.png
- `EXACT-0105` (Satin, Silk): Satin/1014/im_1.png; Silk/1014/im_1.png
- `EXACT-0106` (Satin, Silk): Satin/1014/im_2.png; Silk/1014/im_2.png
- `EXACT-0107` (Satin, Silk): Satin/1014/im_3.png; Silk/1014/im_3.png
- `EXACT-0108` (Satin, Silk): Satin/1014/im_4.png; Silk/1014/im_4.png
- `EXACT-0109` (Satin, Silk): Satin/1015/im_1.png; Silk/1015/im_1.png
- `EXACT-0110` (Satin, Silk): Satin/1015/im_2.png; Silk/1015/im_2.png
- `EXACT-0111` (Satin, Silk): Satin/1015/im_3.png; Silk/1015/im_3.png
- `EXACT-0112` (Satin, Silk): Satin/1015/im_4.png; Silk/1015/im_4.png
- `EXACT-0113` (Satin, Silk): Satin/1016/im_1.png; Silk/1016/im_1.png
- `EXACT-0114` (Satin, Silk): Satin/1016/im_2.png; Silk/1016/im_2.png
- `EXACT-0115` (Satin, Silk): Satin/1016/im_3.png; Silk/1016/im_3.png
- `EXACT-0116` (Satin, Silk): Satin/1016/im_4.png; Silk/1016/im_4.png
- `EXACT-0117` (Satin, Silk): Satin/1017/im_1.png; Silk/1017/im_1.png
- `EXACT-0118` (Satin, Silk): Satin/1017/im_2.png; Silk/1017/im_2.png
- `EXACT-0119` (Satin, Silk): Satin/1017/im_3.png; Silk/1017/im_3.png
- `EXACT-0120` (Satin, Silk): Satin/1017/im_4.png; Silk/1017/im_4.png

## Manual Review Required

- Every near-duplicate pair requires manual review before any cleaning decision.
- Every cross-class exact or near group requires manual label/provenance review.
- No duplicate was automatically removed or excluded by this script.
