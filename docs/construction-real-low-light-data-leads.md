# Genuine low-light crack data: bounded primary-source review

20 October2026. This is source/metadata research for a future experiment, after the original crack-model gain comparator failed. No images, archives, checkpoints or test splits were downloaded; no model was loaded, source pixels inspected, training changed or inference run. Existing calibration and Jan2 heldout remain protected.

## Verified real capture, but not a ready new night-training split

**LCSD is a small, genuinely captured low-light segmentation lead.** The [CrackNex authors' paper, dataset section](https://arxiv.org/html/2403.03063v1#S4.SS1.SSS2) describes102 well-lit training photographs and41 real low-light test photographs taken on the Lehigh campus with an iPad Pro, resized400x400 and manually pixel annotated. The same paper distinguishes `ll_CrackSeg9k`, whose dark test images are generated with Restormer; those synthetic images must not be represented as real night acquisition. LCSD is real low light, but the paper does not establish that every dark image was captured outdoors at night, nor provide physical-scene independence. It is still imagery, not recorded construction video.

The [official code repository](https://github.com/zyaocoder/CrackNex) links the [LCSD data repository](https://huggingface.co/datasets/Chrisathy/LCSD/tree/main) directly. Its public file browser exposes actual `train/image`, `train/mask`, `val/image`, `val/mask`, `train.txt` and `val.txt`; total stated size7.6MB, revision shown `254e9a9`. The linked [dataset card](https://huggingface.co/datasets/Chrisathy/LCSD) declares **Apache-2.0**. This is the author-linked distribution declaration, not an independent clearance of unlisted third-party material. The auto-generated viewer treats image/mask folders as image classes and reports286 rows; that is **not286 independent crack photographs**. A future loader must pair original filenames and masks using the split lists.

Crucially, **the41 low-light examples are the original evaluation partition**. Loading them into a new training set while reporting original LCSD test accuracy would leak evaluation data. A separately named adaptation experiment could deliberately repartition only after source/scene/duplicate auditing and preregistration, but it would relinquish that original benchmark interpretation. This source currently closes neither the genuine night-video TRAIN gap nor the independent field-video evaluation gap.

## Other primary-source leads and actual availability

| Source | Verified capture/GT claim | Asset and rights limits | Suitability now |
|---|---|---|---|
| [CrackTrans authors' repository](https://github.com/JZhongUTK/Pavement-crack-dataset) |455 smartphone pavement images with GT, including shadow, wet, night and grooved backgrounds | Linked Drive page resolves as `CrackTrans.zip`; payload not downloaded. Author says public use, but no explicit standard dataset license, night count, scene split or camera sequence is established in the README. Research title concerns augmentation; exact released real/generated membership needs auditing. | Real-night acquisition lead, **not yet a cleared night TRAIN dataset**, and pavement differs from building walls. |
| [RoadCrack-MM-2025 authors](https://www.mdpi.com/2072-4292/17/20/3488) | Actual June2025 handheld thermal+visible night pavement pairs; annotated cracks generally >.5cm wide | Data availability says authors on request. No immediately downloadable licensed data release was verified. Requires thermal modality for the published fusion system; article license does not itself establish downloadable dataset rights. | Fails present free/no-contact asset requirement; large pavement cracks do not substitute for tiny construction cracks. |

No account, author contact, paid service or installation was used. General low-light image-enhancement datasets without crack GT are not a substitute for crack localization training/validation.

## Existing ev-CIVIL is the most direct recorded-video lead

The [publisher ev-CIVIL release](https://doi.org/10.11583/DTU.25924228) is already documented locally as CC BY4.0 with genuine camera exposure H5 files and publisher timestamped labels. A metadata-only check of the preserved [ZIP inventory](C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/evcivil/field-member-inventory.json), SHA256 `9725f2160fba9b71e6d451ffb97e1d91223737c811e09f8bf311c87d3df01183`, gives:

| Publisher night group | Camera-H5 sequences | Compressed camera-H5 bytes | Current constraint |
|---|---:|---:|---|
| Dec31 |28|82,460,458| Potential future TRAIN source; aliases and actual darkness unverified |
| Dec27 |21|16,164,669| Some existing TRAIN captures; do not count alternate captures as new physical defects |
| Nov18 |17|133,655,994| Preserve entire CAL capture group |
| Jan2 |3|1,921,561| Preserve entire heldout capture group and global IDs32/43/44 |
| Night spalling/March |13|23,481,096| Potential real dark-spall source, but numeric date prefixes are not defect IDs |

The Dec31 folder contains `31_crack_21` (CAL global-ID conflict) and `31_crack_32`/`31_crack_43` (heldout conflicts), among others. Thus **28 is an inventory count, not28 accepted new TRAIN defects**. All current CAL/heldout source groups, canonical IDs/aliases and original exposed IDs must remain excluded. Any safe alternative capture of an existing TRAIN defect stays inside TRAIN and counts as another view, not a physically independent defect/site.

A future acquisition proposal should freeze a metadata-only source list first, then inspect actual native gray-p99 distributions, qualified original exposures, publisher timestamp association, all frame GT and duplicates before acceptance. `Night_outdoor` naming alone does not prove the low-light gain trigger occurs: the coordinator's existing TRAIN audit found20 night-labeled TRAIN views but none actually received gain; CAL had32 gain-applied views. Sampling dark-looking outcomes or using CAL as training would not fix that gap honestly. Freeze accepted future training data and preserve the existing evaluation inputs before fitting anything new.

This review proposes a source direction only. It changes no active dataset, source labels, model, exposure rule or confidence threshold, and claims no added detection capability or Saudi-code compliance.
