# PHELE wall appearance experiment

This experiment uses actual public photographs and publisher bounding-box annotations. It does not use the four private owner videos, fabricated damage, generated video or manufacturer-tile examples. Models remain research-only; no construction category passes release qualification from this dataset.

## Acquisition and license

Source: [PHELE version 4](https://data.mendeley.com/datasets/yzsxcvc8dp/4), DOI `10.17632/yzsxcvc8dp.4`. The retrieved DataCite record declares **CC BY 4.0**. The publisher ZIP is 2,518,108,734 bytes. Its declared whole-archive SHA256 is `d250a5b7206f95783f29c47239f6d195c8ed94cbd8b88e5d96cc1a20dc537801`; this experiment used bounded HTTP ranges and **does not claim whole-archive verification**.

The ZIP inventory contains 2,503 file members. Main `PHELE/` contains 800 photographs with 800 YOLO text annotations and `classes.txt`: publisher TRAIN 560, VAL 160, TEST 80. A separate `RGB-T/` component contains visible/infrared imagery and annotations; it is excluded. Genuine polygon/segmentation masks are not supplied for the selected main PHELE tasks, so this source cannot support the requested mask-model comparison by pretending boxes are masks.

`scripts/acquire_phele_expansion.py` reuses the shared strict range reader. Subsequent image acquisition uses at most four independent bounded requests, checks exact HTTP Content-Range/size, inflates original ZIP records, verifies each uncompressed ZIP CRC32 and records original SHA256. All 801 main annotation/class files and 169 selected original photos were acquired. The combined range ledger records **522,817,622 received bytes**, below the 600,000,000-byte limit. Receipts, archive inventory and original bytes are under `datasets/construction/expansion-public-v1/phele/`.

## Actual taxonomy and visual audit

Raw `classes.txt` has 15 classes. Raw index **5 is `Peeling paint`**, and index **9 is `Mold`**. Although the record describes “Mold/dampness,” the real annotation does not distinguish damp staining from mold. Canonical classes are therefore **`peeling_paint`** and **`mold_appearance`**. No moisture percentage, damp-only diagnostic, fungal species or health diagnosis is implied. The distinct `damp_stained_appearance` capability remains unavailable without its own supporting data and evaluation.

All 800 annotations contain 172 peeling-paint boxes and 39 Mold boxes, across 139 target-positive photos. Selection includes every target-positive image and 30 filename-hash-selected photos without either target annotation. Those 30 are **target-label-absent diagnostic examples**, not independently verified healthy-wall negatives. Other source hazards include animals, furniture, steps, wires and floor hazards; absence of a label is not definitive physical absence.

The TRAIN-only contact sheet shows 32 original wall/ceiling photos with publisher boxes: dark speckled/corner patches, stained-looking areas and peeling finishes, rather than food mold. Several pictures appear to share building/room characteristics. This visual observation does not recover physical site identities or prove independence.

![TRAIN-only source audit; publisher annotations, not predictions](../output/evaluation/construction-expansion-phele-v1/train-only-source-audit.png)

## Frozen preparation and splits

`scripts/prepare_phele_expansion.py` selects examples from annotations before acquiring image content. It retains publisher splits and audits exact SHA256 plus perceptual-hash distance at most four. Duplicate groups retain TEST before CAL before TRAIN; this acquisition found no such duplicates. Camera/site identity was not supplied, and viewpoint changes can defeat image hashing, so the experiment is **not an independent-site test**.

| Split | Photos | Peeling positive photos / boxes | Mold positive photos / boxes |
|---|---:|---:|---:|
| TRAIN | 116 | 73 / 109 | 29 / 33 |
| CAL | 34 | 28 / 41 | 3 / 3 |
| TEST | 19 | 11 / 22 | 3 / 3 |

Prepared images are actual photos fitted into a 640×640 canvas with neutral padding. Coordinates are transformed consistently; no defect pixels are generated. Original/prepared image and label hashes, original dimensions, publisher split, duplicate-group identity and actual class mappings are frozen in `prepared-manifest.json` before training. CAL's three Mold-positive examples are much too few for reliable calibration; TEST is also too small for a release claim.

## Bounded detector comparison

`scripts/train_phele_expansion.py` applies the shared public-box recipe to these frozen PHELE records:

- YOLOv8n: official COCO initializer, new two-class semantic head, 384 input, frozen first four blocks, exactly eight epochs, final checkpoint. This larger fixed budget was chosen before any PHELE model fitting or model results.
- SSDLite320: official TorchVision COCO initializer, fresh three-channel head including background, frozen backbone, 200 updates of eight TRAIN examples.
- One Torch CPU thread per job, including an explicit Ultralytics CPU-thread override and actual training-thread receipts. No private video is loaded.
- Fixed threshold grid selected using CAL only; both original-initialized and final models are evaluated at 0.5, with the final model additionally evaluated at its frozen CAL point. Box matches require IoU ≥0.5; confidence is not a physical measurement.

The "before" models have newly initialized PHELE semantic heads attached to COCO features. They are **not** previously deployed paint/mold detectors; the prior application had no validated detectors for these two classes. This baseline measures training's effect, rather than implying that a crack detector previously understood wall mold.

The update budgets differ and are exploratory; this is not a controlled claim that one architecture is intrinsically better. Initializer identity, source/script hashes, frozen configuration, actual counts, before/after results, original predictions and ONNX assets are recorded with the experiment. Inference speed measured during concurrent training is labeled contended; deployment speed requires isolated local-adapter/runtime measurement.

Operational targets remain at least 95% precision and 90% recall with 50 independent positive and 50 negative examples per released category. This dataset's small photo-only test and unavailable physical grouping cannot satisfy that gate even if a numerical photo score looks high.

## Measured YOLO before/after: quality failed

Eight fixed epochs completed in **382.7 seconds**. The actual training callback confirmed Torch one thread and the Ultralytics CPU-thread constant one. No checkpoint was chosen using TEST or private footage; the exact last checkpoint was exported.

| Checkpoint / threshold | Class | TP | FP | FN | Precision | Recall | Target-label-absent photos flagged |
|---|---|---:|---:|---:|---:|---:|---:|
| Fresh head / 0.50 | Peeling paint | 1 | 3 | 21 | 25.0% | 4.5% | 1/8 |
| Fresh head / 0.50 | Mold appearance | 0 | 9 | 3 | 0% | 0% | 4/16 |
| Final / 0.50 | Peeling paint | 0 | 0 | 22 | No predictions | 0% | 0/8 |
| Final / 0.50 | Mold appearance | 0 | 0 | 3 | No predictions | 0% | 0/16 |
| Final / CAL-selected 0.05 | Peeling paint | 6 | 40 | 16 | 13.0% | 27.3% | 4/8 |
| Final / CAL-selected 0.05 | Mold appearance | 0 | 1 | 3 | 0% | 0% | 1/16 |

At the fixed 0.5 point the final model merely abstains; this is not successful false-alarm reduction while retaining recall. CAL itself has only 4 TP, 130 FP and 40 FN at its chosen 0.05 point, so that low point is explicitly a failed research operating point, not an acceptable deployment threshold.

The **actual shipped CPU serving adapter** reproduces the final selected TEST TP/FP/FN counts. Its six matched peeling boxes have mean IoU **0.6622**; all 40 unmatched peeling boxes remain false positives. These are box labels and box predictions, not precise damage masks.

Export checks pass: ONNX checker, no external tensor data, model SHA256 `cb6e8fe431df9dfc93cee642f79db3770f8999782e611e6158d7c790c86f9127`, fixed TRAIN-image native Torch versus CPU-ONNX maximum absolute error **0.000748** against tolerance 0.002 (relative 0.0001). Numerical export parity does not rescue failed defect accuracy.

Contended single-frame CPU adapter timing on all 19 real TEST photos: **9.17 inspected fps**, median **108.1 ms**, p95 **125.6 ms**, including preprocessing/inference/NMS and excluding disk decode. Other training was running, so this is not isolated hardware acceptance and is not preview fps. The host must report preview/inspection separately.

Evidence and raw records:

- `output/evaluation/construction-expansion-phele-v1/yolov8n-results.json`: exact baseline/final/CAL/TEST counts.
- `yolov8n-predictions.json`: saved predictions and frozen member order.
- `yolov8n-served-box-evaluation.json`: every publisher box, actual CPU-adapter prediction and IoU match.
- `yolov8n-cpu-onnx-parity.json`: numerical export checks.
- `test-photo-comparisons/README.md`: **all 19** real TEST photographs, original/publisher-box/before/after panels. These are photographs, not videos or selected success examples.

The model manifest is `models/construction/expansion-phele-v1/yolov8n/manifest.json`. It is research-only and is not promoted as a working paint/mold detector.

## Measured SSDLite before/after: some real localization, quality failed

SSDLite completed all **200 updates in 196.0 seconds**, one Torch thread. Mean TRAIN loss fell from **7.603** over the first ten updates to **2.739** over the last ten; this is optimization evidence, not proof of reliable detection. Its frozen CAL-only confidence point is **0.35**.

| Checkpoint / threshold | Class | TP | FP | FN | Precision | Recall | Target-label-absent photos flagged |
|---|---|---:|---:|---:|---:|---:|---:|
| Fresh head / 0.50 | Peeling paint | 0 | 0 | 22 | No predictions | 0% | 0/8 |
| Fresh head / 0.50 | Mold appearance | 0 | 0 | 3 | No predictions | 0% | 0/16 |
| Final / 0.50 | Peeling paint | 0 | 1 | 22 | 0% | 0% | 0/8 |
| Final / 0.50 | Mold appearance | 0 | 0 | 3 | No predictions | 0% | 0/16 |
| Final / CAL-selected 0.35 | Peeling paint | 4 | 9 | 18 | 30.8% | 18.2% | 2/8 |
| Final / CAL-selected 0.35 | Mold appearance | 1 | 0 | 2 | 100% on **one** prediction | 33.3% | 0/16 |

CAL at 0.35 has peeling **9 TP / 26 FP / 32 FN**, Mold **0 TP / 0 FP / 3 FN**. The single Mold match on TEST does not establish generalization, a safe precision bound or humidity inspection. Neither category meets the release requirements.

Actual CPU serving reproduces the exact TEST TP/FP/FN totals. Matched-box mean IoU is **0.6863** for peeling and **0.9089** for the sole Mold match. Visual review confirms actual stained wall localization in `test_48`, plus two genuine peeling-paint patches in `test_35`; that same ceiling photo contains a false alarm on the light fixture and multiple missed peeling patches. All 19 photos are retained, including failures.

![Actual stained-wall match, alongside original and publisher annotation](../output/evaluation/construction-expansion-phele-v1/ssdlite320-test-photo-comparisons/test_48.png)

![Peeling matches, missed patches and false alarm on ceiling light](../output/evaluation/construction-expansion-phele-v1/ssdlite320-test-photo-comparisons/test_35.png)

The native Torch/CPU-ONNX check passes on fixed TRAIN pixels: box maximum absolute error **0.000183**, score maximum error **0.000000387**, identical labels. The ONNX checker passes, external tensor data is absent, and model SHA256 is `043cd896209071d3f2607eaa5a061c2b2922df046262151ddf6b9ad5a2b2c40b`.

Contended CPU adapter timing over all 19 TEST photographs is **15.35 inspected fps**, median **64.5 ms**, p95 **78.9 ms**, including preprocessing/inference/NMS and excluding disk decode. This is a measured research-adapter result under possible concurrent training, not sustained isolated 15+ fps qualification or a preview benchmark. The two model runs also use different input sizes and fitting budgets.

Additional artifacts:

- `models/construction/expansion-phele-v1/ssdlite320/manifest.json`: local research asset and exact source-class mapping.
- `output/evaluation/construction-expansion-phele-v1/ssdlite320-results.json`: all fixed-point and CAL-selected before/after counts.
- `ssdlite320-served-box-evaluation.json`: every actual serving prediction and publisher-box match.
- `ssdlite320-cpu-onnx-parity.json`: numerical export evidence.
- `ssdlite320-test-photo-comparisons/README.md`: every TEST photograph, not just the illustrated examples. Before panels show no predictions because the measured fresh-head baseline has zero predictions at 0.5.

Both neural architectures were genuinely trained on real public examples and exported for the local application. The result is **usable research integration with failed accuracy gates**, not completed production-grade wall-fault detection. No private owner clip was used in fitting, calibration or the results above.
