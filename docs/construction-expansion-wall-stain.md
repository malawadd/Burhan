# Wall stain/leakage appearance: measured research failure

An actual SSDLite320 model was fitted to public MBDD2025 publisher `leakage` boxes and exported for local CPU inference. Its canonical class is `damp_stained_appearance`: visible stain/leakage appearance, **not measured humidity, active hidden water, mold identification or a diagnosed cause**. The measured quality fails; this model is not promoted as reliable wall inspection.

## Real data and frozen split

Source: [MBDD2025 publisher record](https://zenodo.org/records/15622584), CC BY 4.0. Original source photographs/XML boxes and CRC/hash receipts are retained. Prepared photographs fit within a 640-square neutral canvas with transformed publisher boxes; no damage pixels are generated. This source provides boxes for the task, not precise damage masks.

| Split | Photos | Target-positive photos | Other-defect target-negative photos |
|---|---:|---:|---:|
| TRAIN | 181 | 40 | 141 |
| CAL | 48 | 11 | 37 |
| New photo TEST | 98 | **47** | **51** |

The 47 positive TEST photos contain **125 publisher leakage boxes**. The 51 negatives contain other annotated defects and lack the publisher leakage label; they are **not verified healthy walls**. Reported negative-image flagging measures this specific diagnostic set, not healthy-wall field false-alarm probability.

Prior eligible TRAIN/CAL roles were retained. A fixed filename-hash selection chose 60 previously unused target-positive and 60 other-defect target-negative photos before acquisition. Exact/perceptual comparison quarantined 22 test candidates resembling prior TRAIN/CAL views, leaving 98; there was no model-result-based backfill. Protected earlier diagnostic capture blocks and all four private owner videos were excluded.

Physical wall/building/site identity remains unknown. The source's coarse capture blocks may overlap old TRAIN/CAL and the new image test; appearance hashing does not prove physical-site independence. This is exploratory photo evidence, not an independent building/site/video evaluation. The frozen prepared manifest SHA256 is `c18b298dd0227ebc08019165a60f95214edfaaa4f9f38c006fbd7d485eff6991`.

## Actual training and operating points

The official TorchVision COCO initializer was loaded, its foreground head replaced with **two channels total: background + one new stain-appearance class**, and the backbone frozen. The fixed recipe ran **200 AdamW updates**, batch 8, one CPU thread, seed 20261021, in **193.1 seconds**. Mean TRAIN loss fell from 7.355 over the first ten updates to 2.330 over the last ten; this does not establish detection quality.

The final checkpoint was used. **Confidence 0.25 was selected on CAL only and remains unchanged in the application manifest.** The fixed 0.5 comparison was already part of the frozen before/after protocol; it is reported separately and is not a replacement operating point selected from TEST. No fitting, checkpoint selection or threshold retuning was performed during this verification.

## Before/after counts: failed accuracy

Box matches require IoU ≥0.5. The before baseline is an uncalibrated newly initialized semantic head on COCO features, not a previously deployed humidity detector.

| Evaluation | Point | TP | FP | FN | Precision | Recall | Other-defect negative photos flagged |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fresh head, native float boxes | Fixed 0.50 | 22 | 15,960 | 103 | 0.14% | 17.6% | 51/51 |
| Final, native float boxes | Fixed 0.50 | 8 | 4 | 117 | 66.7% | 6.4% | 0/51 |
| Final CAL, native float boxes | Selected 0.25 | 5 | 1,026 | 17 | 0.48% | 22.7% | 37/37 |
| Final TEST, native float boxes | Selected 0.25 | 53 | 1,918 | 72 | 2.69% | 42.4% | 51/51 |
| **Actual common CPU adapter**, integer boxes | **Unchanged 0.25** | **54** | **1,917** | **71** | **2.74%** | **43.2%** | **51/51** |
| Actual adapter, predeclared comparison | Fixed 0.50 | 8 | 4 | 117 | 66.7% | 6.4% | 0/51 |

The 0.5 point suppresses most detections while missing 117/125 boxes. The selected 0.25 point floods every target-negative photo. Neither result approaches 95% precision / 90% recall, and 47 positive photos also fall below the minimum 50-positive release count even before physical independence is considered.

The actual-adapter fixed-0.5 row was calculated by filtering already-saved 0.25 predictions; no new model/threshold selection occurred. All predictions, input hashes and counts are retained in the supplemental verification artifacts.

### Why native and served totals differ by one

A dedicated audit holds the ONNX outputs, scores and 0.25 threshold fixed and changes only box-coordinate quantization. On `Hefei11014.jpg` a prediction scores **0.26635471**:

- Float bbox `[343.1139,272.3424,55.1883,75.5734]` has IoU **0.490734** against the publisher box.
- The serving adapter's outward integer bbox `[343,272,56,76]` has IoU **0.501037**.

That single borderline box crosses the IoU match threshold, reproducing exactly the native 53/1,918/72 and served 54/1,917/71 counts. This is not a new defect found through improved model quality, a changed NMS rule or a reason to claim higher accuracy. Both counts remain reported.

## Fresh-model export verification

Verification reconstructed a **new** SSDLite instance from the saved state, installed the correct two-channel head, called `eval()` explicitly and checked every child module is in evaluation mode. It did not reuse the in-memory export wrapper, whose training-mode restoration can otherwise confuse a later reference forward.

Three fixed TRAIN inputs—`Hefei3368`, `Hefei5663`, `Hefei9808`—were pinned in a protocol before verification forward. ONNX checker passes, no external tensor data is present, and fresh native Torch versus CPU-ONNX passes for every input/output. Maximum box error is **0.000473 pixels**, maximum score error **4.92×10⁻⁷**, and output labels are identical. Tolerance is absolute 0.002 / relative 0.0001; both engines use one thread. Numerical parity verifies the failed model was exported faithfully; it does not rescue accuracy.

| Artifact | SHA256 |
|---|---|
| ONNX model | `654384b6653b055583fe2b2a5e73920795ff5d2eb7844d988fd6c9b813649505` |
| Saved native state | `0d378c396a9008c4a876ec7a6c79319cd44578745ba89180a24115ccb5b81cc9` |
| Application model manifest | `9dfe325e0e12d01668aa22b402fca7e6955b6b89d3d87218a682a176ac503569` |
| Original frozen result | `0fb708bd85936d1967d07d98cb5bf3dbb140e36b9c064cb46b0125f921ea7ff6` |

Actual-adapter timing over 98 real photos: median **62.2 ms**, p95 **85.7 ms**, including preprocessing/inference/postprocessing and excluding disk decode. Other training may be active; this is not isolated sustained preview/inspection acceptance. Owner-video and application runtime measurements are separate and must not be inferred from these photo counts.

## Preserved evidence and wording correction

The generic shared helper's frozen `release_blocker` mistakenly mentions “physical-connection/site grouping.” The original result is preserved byte-for-byte. `verification-v1/supplemental-notes-v1.json` records its hash and corrects this to **unknown wall/building physical-site identity, inadequate independent validation and failed measured quality**. This corrects prose only; it changes no result, model, data or threshold.

- Model: `models/construction/expansion-wall-stain-v1/ssdlite320/manifest.json`.
- Data: `datasets/construction/expansion-public-v1/wall-stain/prepared/manifest.json`.
- Frozen before/after results: `output/evaluation/construction-expansion-wall-stain-v1/ssdlite320-results.json`.
- Verification: `output/evaluation/construction-expansion-wall-stain-v1/verification-v1/` contains the before-forward parity protocol, fresh-model parity, full common-adapter 98-photo audit, exact rounding-drift audit and versioned supplemental note.
- Reproduction scripts: `verify_wall_stain_expansion.py`, the common `audit_construction_task_serving.py`, `audit_wall_stain_quantization.py`, and `summarize_wall_stain_verification.py`. None trains or selects a model on TEST.

## Actual owner damp-wall video

The frozen 0.25 model was run on every decoded original frame of private `VID-20261020-WA0002.mp4`: 330 input and 330 H.264 output frames at the source timing, no generated frames or loop. Offline inference/rendering took 46.39 seconds under other CPU work; this is not a live throughput benchmark. The source and decoded-frame hashes, source timestamps, capture gates and predictions are preserved in `output/evaluation/construction-guided-expansion/owner-damp-stain-ssd-v1/report.json`.

Every frame contains candidates (330/330). This does not mean 330 correct detections. At the fixed midpoint, overlapping proposals cover large parts of the view and fail to provide a useful localized assessment. The owner states the wall has a humidity fault; no independent frame-level mask or instrument measurement was supplied, and this video never entered fitting or threshold selection.

![Fixed midpoint of actual owner video: original left, cluttered failed research predictions right](../output/evaluation/construction-guided-expansion/owner-damp-stain-ssd-v1/frame-00165.jpg)

The [complete current original/model recording](../output/evaluation/construction-guided-expansion/owner-damp-stain-ssd-v1-native-pts/original-vs-research.mp4) is registered in the app as **Owner damp wall / stain SSDLite - FAILED research**, under failure diagnostics, with its matching original source preset. All candidates remain unscored. Owner footage remains local/private; no redistribution license is inferred.

The initial artifact encoded all real frames at the nominal average FPS. Source timestamps differ from that nominal schedule by up to42.78ms because the phone clip has variable frame intervals. The current artifact preserves each decoded source presentation timestamp relative to the first frame; all330decoded output timestamps match within**0.000445ms**. Its offline render took60.58seconds under concurrent work, not a live speed result. A correction receipt verifies every input-frame hash, label, score and box equals the initial result. The earlier artifact/report and registry snapshot are retained. The app's existing diagnostic entry now points to the corrected-timing artifact; its model, confidence, source preset and failed assessment are unchanged.
