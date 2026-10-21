# SDNET actual-polygon segmentation comparison

This experiment compares a small semantic segmenter with an instance segmenter on real public steel-connection photographs. It is separate from the factory work and from the old concrete/crack models. Neither source labels nor fitted weights establish fastening torque, concealed anchorage, or universal building-site performance.

## Frozen source data

Source: [SDNET2025, Mendeley Data version 1](https://data.mendeley.com/datasets/cxnj6cp4rb/1), CC BY 4.0. The prepared dataset uses one unmarked publisher 640×640 derivative per original photograph. Annotated overlays, augmented copies and raw images without verified filename correspondence are excluded.

| Split | Unseated-connection photos | Missing-fastener appearance photos | Fixed/negative photos | Total |
|---|---:|---:|---:|---:|
| TRAIN | 224 | 142 | 235 | 601 |
| CAL | 61 | 29 | 37 | 127 |
| TEST | 40 | 28 | 30 | 98 |

The actual archive contains **970 genuine polygon annotations**, comprising 533 labelled loose/unseated connections and 437 labelled missing-fastener appearances. Every annotation used here has one polygon. Masks are rasterized from those polygons, never from rectangular boxes. Publisher descriptions and paper counts must not override the actual audited annotation counts.

The frozen manifest SHA256 is `548d4321d87924582e9f7bb30cd5946d5c47a53c9b8b15a77a9265267df4e117`. Its 780 exact/near-duplicate groups were assigned before fitting. These groups do **not** establish distinct physical connections or sites; the publisher does not supply the required identifiers. Consequently this is research-only evaluation, and 40/28 positive TEST photographs also fall short of the 50-positive-per-category release gate.

The four private construction videos are not used for training, calibration, model selection or this test benchmark. Photographic evidence remains identified as photographs, never presented as continuous video.

## Fixed training and evaluation protocol

- [TorchVision MobileNetV3 LR-ASPP](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.lraspp_mobilenet_v3_large.html): clean official COCO/VOC initializer, new three-channel background/defect heads, frozen backbone and fixed 200 AdamW updates with batch eight at 320×320. Each batch contains three unseated, three missing-appearance and two fixed photographs. Loss is class-weighted cross entropy `[0.2,1,1]` plus half-weight foreground Dice.
- YOLOv8n-seg: clean official COCO segmentation initializer, new two-class configuration, two fixed epochs at 320×320, batch eight, freeze ten layers, and no image augmentation. Labels preserve the genuine polygon vertices. This has a different computation budget from LR-ASPP; training times are reported rather than claiming equal compute.
- Final checkpoints are fixed by the declared budgets. Per-class thresholds are selected only on CAL from `[0.05,0.10,0.20,0.35,0.50,0.65,0.80,0.90]`: prefer nonempty mask-region precision ≥95%, then recall; otherwise region F1, pixel F1 and precision. TEST does not choose checkpoints or thresholds.
- Report the untrained task-head baseline and trained checkpoint at the same fixed 0.5 threshold, plus the after-training CAL-selected operating point. Baseline generic COCO pretraining does not supply these defect classes.
- LR-ASPP evaluation matches the runtime: RGB/ImageNet normalization, 320 logits, softmax, native-resolution probability resize, background/class argmax, class threshold and eight-pixel minimum native connected-component area. Instance matches require same-class **mask IoU ≥0.5**, not just a box enclosing the defect. Pixel precision/recall/IoU and negative-photo alarms are separate metrics.
- Record region precision/recall denominators and Wilson intervals. Intervals assume independent Bernoulli trials and are descriptive only because physical-site independence is unverified. Pixels are correlated; do not attach independent-pixel confidence claims.
- CPU thread limits, preprocessing, inference/postprocessing scope and concurrent jobs are part of each speed receipt. Model-forward speed alone is not the application's inspected-video FPS.

## Reproduction and outputs

Run with the local CPU training environment:

```powershell
.venv-roads-export/Scripts/python.exe scripts/train_sdnet_masks_expansion.py --model lraspp
.venv-roads-export/Scripts/python.exe scripts/train_sdnet_masks_expansion.py --model yolo
```

Models and initializer provenance are kept under `models/construction/expansion-sdnet-masks/`. Protocols, calibration decisions, test results, timing, ONNX parity and fixed-order real-photo evidence are kept under `output/evaluation/construction-expansion-sdnet-masks/`. Each run verifies the frozen image and mask hashes before training. Research manifests explicitly identify `brackets_railings`, preprocessing, exact ONNX SHA256 and calibration thresholds. No category is automatically released or selected in the UI.

The commands above record the original invocation. Existing trained experiment artifacts are protected against accidental overwrite; use a new experiment version for another fit. Exact source snapshots for both completed fits are archived with their protocols. Later verifier/guard repairs do not change the recorded training-source hashes or model weights.

Verification includes Python compilation of the training/deployment-check scripts, both manifests accepted by the serving environment's hash/schema validator, and a deliberate repeated `--model yolo` invocation rejected before fitting with the preservation guard. The rejected invocation is an expected safeguard check, not a model-training failure.

## LR-ASPP measured results

The fixed 200-update fit completed in **317.46 seconds**, using 1,600 sample exposures covering 541 distinct TRAIN photographs. Mean loss fell from 0.912 for the first ten updates to 0.353 for the last ten. Both CAL-selected thresholds were 0.5, so selected and fixed-point after-training metrics coincide.

| Category | Stage | Mask-region TP / unmatched / FN | Region precision / recall | Pixel precision / recall / IoU | Flagged category-negative images |
|---|---|---|---|---|---|
| Unseated connection | Untrained task heads, 0.5 | 0 / 3,213 / 65 | 0% / 0% | 0.23% / 2.59% / 0.21% | 58 / 58 |
| Unseated connection | Trained, CAL 0.5 | 14 / 84 / 51 | 14.29% / 21.54% | 48.75% / 47.91% / 31.87% | 15 / 58 |
| Missing-fastener appearance | Untrained task heads, 0.5 | 1 / 1,202 / 70 | 0.08% / 1.41% | 0.37% / 5.44% / 0.35% | 64 / 70 |
| Missing-fastener appearance | Trained, CAL 0.5 | 10 / 64 / 61 | 13.51% / 14.08% | 30.91% / 44.35% / 22.27% | 17 / 70 |

These are **annotation-matched research metrics**, not verified physical false-alarm rates. The publisher's class-specific files do not establish exhaustive annotations for the opposite category. Manual inspection of TRAIN `Loosen-10` shows that an unannotated large hole attracts a missing-fastener candidate, while TRAIN `Missing-1` labels visible connection seats/nut-absence appearances rather than a universal empty-hole rule. Intended holes and required fastening layouts need a reference. The class-negative denominators above include opposite-category images under the original frozen evaluation, so they must not be described as independently confirmed healthy installations. Fixed-only negatives and a scope-aware view are reported separately when available. No thresholds are changed in response to this audit.

One-thread Torch timing across 98 TEST photos was 5.82 FPS before and 9.76 FPS after, including image decode, normalization, network and native probability resize, excluding connected-component extraction. These runs experienced different concurrent CPU load; this is **not evidence that training accelerated the architecture**. Ten TRAIN samples gave 13.48 FPS for ORT logits inference alone, with first-call cost included. Neither value is a sustained application/video qualification or a ≥15-FPS inspection claim.

The frozen-point deployment check uses the actual `ConstructionTaskDetector` ONNX adapter, with the same weights and CAL thresholds, including native-mask decoding. It reproduces the same 14/10 region matches and 51/61 misses. Across all 98 photos its measured complete-adapter rate was **5.57 FPS**, p95 **230 ms**, during other training activity; this is distinct from the independent raw preview.

| Scope-aware deployed view | Region TP / unmatched / FN | Precision / recall | Pixel IoU | Publisher-Fixed negatives flagged |
|---|---|---|---|---|
| Unseated connection: own-category photos + Fixed | 14 / 67 / 51 | 17.28% / 21.54% | 33.69% | **6 / 30 (20%)** |
| Missing appearance: own-category photos + Fixed | 10 / 59 / 61 | 14.49% / 14.08% | 22.35% | **13 / 30 (43.33%)** |

The scope-aware view omits 28 opposite-category photos for unseated connections and 40 for missing appearance because cross-category annotation completeness is not established. It is an additional diagnostic after the protocol was frozen, not an alternative threshold-selection result. All-category original metrics remain preserved. There are 12 unseated proposals and 29 missing-appearance proposals on the 30 publisher-Fixed photos. These alarm levels remain unacceptable for automatic inspection. Full denominators and descriptive Wilson intervals are saved in `lraspp-runtime-frozen-point.json`.

### Export verification repair

The first export's verification failed because a newly constructed wrapper defaulted to training mode. Torch export restored that wrapper's `train(True)` recursively after exporting, causing the following Torch reference to use training-mode BatchNorm while ONNX correctly used evaluation mode. A controlled TRAIN-only reproduction switched 260 modules into training mode and produced a 7.7786 maximum logit difference. Exporting the wrapper in eval mode and explicitly asserting every reference module is in eval mode fixes the verifier.

The original checkpoint, ONNX, thresholds and numerical tolerance were **unchanged**. Evaluation-mode comparison on the same TRAIN source had max error `1.67e-5`; disabling MKLDNN gave `2.00e-5`, and disabling ORT graph optimizations gave `1.57e-5`. All ten fixed TRAIN parity cases passed the original `atol=0.003, rtol=0.003` guard, with maximum error below `1.86e-5`. The original failed run log/source, pending manifest and original model hashes are preserved alongside the diagnosis. No TEST accuracy or calibration was used to choose the repair.

Twelve fixed-order TRAIN photographs provide real-image, publisher-polygon and research-prediction panels, with source/prediction pixel receipts. They are training diagnostics, not unseen performance proof or video. The supplied four private videos remain outside the experiment.

The model fails release accuracy and independent-data requirements and remains explicitly research only.

## YOLOv8n-seg measured results

The fixed two-epoch fit and end-of-training CAL validation completed in **565.13 seconds**. Explicit callbacks verified one Torch intra-op thread, one inter-op thread and the patched Ultralytics CPU thread constant before prediction, at training start, after both epochs and after training. The genuine polygon-label dataset contains the same frozen 601 TRAIN photographs; no new sources, masks or images were fabricated.

| Category | Stage | Mask-region TP / unmatched / FN | Region precision / recall | Pixel precision / recall / IoU |
|---|---|---|---|---|
| Unseated connection | Untrained task heads, 0.5 | 0 / 28 / 65 | 0% / 0% | 0.91% / 6.33% / 0.80% |
| Unseated connection | Trained, fixed 0.5 | 0 / 0 / 65 | No predictions / 0% | 0% / 0% / 0% |
| Unseated connection | Trained, CAL 0.10 | 17 / 72 / 48 | 19.10% / 26.15% | 27.35% / 36.39% / 18.51% |
| Missing-fastener appearance | Untrained task heads, 0.5 | 0 / 4 / 71 | 0% / 0% | 0.06% / 0.45% / 0.06% |
| Missing-fastener appearance | Trained, fixed 0.5 | 0 / 0 / 71 | No predictions / 0% | 0% / 0% / 0% |
| Missing-fastener appearance | Trained, CAL 0.35 | 0 / 0 / 71 | No predictions / **0%** | 0% / 0% / 0% |

The selected missing-appearance operating point misses every TEST instance. Zero proposals do not establish successful inspection or acceptable precision. The mask-region recall gain for unseated connections comes with substantial unmatched proposals; it fails the release gate. These metrics use the original all-category annotation view and retain the annotation-completeness limitations described above.

Single-image Torch/Ultralytics prediction, including decode, letterbox, NMS and retina native masks, measured **1.69 FPS before / 4.02 FPS after**, with p95 897.5 / 425.8 ms. Other CPU jobs and first-call costs differ, so these values are a receipt, not evidence of speed optimization. Model-forward timings from the built-in validator are excluded from the application-speed claim.

### YOLO deployment parity

The ONNX has raw shapes `(1,38,2100)` and `(1,32,80,80)` at 320×320 for the two classes plus 32 mask coefficients. Twelve fixed TRAIN photos passed the unchanged raw-output tolerance `atol=0.003, rtol=0.003`; maximum box/class/coefficient error was `0.00139` and prototype error was below `2.4e-5`. CPU execution was one thread and the decoder source stayed unchanged during the check.

Native candidate counts matched on those sources. Native masks are **not byte-identical**: rounded runtime box cropping versus continuous Torch cropping changes up to 66 pixels among the inspected cases. The worst nonempty class-union IoU was 0.98751. Empty outputs for both implementations are recorded as empty comparisons, not affirmative evidence that the missing-appearance class works. The separate deployed frozen-point TEST receipt measures the actual adapter's decisions; no threshold is changed to repair a crop-boundary difference.

The complete deployed ONNX adapter reproduces **17 unseated matches / 48 misses**, and **0 missing-appearance matches / 71 misses**. On the 30 publisher-Fixed negatives, unseated candidates flag **10 photos (33.33%)**, producing 15 proposals. Missing-appearance emits nothing on the negatives or positives. Omitting opposite-category photos from the diagnostic scope gives unseated precision **23.29%**, recall **26.15%** and pixel IoU **21.04%**. The complete adapter measured **9.08 FPS**, p95 **126.0 ms**, during concurrent training; this is not an isolated video or preview benchmark. Exact counts, scope and decoder SHA256 are in `yolov8n_seg-runtime-frozen-point.json`.

Both mask architectures are retained for explicit local research selection, not automatically promoted. The newer instance segmenter has slightly higher unseated mask-region recall at its CAL-selected point, while LR-ASPP has better pixel overlap and detects some missing-appearance annotations. Neither meets the required precision, recall or independent data gates.
