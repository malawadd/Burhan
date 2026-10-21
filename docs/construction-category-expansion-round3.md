# Construction damage category expansion, round three

20 October 2026. The user requested actual mistakes/damage, useful localization and additional conditions beyond cracks/spalling. A dataset label, checkpoint download or source-authored ground-truth box is **not** a measured local detection. This study keeps acquisition, working model implementation, per-class measurements and original video evidence separate. No Saudi code acceptance or structural severity is inferred.

## Selected public source and actual assets

The [dacl10k authors' toolkit](https://github.com/phiyodr/dacl10k-toolkit) publishes a real bridge-inspection photo dataset, multilabel polygons and **CC BY-NC 4.0** data terms. Its 19 ordered channels include 13 named damage classes and six object categories. Relevant labels include rust, exposed reinforcement, wet spots, efflorescence, rockpockets, cavities, spalling and cracking. The same pixel may have several labels. A softmax/argmax conversion would silently discard this property.

The [authors' detailed class descriptions](https://arxiv.org/html/2309.00460v1) clarify two easily misleading labels. **Hollowareas are closed inspector chalk/crayon markings after acoustic sounding**, not visually observable internal voids. The corresponding capability is therefore named "Inspector-marked hollow-area boundary (chalk)"; no internal-delamination detection is claimed. **WConccor** combines visually similar washout, concrete corrosion and planar erosion/abrasion; later descriptions call this Concrete Corrosion. The internal canonical alias `washout` is retained for compatibility, while its user-facing name is **Concrete surface loss / erosion appearance**. It does not distinguish water erosion, frost-thaw, chemical attack or other mechanisms.

The official AWS development archive returned HTTP 403 in this session. The authors link a [Voxel51 dataset explorer/mirror](https://huggingface.co/datasets/Voxel51/dacl10k), which was available. The mirror card incorrectly declares CC BY 4.0; the original authors' **CC BY-NC 4.0** terms are retained. Pinned mirror commit: `7aa4a3e6fd8818d94bdf51b9d02ea0443808f68b`. Its 64,117,047-byte `samples.json` has SHA-256 `329e842ee83c700c3717b629ea199faa7a9d3451b2d5fff5923b5f44e8abc588`. Images remain unaltered JPEGs; source polygons and source split names are preserved in separately hashed JSON.

Actual acquisition is **95 real photos**:

| Partition | Count | Purpose and limitation |
|---|---:|---|
| Negative training | 20 | From official training images; no published Crack, ACrack or Spalling polygon. Other real damage can be present. |
| Negative calibration | 10 | Different official-training images, reserved from fitting the new ev-CIVIL crack/spall detector. |
| Negative held-image diagnostic | 10 | Official-validation images reserved from that fitting. Physical bridge identities are unavailable, so this is image separation, not proven bridge separation. |
| Broad damage diagnostic | 55 | Union of the first eight eligible official-validation images per 14 relevant damage/condition labels; metadata-hash ordering fixed before pixels/inference. Images can have multiple classes. |

`datasets/construction/dacl10k/round3-selection-before-inference.json` records immutable selection. `round3-asset-manifest.json` records every original image path, source URL, class inventory and image/annotation hashes; SHA-256 `7f192af6e6df8d2cb92e52ca70b4e7d657d2095fae7c2923aa1885cf6fcae882`. The 40-image negative manifest SHA-256 is `0413a8e3b596af25b72469a93e914956824d0be39bd4da5b78c7ca2d64cf402d`. Identical selected image hashes across partitions abort acquisition. Near-duplicate scenes or bridges cannot be ruled out without original bridge/session identifiers.

These are **still photographs**. They are not encoded into a fake inspection video. Verified additional ev-CIVIL moving camera sequences are handled separately by the construction training agent, with exposed capture groups excluded from fitting. That dataset supplies crack/spalling classes; it does not validate rust, efflorescence or exposed-rebar video detection.

## Actual reusable trained model

The dataset-author [DACL challenge Space](https://huggingface.co/spaces/phiyodr/dacl-challenge) is linked by the official toolkit and includes an actual **SegFormer MiT-B1, 19-channel model trained for ten epochs at 512×512**, according to the author's serving code. This is a trained damage model, rather than an ImageNet-only backbone. Pinned Space commit: `f5bee34782e7e02393ba4f206a95ba5ed67eb8a5`. Original checkpoint: **54,833,953 bytes**, SHA-256 `166b778bdbc57ef2bfd5f71bc6ae724bf598d1a4c6a1b69afa4fbd936f2d965b`.

The author file is a full Torch model pickle. Its global references were inspected without executing it, then restored using **`weights_only=True` and an explicit reviewed Torch/SegFormer type allowlist**. No unrestricted pickle load was used. A plain 208-tensor state dictionary was extracted, with 13,682,644 stored elements and SHA-256 `457777454570cf8f2c7c16d59869a7a9a5d0b4ac6acc87fee5484a3beaa270db`. Extraction performed **no model forward pass**. Original source files, safe extraction report and manifests remain under `models/construction/dacl19/`.

The reviewed Space README does **not** separately declare a checkpoint/code license. Dataset licensing does not establish proprietary redistribution permission for its trained demo weights. Commercial clearance remains unresolved; this is an experimental research candidate.

The [published DACL baseline study](https://openaccess.thecvf.com/content/WACV2024/papers/Flotzinger_dacl10k_Benchmark_for_Semantic_Bridge_Damage_Segmentation_WACV_2024_paper.pdf) compares CNN baselines, including an FPN/EfficientNet-B4 baseline. Its published IoU is not attributed to the Space's MiT-B1 checkpoint or this laptop. No official ready-to-use lightweight MobileNet damage checkpoint was verified during this search. Large DINO/ViT ensembles were not selected as a credible laptop starting point.

[CODEBRIM's official repository](https://github.com/MrtnMndt/meta-learning-CODEBRIM) supplies an additional concrete-defect research dataset and author code under research/educational restrictions. It is a potential source for crack, spallation, exposed bars, efflorescence and corrosion. Its patch classification setup does not establish full-frame semantic masks or real video. It was not downloaded because the selected DACL source already supplies the broader labeled masks and an author-trained checkpoint needed for the immediate experiment.

## Local runtime and measurement protocol

`scripts/export_dacl19.py` prepares fixed 320/512 ONNX candidates. `edge_runtime/profiles/construction_dacl.py` implements a separate adapter so existing crack/PPE decoding stays independent. The model takes uint8 BGR NCHW input, with RGB/ImageNet normalization inside the graph, returning 19 native stride-four logits. Per-channel sigmoid preserves overlapping labels. Native masks use strict probability `>0.5`, a predeclared four-grid-pixel component minimum, largest 20 components per class and largest 100 observations overall. Original-image projection uses nearest-neighbor geometry. Objects such as bearings are not automatically reported as damage. No trained model is downloaded at runtime.

Preprocessing is **an approximation of the author serving pipeline**, not bit-identical reproduction: the author first converts RGB to floating point `[0,1]`, then applies torchvision Resize and normalization; this local adapter first resizes uint8 BGR with OpenCV, then performs RGB/255 and normalization inside ONNX. Integer rounding, resize kernels and version-dependent antialiasing can differ. The already measured 512/320 reports describe this actual local pipeline; no silent preprocessing correction or transfer of those scores to a different pipeline was made.

The default enables 14 condition names, including concrete washout appearance. These are model labels, not proof of mechanisms such as hidden delamination, active leaking, chemical attack or loss of reinforcement section. The capability ledger in `config/construction-capabilities.json` distinguishes untested research candidates, locally **measured_weak** labels and experimental support. Availability of all channels does not mean 14 conditions were solved. Existing crack support retains its known thin-crack failure; spalling is not silently credited to a crack-only detector.

`scripts/evaluate_dacl_round3.py` freezes the exact image inventory and operating point before inference. It reports per-class foreground pixel TP/FP/FN, precision, recall, IoU, positive-image coverage, false flags on class-absent images and derived connected-region TP/FP/FN at IoU 0.50. Connected semantic regions are a diagnostic derived from masks, not native object-instance labels. Original publisher-style Pillow polygon filling is restored from normalized mirror polygons; individual boundary pixels may differ from the original JSON representation.

The visual export chooses the first actual annotated example of each class in the frozen subset, keeping **RAW CAMERA PHOTO / PUBLISHER ANNOTATION / ACTUAL LOCAL MODEL MASK** in separate panels. Failed detections remain visible. The 55-image subset comes from the authors' public validation partition; checkpoint-training/selection overlap is not independently established. Its scores are a small diagnostic, not an untouched test, an all-stage guarantee or Saudi-site validation.

Four decoder fixtures passed in **0.38 seconds**: overlapping class masks, boxes/masks decision parity, nearest-neighbor original coordinates and model-grid area filtering. These generated arrays are unit tests, not damage data or model evidence. Actual model results and dedicated capacity measurements are recorded below; accuracy-audit incidental timings are excluded from the speed headline.

## Reproduction

```powershell
.venv\Scripts\python.exe -X utf8 scripts/fetch_dacl19_model.py
.venv\Scripts\python.exe -X utf8 scripts/fetch_dacl_round3.py
.venv-roads-export\Scripts\python.exe -X utf8 scripts/export_dacl19.py --extract-only
# The commands below run model forward passes: reserve the shared compute window.
.venv-roads-export\Scripts\python.exe -X utf8 scripts/export_dacl19.py --size 512
.venv\Scripts\python.exe -X utf8 scripts/evaluate_dacl_round3.py --size 512 --output output/evaluation/construction-dacl19-round3/512
.venv-roads-export\Scripts\python.exe -X utf8 scripts/export_dacl19.py --size 320
.venv\Scripts\python.exe -X utf8 scripts/evaluate_dacl_round3.py --size 320 --output output/evaluation/construction-dacl19-round3/320
.venv\Scripts\python.exe -X utf8 scripts/benchmark_dacl19.py --size 512 --output output/evaluation/construction-dacl19-round3/512-speed.json
.venv\Scripts\python.exe -X utf8 scripts/benchmark_dacl19.py --size 320 --output output/evaluation/construction-dacl19-round3/320-speed.json
```

The export environment adds transformers 4.57.6 and associated tokenizer/safetensors libraries; the app has no Torch/transformers runtime dependency. All inference remains local. The model's input size, actual local execution device, preprocessing, thresholds and model SHA are exposed in its description.

## Measured local results: fixed 512 versus 320

Both ONNX exports and all 110 photo evaluations completed locally at the predeclared strict probability `>0.5`, four-native-grid-pixel component minimum, 20 components per class and 100 observations total. No detector threshold, weight, class or example selection was changed after seeing these results. The 320 graph uses the same learned state at a smaller geometry; it was not retrained for that geometry. The four-grid-pixel minimum covers a larger source area at 320, so this is the complete configured-pipeline tradeoff, not a resolution-only ablation.

**The broad model remains weak on most categories.** Some real masks overlap meaningful defects, but a tiny overlap is not a successful localization. Results below include pixel precision/recall/IoU, one-to-one connected-region matches at IoU0.50 and class-absent-photo false flags. Chalk-boundary segmentation is the strongest measured label; that is an inspector marking rather than a diagnosis of internal damage. The weaker 320 model misses every derived crack region at IoU0.50 despite touching crack pixels in some images.

### 512×512 actual masks on the same 55 photos

| Label | Pixel precision | Pixel recall | Pixel IoU | Region TP / FP / FN at IoU0.50 | Positive photos ≥50% GT coverage | Class-absent photos flagged |
|---|---:|---:|---:|---:|---:|---:|
| crack | 25.2% | 51.1% | 20.3% | 2 / 107 / 25 | 6/15 | 15/40 |
| alligator crack | 37.3% | 45.5% | 25.8% | 2 / 55 / 7 | 3/8 | 7/47 |
| wetspot | 18.4% | 57.8% | 16.2% | 4 / 45 / 6 | 4/9 | 17/46 |
| efflorescence | 67.6% | 43.3% | 35.9% | 6 / 40 / 43 | 5/12 | 8/43 |
| rust | 69.2% | 21.0% | 19.2% | 26 / 66 / 93 | 17/30 | 5/25 |
| rockpocket | 33.4% | 73.2% | 29.8% | 4 / 32 / 6 | 4/8 | 12/47 |
| Inspector chalk boundaries | 77.1% | 77.2% | 62.8% | 12 / 20 / 10 | 13/17 | 5/38 |
| cavity | 14.7% | 70.6% | 13.8% | 3 / 130 / 60 | 3/12 | 22/43 |
| spalling | 69.9% | 34.5% | 30.0% | 12 / 71 / 89 | 12/36 | 7/19 |
| graffiti | 71.2% | 44.9% | 38.0% | 4 / 16 / 23 | 5/11 | 4/44 |
| weathering | 27.4% | 32.1% | 17.3% | 7 / 101 / 40 | 15/28 | 10/27 |
| residual formwork | 31.2% | 62.3% | 26.3% | 3 / 37 / 8 | 5/8 | 11/47 |
| exposed rebar | 60.9% | 15.9% | 14.4% | 4 / 19 / 13 | 6/9 | 4/46 |
| Surface loss / erosion appearance | 12.8% | 7.4% | 4.9% | 2 / 44 / 10 | 1/8 | 19/47 |

### 320×320 actual masks on the same 55 photos

| Label | Pixel precision | Pixel recall | Pixel IoU | Region TP / FP / FN at IoU0.50 | Positive photos ≥50% GT coverage | Class-absent photos flagged |
|---|---:|---:|---:|---:|---:|---:|
| crack | 15.4% | 20.5% | 9.6% | 0 / 44 / 27 | 1/15 | 9/40 |
| alligator crack | 25.1% | 33.0% | 16.6% | 1 / 27 / 8 | 2/8 | 9/47 |
| wetspot | 11.5% | 51.7% | 10.4% | 3 / 20 / 7 | 4/9 | 10/46 |
| efflorescence | 43.1% | 31.1% | 22.0% | 4 / 19 / 45 | 3/12 | 6/43 |
| rust | 39.6% | 19.4% | 15.0% | 19 / 37 / 100 | 14/30 | 3/25 |
| rockpocket | 42.0% | 46.1% | 28.2% | 3 / 20 / 7 | 3/8 | 7/47 |
| Inspector chalk boundaries | 68.6% | 69.7% | 52.9% | 9 / 14 / 13 | 9/17 | 3/38 |
| cavity | 8.5% | 39.4% | 7.5% | 2 / 55 / 61 | 2/12 | 12/43 |
| spalling | 59.7% | 28.1% | 23.6% | 9 / 39 / 92 | 7/36 | 5/19 |
| graffiti | 61.7% | 36.7% | 29.9% | 4 / 8 / 23 | 4/11 | 1/44 |
| weathering | 28.5% | 23.3% | 14.7% | 8 / 67 / 39 | 12/28 | 13/27 |
| residual formwork | 28.3% | 35.5% | 18.7% | 1 / 16 / 10 | 2/8 | 7/47 |
| exposed rebar | 57.6% | 15.5% | 13.9% | 3 / 9 / 14 | 5/9 | 2/46 |
| Surface loss / erosion appearance | 11.7% | 3.5% | 2.8% | 0 / 31 / 12 | 1/8 | 13/47 |

A class-absent image can contain other genuine damage; these false flags are therefore class-confusion/false-localization diagnostics, not claims that all flagged photos are healthy. Connected semantic regions can fragment or merge, so their TP/FP/FN are explicitly derived diagnostics; pixel counts in the JSON remain the primary spatial measure. Public validation overlap with the checkpoint is unknown, and correlated bridge views cannot be ruled out. No result is an independent all-stage or Saudi construction validation.

### Dedicated local warm adapter capacity

Actual Intel UHD620 OpenVINO FP32, one stream, six fixed varied real photos, two passes per rendering mode after warmup; source longest edge at most 1920 pixels. Twelve observations per mode. Original JPEG reading, model compilation, camera/video decode, dashboard transport and shared-runtime scheduling are excluded. Full masks include nearest-neighbor per-component output masks at that analysis geometry. This is bounded serial adapter capacity, **not moving-video or end-to-end FPS**. Accuracy-audit timings are not used for this table.

| Input | Output mode | Mean adapter ms | p50 ms | p95 ms | Serial adapter FPS |
|---|---|---:|---:|---:|---:|
| 512 | boxes | 336.4 | 331.5 | 379.8 | 2.97 |
| 512 | masks | 411.6 | 357.1 | 713.8 | 2.43 |
| 320 | boxes | 108.9 | 108.8 | 116.4 | 9.18 |
| 320 | masks | 158.8 | 146.3 | 250.2 | 6.30 |

**Neither broad-model geometry meets 15 FPS on this laptop.** The 512 model is the quality/reference photo-review variant; 320 is an explicitly weaker, faster diagnostic option. It is not substituted into the 512 accuracy claim. Factory15-FPS qualification is measured separately with the factory model. Applying this broad classifier to a video is executable, but these photos do not validate temporal stability or real-video localization for rust, moisture or other added conditions.

Ledger statuses describe the measured evidence: newly measured labels below 0.50 pixel IoU are marked `measured_weak`; the chalk-boundary label is `supported_experimental` with its restricted photo scope. That 0.50 status boundary was chosen after measurement for descriptive review, not tuned into the model or represented as an industry acceptance rule. Existing crack and new crack/spall video-model identities retain their own evidence and metrics; supplemental DACL results are recorded separately.

### Files and integrity

- Primary `models/construction/dacl19/dacl-segformer-b1-512.onnx`: SHA-256 `080fb07297bbb2841fbd5f49b9d8888da53ee1c2adc1ea33bdd2addb8cc9ea7f`.
- Alternate `models/construction/dacl19/dacl-segformer-b1-320.onnx`: SHA-256 `c89aa3002de737cf19ea3043704af36837ad92267466f1234ac2bcecee60ea56`.
- `output/evaluation/construction-dacl19-round3/protocol-before-any-export.json` freezes method before any model forward.
- `output/evaluation/construction-dacl19-round3/{512,320}/report.json` contains full pixel TP/FP/FN, image outcomes and derived matches. Each folder contains 14 first-inventory raw/annotation/model triptychs; these are photographs, not encoded video.
- `output/evaluation/construction-dacl19-round3/{512,320}-speed.json` contains independent dedicated capacity observations.
- `output/evaluation/construction-dacl19-round3/gallery-manifest.json` maps each class to the first annotated original photo and raw/annotation/model comparison, including that case's actual spatial metrics.
- `scripts/summarize_dacl_round3.py` refreshes this measured section and capability evidence from completed reports without running inference.
