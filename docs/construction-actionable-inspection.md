# Construction video: damage evidence instead of object boxes

Research and implementation date: 20 October 2026.

The earlier videos did not demonstrate dependable damage detection. Person, helmet, vehicle and machinery boxes show object recognition, not a construction mistake. The bareheaded-worker clip also contains known helmet misses. The interview is context footage. The bridge clip produces false crack candidates around steel joints. These are retained as failure diagnostics; none establishes successful construction inspection.

## Changes to the live output

Context objects are now hidden by default. `show_context_objects: true` is an explicit debug option. Only the detector's explicit `NO-Hardhat`, `NO-Safety Vest` and `crack` channels can produce an issue candidate. A person without a separate helmet detection never automatically means a missing helmet.

Each visible candidate has a number and a plain-language explanation: what the model predicted, where it is in analysis pixels, why it could matter and the actual next check. For a possible crack, the next check is to distinguish a real crack from a joint, stain or shadow, then measure its width with a calibrated tool. Confidence is a model score, not certainty or a code verdict. The full raw object observations remain available for debugging, while `domain_state.review_findings` carries actionable explanations.

## New genuine damaged-surface recordings

The [ev-CIVIL dataset](https://doi.org/10.11583/DTU.25924228), published by Kashita Niranjan Udayanga Gangoda Withana Gamage, Silvia Tolu, Matteo Fumagalli, Tobi Delbruck and Cesar Cadena, with defect annotation contribution by Xuanni Huo, supplies actual recorded grayscale exposures and timestamped crack/spalling bounding boxes. Its [author repository](https://github.com/gwgknudayanga/evCIVIL_Event_based_defect_detection) and [archived paper](https://arxiv.org/abs/2504.05679) explain the DAVIS346 acquisition. The dataset license is **CC BY 4.0**; the repository's code license is separate. Field recordings were collected in Denmark, Switzerland, Portugal and Spain. They are not Saudi project or building-code acceptance records.

Three sequences were selected by their publisher category and bounded download size **before any local model inference**. Original pixels were visually checked before inference. No model-generated label was used as ground truth.

| Sequence | Genuine visible condition | Original exposures | Publisher-labelled frames used | Intended evidence |
|---|---|---:|---:|---|
| `crack_10` | Prominent branching surface crack; camera moves across the damaged surface | 162 | 152 | Whether a local crack detector actually localizes known damage |
| `crack_89` | Thin dark crack with difficult exposure | 24 | 8 | A harder case, with misses retained |
| `spalling_35` | Chipped/missing concrete at an exposed edge | 72 | 72 | An explicit unsupported damage class; the installed crack model has no spalling output |

These counts describe a small selected diagnostic subset, not population accuracy. Some original frames have no nearby publisher annotation and are excluded from accuracy rather than treated as sound negatives. `crack_89` is sparsely annotated. Publisher boxes do not provide pixel masks, physical width, severity, cause or acceptance labels. The `spalling_35` edge can be overexposed; original motion remains visible, and unsupported status is retained regardless of crack detections.

The original field ZIP is approximately 10.87 GB. Only selected raw `frames.h5` and annotation `npy` members were obtained using HTTP byte ranges. Every selected member was read to EOF with ZIP CRC verification, then given a local SHA-256/provenance file. No event-stream volumes were needed. Untrusted NPY files are loaded with `allow_pickle=False`.

## Time, labels and videos

These are existing-infrastructure surface-condition recordings, not evidence of a specific pouring, reinforcement or waterproofing installation mistake. No defect cause, exact structure type or Saudi construction-stage compliance is inferred from the frame alone.

The videos contain **one encoded frame per original camera exposure** in original order. PyAV/libx264 uses the original microsecond exposure timestamps as VFR presentation timestamps shifted to zero. Reopening each output verified its frame count and timestamp sequence: maximum timestamp discrepancy **0.0 seconds** for all three original-frame videos. There are no repeated stills, generated scenes, interpolated frames or fabricated events. Display scaling and explanatory model overlays are derived output; untouched camera arrays remain local.

Publisher annotation association follows the author's first-timestamp-at-or-after convention, with a bounded half-frame-period plus 1 ms maximum gap. A distant annotation is never carried into another frame. All original timestamps and actual assigned annotation timestamps are retained in `annotations.json`. HTML video playback uses the encoded presentation times. The edge runtime's current nominal-FPS file scheduler is not a full VFR scheduler, so a live replay of these files is not presented as exact source-time validation.

The explanatory output now separates model predictions from external labels with a visible legend: **solid green** is a model box that matches the publisher box at IoU .50; **solid amber** is an unmatched model candidate. **Dashed red/cyan** boxes are external publisher missed/matched crack annotations; **dashed gray** is unsupported spalling truth. The application also shows the separate expected/predicted/missed counts and source coordinates. External labels never enter the model input and are never presented as model detections.

## Localization protocol

The final viewer review also corrected diagnostic IoU display for unmatched boxes: geometric overlap is measured against all same-class annotations, including an annotation already claimed by a higher-confidence prediction. One-to-one assignment remains unchanged, so an extra overlapping box can still be unmatched. Thirty-nine stored diagnostic values were corrected across the six case JSONs; no prediction coordinates, confidence, matched IDs, TP/FN/FP counts or reported accuracy changed. Two geometry tests cover duplicate-box overlap and crack-versus-spalling class separation. Unmatched IoU is not silently zeroed simply because an annotation was already used.

The operating point was frozen before inference: existing pinned YOLOv8n crack model, confidence **0.25**, NMS **0.45**, explicit OpenVINO GPU FP32, box rendering. A prediction matches one publisher defect only when its class agrees and **bounding-box IoU is at least 0.50**. Predictions are greedily matched by confidence, each expected defect once. A crack prediction over a spall is not a correct spalling detection. Unlabelled frames do not enter precision/recall. Adjacent frames are correlated and cannot be presented as independent buildings or defects.

The model has not been retrained on these sequences. Training overlap with its original public crack corpus is unknown. This evaluation is diagnostic; it is not independent Saudi-site validation. Source sequence choice and thresholds were not optimized against these video results. The benchmark script stores every frame's prediction and label association so that apparent success can be inspected rather than inferred from video-level status.

Model timings include preprocessing, local inference and candidate decode; they exclude source decode, explanatory video rendering and encoding. They are not a claim that every high-resolution construction stream can run at that rate. Factory throughput is benchmarked separately.

## Measured existing-model baseline

| Real sequence | Box TP / FN / unmatched model boxes | Box recall | Box precision | Warm model FPS | Model p95 |
|---|---|---:|---:|---:|---:|
| `crack_10` | 14 / 148 / 157 | 8.64% | 8.19% | 15.32 | 70.53 ms |
| `crack_89` | 0 / 8 / 8 | 0% | 0% | 17.53 | 60.11 ms |
| `spalling_35` | 72 unsupported annotations; no spalling class | Not scored | Not scored | 14.91 | 73.77 ms |

This is a genuine failure under the preregistered **whole-instance box IoU .50** protocol. It must not be described as good inspection recall. Importantly, “unmatched model box” is a metric classification, not necessarily a prediction of nonexistent damage: the prominent branch in `crack_10` is visibly damaged and the model often marks its lower branch, but its box does not cover the publisher's entire branching-crack instance. At source frame 80, the predicted box is `[190,131,156,87]` and the publisher's whole-instance box is approximately `[144.59,42.62,201.41,186.36]`, IoU **0.362**. It is still a miss plus an unmatched prediction at the frozen threshold. Pixel-mask accuracy cannot be recovered from box labels.

There is usually only one model candidate in these frames. Merging fragments cannot recover the missing upper branch, and enlarging boxes to fit the exposed labels would fabricate an apparent improvement. No such box expansion or IoU relaxation was adopted. Fourteen frames meet the full box criterion; all other predictions and misses remain visible and counted. The actual clip now shows known damage and exactly what the detector covers and misses, rather than unrelated object boxes.

## Reproduce

Use the isolated export environment for `h5py==3.16.0`, `av==19.0.1` and video preparation. They are not added to the edge application's runtime dependencies.

```powershell
.venv\Scripts\python.exe -X utf8 scripts\fetch_evcivil.py
.venv-roads-export\Scripts\python.exe -X utf8 scripts\prepare_evcivil.py
.venv-roads-export\Scripts\python.exe -X utf8 scripts\evaluate_construction_defect_video.py --engine openvino --device GPU --output-tag baseline
```

Files are under `datasets/construction/evcivil/` and `output/construction-{sequence}-baseline.{mp4,json}`. The JSON contains model/source provenance, the precise evaluation scope and all per-frame matches, misses and false alerts. `output/construction-evcivil-baseline-summary.json` is the compact benchmark table.

## Remaining construction scope

The current model detects only crack candidates and the earlier PPE channels. It does not detect spalling, honeycombing, exposed reinforcement, incorrect reinforcement spacing, missing waterproofing, inadequate compaction or hidden work. The Saudi lifecycle checklist remains an evidence plan with unassessed checks. A real chipped-concrete video makes an unsupported condition visible; it does not create a spalling detector.

Further model leads were investigated without treating their names or reported image scores as validation. [EfficientCrackNet](https://openaccess.thecvf.com/content/WACV2025/papers/Zim_EfficientCrackNet_A_Lightweight_Model_for_Crack_Segmentation_WACV_2025_paper.pdf) is a relevant lightweight segmentation architecture, but usable official pretrained weights were not established in this work. Public spalling checkpoints without a clear model architecture, documented training domain and transferable license were not silently installed as working detection. Extending the product requires actual class-specific models and independently labelled source-separated videos, including sound surfaces, joints, stains, shadows and exposure failures.

## Joint-aware candidate comparison protocol (frozen before inference)

The experimental candidate is [levanell's YOLOv8n cracks/joints checkpoint](https://huggingface.co/levanell/yolov8n-seg-cracks-joints/tree/44e73b6b015247bde629e9e7f12efef2ba4c7f3b), declaring AGPL-3.0. The author describes approximately 6,187 merged wall-crack/drywall-joint training images; those source licenses and independent training-domain coverage are not established. The checkpoint SHA-256 is `4aee1057161d3ce0130f438d118a3920997462f1ce187f24767c7ea5e1905e00`. Restricted `weights_only=True` loading permits only inspected standard Torch/Ultralytics globals, with no author code execution.

The frozen settings match the baseline: confidence .25, NMS .45, class-specific box IoU .50, 640-pixel letterbox RGB input and explicit OpenVINO GPU FP32. Joint predictions remain non-issue context, never crack positives. The **default switch rule** requires improved aggregate localization recall across **both** crack sequences and precision no worse than the baseline. Baseline aggregate is TP 14, FN 156, FP 165: recall **8.24%**, precision **7.82%**. No per-video model cherry-picking is permitted. Spalling stays unsupported. The machine-readable rule is `datasets/construction/evcivil/joint-candidate-protocol.json`.

These clips are already exposed by the baseline diagnostic. A candidate result is a reused development comparison, not new independent validation. No video-specific box enlargement, score threshold relaxation or annotation change is part of the comparison.

## Joint-aware result and application selection

The candidate passed the frozen aggregate rule. This is an actual model-weight change with the same geometry and operating thresholds, not box enlargement or a relaxed localization rule. Both crack cases now select the **same explicit candidate**; baseline outputs remain in the diagnostic archive. The general crack model default is kept compatible; the real-video presets explicitly set `crack_variant: joint_aware_candidate`, `engine: openvino`, `openvino_device: GPU`, `confidence: 0.25` and `mask_rendering: boxes`. Joint detections are non-issue context and hidden by default.

| Exposed public sequence | Baseline TP / FN / FP | Joint-aware TP / FN / FP | Baseline → joint-aware box recall | Baseline → joint-aware box precision | Joint-aware warm model FPS / p95 |
|---|---|---|---|---|---|
| `crack_10` | 14 / 148 / 157 | **136 / 26 / 87** | 8.64% → **83.95%** | 8.19% → **60.99%** | 15.19 / 78.24 ms |
| `crack_89` | 0 / 8 / 8 | **0 / 8 / 9** | 0% → **0%** | 0% → **0%** | 14.65 / 94.29 ms |
| **Aggregate across both** | **14 / 156 / 165** | **136 / 34 / 96** | **8.24% → 80.00%** | **7.82% → 58.62%** | Separate per-sequence model capacity above |
| `spalling_35` | 72 unsupported annotations | **72 unsupported annotations** | Not scored | Not scored | 15.32 / 71.15 ms |

The actual branching-crack footage now contains many legitimate whole-instance localized detections. The unchanged challenge failure and 96 unmatched boxes across scored crack frames remain material limitations. **80% box recall and 58.62% precision on two reused clips are not dependable structural inspection.** The difficult sequence still misses every labelled crack at the .50 criterion. The spalling clip demonstrates a real damaged edge that this product cannot yet detect; it never becomes a “clear” inspection.

The FPS figures measure preprocessing, local model execution and candidate decoding only. They exclude video decode, explanatory overlays, encoding, capture timing and UI work. The challenge's 14.65 FPS is below 15; none of these figures is an end-to-end 15 FPS construction guarantee. This does not change the separately benchmarked factory requirement. Pixel-mask IoU is unavailable because these videos have publisher boxes rather than masks. Adjacent frame labels are correlated and are not independent physical defects.

The pinned ONNX candidate is `models/construction/crack-joints-yolov8n-seg.onnx`, SHA-256 `4d48232d7b1427ac468022d2cc11a344885629cac71ad48bd2ff4c6659302246`. Source checkpoint SHA and restricted loading provenance are stored beside it. Both live and offline decoding use the explicit class order `[joint, crack]`; no joint output is relabelled as a defect. Ten focused tests cover preprocessing, baseline/candidate class separation, issue-only rendering, actionable guidance, temporal events and ROI coordinates with mocked engines. Real execution evidence is the actual video evaluation, not these fixtures.

Reproduce the candidate export and identical evaluation in the isolated export environment:

```powershell
.venv-roads-export\Scripts\python.exe -X utf8 scripts\export_construction_joint_candidate.py --download
.venv-roads-export\Scripts\python.exe -X utf8 scripts\evaluate_construction_defect_video.py --engine openvino --device GPU --candidate-joints --output-tag joint-candidate
```

The explicit setup flag downloads only the pinned checkpoint and verifies its SHA-256. Export uses `torch.load(weights_only=True)` with an inspected standard Torch/Ultralytics allowlist. It does not execute downloaded author code. Inference never downloads models. The local ONNX graph is hash-checked by the application. The weights are AGPL-3.0; underlying merged Roboflow training-source permissions are not independently cleared for commercial redistribution.

`output/evaluation/construction-video-model-comparison.json` records the paired aggregate and frozen-rule decision. Per-frame evidence and timestamp-verified H264 videos are `output/construction-{sequence}-joint-candidate.json` and `.mp4`. Original camera arrays, timestamps, publisher labels, source/member CRCs and SHA hashes remain under `datasets/construction/evcivil/`.

## Lossless live-replay input fidelity

The localization measurements above use the **original camera arrays**, not frames decoded from the lossy H264 browser recording. A lossy source replay can change pixels and therefore model outputs; its offline accuracy must not be assumed identical. The H264 annotated player remains a readable result recording.

The separate source-only preparation script `scripts/prepare_evcivil_lossless.py` uses **lossless FFV1 grayscale in a NUT container**, with the original microsecond presentation timestamps. NUT is intentional: the standard FFmpeg [Matroska muxer](https://github.com/FFmpeg/FFmpeg/blob/master/libavformat/matroskaenc.c) writes a millisecond `TimestampScale`, which would round these camera exposures. A source replay container does not need browser-native playback; OpenCV/FFmpeg reads it while the UI continues using annotated H264.

The script requires one encoded frame per actual exposure in original order. It rejects any PyAV timestamp/count mismatch, any PyAV grayscale pixel difference, or any OpenCV BGR difference from `cv2.cvtColor(raw_gray, COLOR_GRAY2BGR)`, checking **every original frame**. The per-file provenance and combined report retain the source license, author attribution, SHA-256, decoder backend, frame counts and maximum timestamp error. This verifies input identity rather than assuming a codec name guarantees it. No model is loaded during preparation.

```powershell
.venv-roads-export\Scripts\python.exe -X utf8 scripts\prepare_evcivil_lossless.py
```

The original-video presets should use `datasets/construction/evcivil/{sequence}/{sequence}-lossless-frames.nut`. The current runtime's nominal-FPS scheduling still does not become a fully VFR-aware scheduler merely because the container stores exact camera timestamps. A real runtime test must report which original frames were processed, skipped or dropped; the offline whole-recording metrics are not automatically live-session accuracy.

The lossless preparation and verification **completed successfully in 4.3 seconds**, without loading or running a model:

| Genuine sequence | Original / decoded frames | OpenCV BGR versus raw gray → BGR | PyAV gray versus raw gray | Maximum PTS discrepancy |
|---|---:|---|---|---:|
| `crack_10` | 162 / 162 | Every pixel identical | Every pixel identical | **0.0 microseconds** |
| `crack_89` | 24 / 24 | Every pixel identical | Every pixel identical | **0.0 microseconds** |
| `spalling_35` | 72 / 72 | Every pixel identical | Every pixel identical | **0.0 microseconds** |

All three sources open through OpenCV's FFMPEG backend. Their SHA-256 hashes and exact checks are in [the lossless source verification report](../output/evaluation/construction-lossless-source-verification.json), the `.provenance.json` beside each `.nut`, and `source.lossless_replay_input` in each case JSON. This establishes pixel identity with the offline benchmark input. It does not establish identical end-to-end scheduling, runtime coverage or new independent accuracy; those require the separate actual source/runtime test. The annotated H264 browser recordings remain lossy visual outputs and are not used as model inputs for this identity claim.
