# Construction video observations (local, experimental)

This module runs two real, downloaded neural models on original camera frames or local video. Select **PPE** or **surface crack** mode; only one model runs for each frame. It does not certify Saudi Building Code compliance. No box, a green box, or a clear view does not pass a building inspection. All concealed work, material tests, dimensions, strength, engineering severity and unsupported stages remain **unassessed**.

The separate [Saudi inspection study](saudi-construction-inspection-study.md) and [inspection checklist](saudi-construction-checklist.md) record the much larger inspection scope. A visible PPE observation is a site observation, not evidence that a structural, electrical, fire or plumbing check passed.

## Real models and exact contracts

| Mode | Pinned source | Local ONNX SHA256 | Input and output |
|---|---|---|---|
| PPE | [mancalazure/construction-ppe-safety, revision 70cc801](https://huggingface.co/mancalazure/construction-ppe-safety/tree/70cc801710f246765f70ada7248d9560ec749bb2) | `1d5f3cf69883e89ba6e2b3836a0b21623044e9a682f1f00dfbacac97b99baf94` | Author serving example specifies RGB **stretch**; application now defaults to aspect-preserving **letterbox** after the documented geometry study. Fixed 640×640 FP32 [0,1]; 1×14×8400 output, 10 classes, class-aware NMS |
| Surface crack | [beaunix/smart-crack-vision, revision 9442007](https://huggingface.co/beaunix/smart-crack-vision/tree/9442007ef9c1003880baa5796d253608691b37c4) | `4b789658646d790079d3e78d9d3ed475030d740a4b8412aff52248189a114cbd` | RGB letterbox 640×640, FP32 [0,1]; 1×37×8400 boxes/score/mask coefficients + 1×32×160×160 mask prototypes |

PPE classes in verified ONNX metadata order: Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone, Safety Vest, machinery, vehicle. Review events use only explicitly predicted NO-Hardhat and NO-Safety Vest channels; no missing-equipment inference is made from a person with no detected hat/vest. Mask outputs are observations only. Model class scores are not calibrated probabilities or regulatory conclusions.

Crack mode returns a visible surface-crack candidate and mask, with no calibrated millimetres, crack cause, construction material classification, acceptability, structural strength or severity. This model was trained on mixed road/wall crack images, not a qualified Saudi concrete inspection set.

Both are original ONNX downloads; no downloaded Python code or pickle checkpoint executes. ONNX input contracts and file hashes are checked. Inference never downloads or uploads. ONNX Runtime CPU (4 threads, busy spinning disabled) is the portable default; OpenVINO CPU or the laptop Intel GPU is an explicit option. OpenVINO FP32 is requested, with no quantization or resized model export.

## Genuine video and annotated data

- [Men at work in building site](https://commons.wikimedia.org/wiki/File:Men_at_work_in_building_site.webm): Akum20, own work, 12 March 2024, **CC0 1.0**, original 1080×1920, 25 FPS, 131 frames / 5.24 seconds. Local `datasets/construction/men-at-work-cc0.webm`, SHA256 `f7cbf03f56e0a864ddba057c47b8a67c57f176e10e04995c59e14d8ab00024d7`. Genuine on-site mortar mixing and moving people; no generated scene or still-image loop.
- [A construction framer talks about protecting his crew from falls](https://commons.wikimedia.org/wiki/File:A_construction_framer_talks_about_protecting_his_crew_from_falls.webm): National Institute for Occupational Safety and Health, 2014, US federal **public domain**. Original 1920×1080, 23.976 FPS, 5863 frames / 244.55 seconds. Local `datasets/construction/framer-fall-protection-public-domain.webm`, SHA256 `017422936ccbab19830136730acec895fb788a06b6e1b195a3cb300530d2728a`. Visually inspected samples at 10,35,65,100,150 seconds show live-action interview in a timber frame building; 200 seconds is a dark transition. It is an interview, not representative site CCTV or an inspection-ground-truth dataset. Extracted original source frames and contact sheet are in `output/niosh-source-*`.
- [Atharion PPE public v1.0](https://github.com/atharion-team/ppe-dataset-v1): all **169 test images**, original boxes remapped only to Person/Hardhat/Safety Vest. Mixed **AGPL-3.0** and **CC BY 4.0** sources, NOTICE and per-image attribution retained. Archive SHA256 `f5f6f1966c4bdfff4151ee68fa0da21f446e1b788d15280e5f70028c69ec88c5`. Training overlap with this model is unknown; this is a compatibility diagnostic, not independent accuracy. Negative PPE channels have no ground truth in this set and are not scored as false positives or verified violations.
- [Ultralytics Crack-Seg](https://docs.ultralytics.com/datasets/segment/crack-seg/): all **112 supplied test images**, real road/wall surfaces and polygon labels, Public Domain as declared by distributor, original University/Roboflow attribution retained. Archive SHA256 `814b566d61e890024d15f22a3e6b14e252f781ab83f60a8d59e6684817efecdc`. Training overlap unknown. Report both bounding-box detection and foreground pixel IoU/Dice; never inflate segmentation accuracy with background true negatives.
- [Raw bridge inspection video, Zenodo record 19268973](https://zenodo.org/records/19268973): N. T. Tsegaye, 2026, **CC BY 4.0**. Original `bd_vid_0031.mp4`, 3840×2160, 29.946 FPS, 364 frames / 12.155 seconds, 151,145,752 bytes, SHA256 `03dd30b8dacf531990b7867273cb4041209b48acf0f2bf064ceddd9517a050a9`. Selected via HTTP byte ranges from the publisher's raw-video ZIP; original MP4 bytes preserved and entry CRC verified, avoiding an 18.46 GB full archive download. Provenance and licence are beside `datasets/construction/bridge_raw/bd_vid_0031.mp4`. Original sampled frames at 0,3,6,9 seconds show actual bridge concrete underside and steel girders, without model annotations or animation. Fine hairline indications can disappear during 4K-to-640 downscaling; source dimensions do not mean the model inspects every original pixel. Separate published image labels do not supply verified video-frame ground truth.
- Five frames (0,25,50,75,100) of the CC0 original were visually annotated **before inference**, only for the primary foreground worker and clearly visible bare head: 10 boxes. Fixed labels `datasets/construction/video-primary-worker-labels.json`, SHA256 `89f4695308b8039caa1efb224fe53fd4dce72788ac012ab3118187a300ae8b50`. The file records the subject/size scoring region, annotation protocol and per-image hashes. These five correlated frames are a tiny non-blind human/agent annotation diagnostic, never an independent test, per-site safety approval, or rate of real noncompliance. Labels were not used to choose a threshold or weights.

Crack still-image evaluation and unlabelled raw-video execution do not establish crack-video accuracy. There is no claim that we found data for every Saudi construction process or that any detector covers all code requirements.

## Measured speed and accuracy

The completed paired benchmark is `output/evaluation/construction-benchmark.json`. Each mode used the same original 640×640 weights, confidence .25, class-aware NMS IoU .45 and matching IoU .5. No tuning used the test labels.

| Diagnostic | CPU ONNX Runtime | Intel UHD 620 OpenVINO GPU | Accuracy before → after |
|---|---:|---:|---|
| PPE, all 169 labelled public test images | 9.61 serial FPS; P95 139.20 ms | 20.04 serial FPS; P95 54.46 ms | TP 631 / FP 185 / FN 254 identical; precision 77.33%, recall 71.30% |
| Surface crack, all 112 supplied test images | 6.84 serial FPS; P95 182.54 ms | 13.78 serial FPS; P95 79.52 ms | TP 101 / FP 29 / FN 47 identical; precision 77.69%, recall 68.24% |
| Crack foreground segmentation | Pixel IoU .616858, mean image Dice .728275 | Exactly the same | TP 288639 / FP 83968 / FN 95311 foreground pixels; background true negatives excluded |
| Genuine 131-frame CC0 video, model + tracking + annotation | 9.81 serial profile FPS; P95 117.43 ms | 19.40 serial profile FPS; P95 57.79 ms | Both engines yield the same tiny labelled-frame failures below |

**The actual moving-video diagnostic exposed a major failure.** In the five frozen primary-worker frames, the model missed the visibly bare head in all five: NO-Hardhat TP 0 / FP 0 / FN 5. It also failed to match the primary worker's whole-person box: Person TP 0 / FP 2 / FN 5. It detects other people and explicit no-vest candidates in the same genuine footage, but this does not offset the primary-worker misses. The dataset image recall is not representative of this video. Faster inference does not make this a dependable safety system. Both CPU and GPU preserved these failures, not only the image-set successes. No claim of construction inspection coverage, reliable missing-PPE recall or Saudi site qualification is justified.

Original source video has 25 FPS; the 19.40 serial profile FPS measurement is below that. Offline exports process all131 frames and preserve their order and original nominal frame rate (index/FPS timing rather than passthrough container PTS); processing takes longer than the source duration. A real-time 25 FPS stream requires visible sampling/drop accounting or additional acceleration; it cannot honestly be described as inspecting every source frame at camera speed.

Original inference outputs are `output/construction-observed-real-video-baseline.mp4` and `output/construction-observed-real-video-optimized.mp4`; the source frames are unchanged before processing. The annotated still `output/construction-observed-real-frame-optimized.jpg` shows actual model predictions, including its misses. These are no simulation, no generated video and no static-image loop.

Model loading, first cold inference and warm timing are separated in JSON. Model timing includes preprocessing, inference, decode and masks, excluding image disk read. Video profile timing additionally includes tracking/annotation, excluding video decode/encode. Those timings exclude capture scheduling, queueing, evidence persistence and preview. The separate shared-runtime video report measures those concerns. These measurements are serial capacities, not a promise that every frame of a faster camera is inspected.

## Reproduce

```powershell
.venv\Scripts\python.exe -X utf8 scripts/setup_construction.py --data
.venv\Scripts\python.exe -X utf8 scripts/evaluate_construction.py
.venv\Scripts\python.exe -X utf8 -m pytest tests/test_construction.py -q
```

Setup explicitly downloads pinned files; subsequent inference is local. Use profile `construction`, real `file:` / `camera:` inputs, and profile options `mode: ppe` or `mode: crack`, `engine: openvino`, `openvino_device: GPU`, `stage: <catalog stage id>`. Demo input is rejected. `supported_capabilities` lists only the selected model's capabilities; `code_compliance` is always `not_determined`.

Unit tests use small tensors and controlled stubs only to validate contracts, NMS, masks, source rejection, event deduplication and absence-of-PPE semantics. Those tests are not the visual proof: downloaded genuine videos and original annotated images are the proof inputs.

## Licensing limits

The PPE author declares CC BY 4.0 weights, but the downloaded ONNX metadata says AGPL-3.0 and the architecture/export derives from Ultralytics. The crack author explicitly licenses AGPL-3.0. Do not infer proprietary redistribution clearance from the PPE card alone. Model/dataset source attribution and original archive notices are retained. Public accessibility does not remove applicable licence obligations.

## Aspect ratio investigation after the genuine-video failure

The pinned PPE card describes stretch preprocessing, while [Ultralytics prediction documentation](https://docs.ultralytics.com/modes/predict/) normally preserves aspect ratio by letterboxing. Compressing a portrait 1080×1920 recording into a square changes body/head geometry. We therefore added an explicit `ppe_preprocess` option (`stretch` or `letterbox`) rather than silently changing model semantics.

Before running the comparison, `scripts/evaluate_construction_geometry.py` fixed its selection rule: maximum macro F1 across Person, Hardhat and Safety Vest on **all 170 separate supplied VALID images**, confidence .25/NMS .45/matching IoU .5 unchanged; exact tie chooses letterbox for aspect preservation. It freezes the selected geometry and validation file hashes before replaying the 169 test images and already-seen five labelled video frames. Test/video replays after failure investigation are reused diagnostics, not a fresh independent holdout. The original stretch benchmark and annotated exports are retained.

Results are recorded separately in `output/evaluation/construction-ppe-geometry.json`. All 170 VALID images are already 640×640, so the two geometries produced **identical** TP 701 / FP 187 / FN 293 and macro F1 .74222138. These square images provide no evidence to rank geometry on portrait footage; the predefined aspect-preserving tie-break chose letterbox. The choice was frozen before diagnostic test/video replay.

On 169 TEST images, counts remain TP 631 / FP 185 / FN 254. The selected GPU run achieved 18.71 serial FPS (P95 62.08 ms). On all 131 original video frames, selected letterbox achieved 19.30 serial profile FPS (P95 56.00 ms). On the five already-seen primary-worker frames, Person improved from TP 0 / FP 2 / FN 5 to TP 3 / FP 2 / FN 2; NO-Hardhat improved from TP 0 / FP 0 / FN 5 to TP 1 / FP 0 / FN 4. **Bare-head recall is still only 1/5 in this tiny reused diagnostic.** The geometry correction helps but does not qualify this detector as dependable safety monitoring.

Letterbox is now the application default; explicit `ppe_preprocess: stretch` reproduces the original baseline. The original benchmark script deliberately retains its explicit stretch protocol. Reproduce the selection and final replay with `python scripts/evaluate_construction_geometry.py`. Final annotated export: `output/construction-observed-real-video-optimized-letterbox.mp4`; original stretch exports are retained. A change in geometry is explicitly distinguished from unchanged-weights CPU/GPU acceleration.

## Actual raw-video execution beyond the short PPE clip

The following are genuine original recordings, processed in source order with every selected source frame used once. This is offline execution evidence; it does not imply camera-rate coverage or video accuracy.

| Source segment | Source / analysis dimensions | Frames | Content duration | Warm serial profile FPS / P95 | Offline video pipeline time |
|---|---|---:|---:|---:|---:|
| Raw bridge `bd_vid_0031.mp4`, 0–5s, crack mode | Original 3840×2160; analysis capped at 1920×1080, then model letterbox 640 | 150 | 5.009s | **6.69 FPS / 223.54 ms** | **32.60s**, excluding 14.91s model load |
| NIOSH original interview, 10–20s, PPE letterbox | Original and analysis 1920×1080; model letterbox 640 | 240 | 10.010s | **19.05 FPS / 60.18 ms** | **19.28s**, excluding 11.96s model load |

Profile timing includes preprocessing, inference, output masks, tracking and annotation. The full offline pipeline additionally includes original video decode, input scaling and annotated export; it does not include model construction or the final source-file hash read. The first inference is included in pipeline time; the serial profile average excludes the first frame. Export is downscaled to a 1280-pixel long edge for viewing. Original source files remain unmodified, and JSON records coordinate mapping and per-frame source time.

**The bridge clip exposed false candidates on steel joints and edges.** The network produced 423 candidate boxes across 149 of the 150 frames, but these are not 423 verified cracks. Visual review of the first annotated frame shows beam joints highlighted as cracks; such proposals require rejection by a competent reviewer. Fine concrete hairline indications may disappear at the 640-pixel model input. No video ground truth was available to quantify field precision/recall. The separate Zenodo image files have independent `bd_img_*` names, without a verified frame/timestamp mapping, so their labels were not transferred to the video by matching numeric suffixes. The crack model's .616858 image-set IoU does not establish reliability on this raw bridge footage.

Artifacts:

- `output/construction-bridge-real-crack.mp4`, `output/construction-bridge-real-crack.json`, and actual annotated frames `output/construction-bridge-real-crack-frame-0.jpg` / `...-frame-75.jpg`.
- `output/construction-niosh-real-ppe.mp4`, `output/construction-niosh-real-ppe.json`, and original-model annotated frames at offsets 0 / 120.
- An additional original raw bridge clip is available at `datasets/construction/bridge_raw/bd_vid_0009.mp4`: 3840×2160, 914 frames, 30.028496 FPS, 30.438 seconds; SHA256 `f74413de3a3997108435754b7c462f06072d2a695ca265f503afdb89f4ebcf7b`. It is not included in the 150-frame model execution above. Re-fetch one original ZIP entry using `scripts/fetch_construction_bridge_video.py --member bd_vid_0009.mp4`; per-clip source, CRC and licence files are retained beside it.

```powershell
.venv\Scripts\python.exe -X utf8 scripts/benchmark_construction_video.py --source datasets/construction/bridge_raw/bd_vid_0031.mp4 --mode crack --seconds 5 --max-edge 1920 --output output/construction-bridge-real-crack
.venv\Scripts\python.exe -X utf8 scripts/benchmark_construction_video.py --source datasets/construction/framer-fall-protection-public-domain.webm --mode ppe --start 10 --seconds 10 --max-edge 1920 --output output/construction-niosh-real-ppe
```

OpenVINO and imageio-ffmpeg versions are separate concerns: inference uses the application environment's OpenVINO 2026.1.0; optional browser-compatible H.264 transcoding uses **imageio-ffmpeg 0.6.0 only in `.venv-roads-export`**, with bundled FFmpeg 7.1. No new runtime dependency or cloud service was introduced for encoding.

## Explicit material ROI and optional box-only video rendering

`profile_options.roi` accepts normalized `[x1,y1,x2,y2]` in the analysis frame. The operator selects the target material; the model does **not** identify concrete automatically. Only that crop goes through inference. Boxes are translated back to full analysis-frame coordinates; dense masks, when enabled, are drawn inside the corresponding crop. All space outside the ROI remains unassessed. This does not add a pass/fail code check, automatic tracking of the material, calibrated dimensions, or an image classifier for material types.

One example concrete-panel ROI, `[.59,.15,.73,.60]`, was selected visually from the **original unannotated 3-second bridge frame**, based on material boundaries rather than model output. The right edge was tightened from .75 to .73 by visual review to exclude a steel sliver before ROI inference. Selection protocol, original-frame hash and source hash are recorded in `datasets/construction/bridge-concrete-roi.json`; the material-only selection image is `output/bridge-source-03-material-roi.jpg` (original image plus a manually drawn ROI rectangle, not a model prediction). Because the camera moves, this fixed ROI cannot be assumed to follow concrete through the entire clip. It is an operator setup example, not evidence that steel was recognized or that the whole bridge was inspected. Full-frame false-positive evidence above is retained.

`profile_options.mask_rendering: boxes` skips dense crack-mask reconstruction and overlay. It retains the same ONNX computation, confidence threshold, class-aware NMS and box candidates. `masks` remains the full-mask option used for pixel IoU/Dice. Box-only output cannot claim pixel-mask performance. The pinned neural network is unchanged; rendering choice does not make the model a concrete classifier.

Seven focused construction tests pass, including an exact candidate-count/score/box parity check on fixed model-output tensors between `masks` and `boxes`, and checks that operator ROI boxes/masks map back correctly without mutating the model output. These small contract fixtures are unit tests only; they are not simulated proof of real inspection accuracy. Real-video timing of these new options is recorded separately when run.

## Box-only parity, material-ROI replay and shared-runtime measurements

A real recheck of **all 112 original crack test images** in `boxes` mode produced the same TP 101 / FP 29 / FN 47 as the saved full-mask GPU baseline. Every per-image ground-truth count, prediction count and true-positive count matched. Box-only timing was 13.95 serial FPS, P95 90.08 ms; the earlier full-mask image benchmark was 13.78 FPS, P95 79.52 ms in a separate timing window. This small-image comparison does not establish a meaningful speed improvement; the options are separately recorded rather than choosing a best timing. Pixel metrics are omitted for boxes mode. Report: `output/evaluation/construction-boxes.json`.

The frozen concrete-material ROI was then applied to an actual **2.5–3.5 second** raw bridge excerpt, with source 4K capped to 1920×1080 analysis and no frame looping/generation. All 30 selected source frames were processed once. There were 34 proposals across 26 frames, **not 34 verified concrete cracks**. Profile throughput was 13.79 FPS, P95 85.63 ms; processing/export took 5.80 seconds for 1.002 seconds of footage, excluding 14.95 seconds of model load. Report and original overlay: `output/construction-bridge-concrete-roi.json` / `.mp4` / `...-frame-15.jpg`.

**The material ROI did not demonstrate a detector fix.** In the actual midpoint overlay, the prominent thin horizontal crack-like feature around analysis pixel (1250,465), inside the concrete ROI, is not boxed. A low-score .25 proposal appears instead near the lower-right panel/joint boundary around (1370,630). This is a visible qualitative miss/mislocalization, with no material-specific ground truth or measured video recall to resolve it. The ROI only records operator scope; it is not automatic material segmentation, proof of concrete-crack detection, or an acceptance decision. Original whole-frame steel-joint false-positive examples and the selected-ROI failure remain available.

The parent also measured the actual shared runtime, with real source-clock pacing and Python socket connections denied:

| Actual input / requested sampling | Processed / captured | Source frames intentionally skipped | Runtime decision P95 | Result |
|---|---:|---:|---:|---|
| NIOSH real video, 15s excerpt, requested 15 FPS, 200ms decision budget | **225 / 225**, zero queue drops, late results or inspection errors | **134** | **65.5 ms** | Sustained selected-frame processing; about 14.8 FPS including shutdown overhead. This does not inspect all source frames and does not establish safety accuracy. Station initialization 14.03s. |
| Raw 4K bridge, 10s excerpt, requested 10 FPS, boxes, 200ms decision budget | **32 / 32**, zero queue drops, late results or inspection errors | **267** | **128.6 ms** | **Failed the requested throughput: about 3 FPS**, 10.8s run, with source lag around 404ms. Station initialization 14.16s. Queue-drop zero does not erase source coverage gaps. |

Raw 4K bridge capture/decode remains a bottleneck even when analysis is capped and mask reconstruction is skipped. A model-only FPS number is not the video application's throughput. See `output/evaluation/construction-runtime-ppe.json` and `output/evaluation/construction-runtime-crack-boxes.json` and the actual source-paced runtime benchmark script.

An offline source-proxy preparation tool is provided at `scripts/prepare_construction_video_proxy.py`. It area-downscales and H.264-encodes one original frame per proxy frame, with explicit nominal-FPS timestamp normalization, full decoded frame-count/FPS/duration checks, source/output hashes and loss-of-detail scope. It does not generate scenes, synthesize/interpolate frames, add model overlays, or establish raw 4K real-time conversion. Any later proxy throughput measurement must identify the lower-resolution input separately and must not assume equal crack accuracy.

```powershell
.venv\Scripts\python.exe -X utf8 scripts/evaluate_construction.py --mode crack --mask-rendering boxes --engine gpu --skip-video --output output/evaluation/construction-boxes.json
.venv\Scripts\python.exe -X utf8 scripts/benchmark_construction_video.py --source datasets/construction/bridge_raw/bd_vid_0031.mp4 --mode crack --start 2.5 --seconds 1 --max-edge 1920 --mask-rendering boxes --roi .59 .15 .73 .60 --output output/construction-bridge-concrete-roi
```

The H.264 viewing-copy helper `scripts/transcode_construction_videos.py` uses explicit FFmpeg `-fps_mode passthrough`, then fully decodes both source and temporary target using OpenCV. It checks decoded frame counts, nominal FPS, frame-derived duration (within 2ms), and dimensions; a mismatch rejects the temporary result before publishing the final `-h264.mp4`. Its manifest records both decoded inventories and file hashes. H.264 viewing copies are lossy encodings of the original actual inference overlays; they never supply the pixel-IoU measurements.

## Verified browser-playable viewing copies

All five actual inference exports were converted in one helper call, then every source and target was fully decoded using OpenCV 5.0.0. Each pair preserves decoded frame count, FPS, frame-derived duration and dimensions; targets were published only after these checks passed. Full hashes, source/target inventories and encoding scope are in `output/construction-h264-manifest.json`.

| Actual export and H.264 viewing copy | Frames before / after | FPS before / after | Duration before / after |
|---|---:|---:|---:|
| Original stretched site clip: `output/construction-observed-real-video-baseline-h264.mp4` | 131 / 131 | 25 / 25 | 5.240 / 5.240s |
| Selected letterbox site clip: `output/construction-observed-real-video-optimized-letterbox-h264.mp4` | 131 / 131 | 25 / 25 | 5.240 / 5.240s |
| Whole-frame bridge false-candidate example: `output/construction-bridge-real-crack-h264.mp4` | 150 / 150 | 29.946 / 29.946 | 5.009 / 5.009s |
| NIOSH real interview observations: `output/construction-niosh-real-ppe-h264.mp4` | 240 / 240 | 23.976 / 23.976 | 10.010 / 10.010s |
| Concrete ROI miss/mislocalization example: `output/construction-bridge-concrete-roi-h264.mp4` | 30 / 30 | 29.946 / 29.946 | 1.002 / 1.002s |

These are lossy H.264/yuv420p viewing copies with fast-start metadata and original frame order, not new model inference or new ground truth. No frames were invented/interpolated to make the examples look real-time. All original prediction JSON and original MPEG-4 annotation exports remain available, including negative examples.

```powershell
.venv-roads-export\Scripts\python.exe -X utf8 scripts/transcode_construction_videos.py --input output/construction-observed-real-video-baseline.mp4 --input output/construction-observed-real-video-optimized-letterbox.mp4 --input output/construction-bridge-real-crack.mp4 --input output/construction-niosh-real-ppe.mp4 --input output/construction-bridge-concrete-roi.mp4
```

## Completed, verified 720p source proxy

`datasets/construction/proxies/bd_vid_0031-720p.mp4` is now available for a **separate lower-resolution input** benchmark. It contains all **364 actual original frames**, 1280×720, original nominal FPS 29.94625059368 → proxy 29.94625059326, and frame-derived duration 12.1551110000 → 12.1551110002 seconds. Full source and proxy were decoded to verify counts, and FFmpeg reported **zero duplicated and zero dropped frames**. It is 11,862,706 bytes, SHA256 `6d8d5f675e4fef252f5a363dbb9678a044d414d0cb1dccbfa454b58b69f7724f`.

Verification and source attribution are in `datasets/construction/proxies/bd_vid_0031-720p.proxy.json`; a CC BY 4.0 attribution/change notice is adjacent at `bd_vid_0031-720p.LICENSE.md`. The source remains the unmodified Tsegaye raw bridge recording. No model overlays or ground-truth labels were added to this input proxy.

**Timestamp normalization is explicit.** The original vendor file's presentation timestamps drift slightly relative to its reported nominal frame rate/duration. A direct passthrough prototype preserved 364 frames but failed strict duration/FPS validation and was not published. The final proxy maps original decoded frame ordinal i to proxy frame ordinal i at the original nominal FPS; no image/frame is generated, duplicated, dropped or interpolated. The original last decoded PTS is 12126.290ms; proxy last PTS is 12121.718ms. Consult the original recording for original capture PTS. The original irregular PTS are not claimed preserved.

Only spatial area downscaling, lossy H.264 coding, explicit nominal-FPS timestamp normalization and audio omission occur. Detail can be lost, so the crack detector's accuracy is **not assumed preserved**. This is offline source preparation, not a demonstration of live 4K-to-720p conversion or raw 4K real-time inspection. Actual original full-frame/ROI detection failures remain documented and viewable.

```powershell
.venv-roads-export\Scripts\python.exe -X utf8 scripts/prepare_construction_video_proxy.py --source datasets/construction/bridge_raw/bd_vid_0031.mp4 --output datasets/construction/proxies/bd_vid_0031-720p.mp4 --max-edge 1280
```

## Actual source-paced 720p proxy runtime result

The frozen local crack model was separately measured on the verified 1280×720 source proxy, using the same requested **10 FPS**, 10-second excerpt, box rendering, GPU and 200 ms decision budget as the failed raw 4K run. Python socket connections were denied during this measurement. Report: `output/evaluation/construction-runtime-crack-720p.json` (input SHA256 `6d8d5f675e4fef252f5a363dbb9678a044d414d0cb1dccbfa454b58b69f7724f`).

The source-paced proxy run captured and processed **100 / 100 scheduled frames**, with **zero queue drops, late results or inspection errors**, P95 decision time **84.9 ms** and maximum **141.7 ms**. It skipped **197 original source frames** under the explicit sampling policy. Overall session throughput was **9.8 FPS including drain/shutdown overhead**, replay lag was **28.37 ms**, and station initialization took **15.58 seconds**. This met the scheduled 10 FPS excerpt; it is neither a 15 FPS result nor inspection of every original frame.

For comparison, the actual unmodified raw 4K source under the same requested 10 FPS processed only **32 / 32 captured frames**, about **3 FPS**, with **267 source skips**, P95 128.6 ms and roughly 404 ms replay lag. The proxy result demonstrates a viable lower-resolution input path on this laptop; it does **not** establish live 4K downscaling speed. The proxy was prepared offline, is spatially downscaled/lossy, and uses the explicitly documented nominal-FPS timestamp normalization. No paired accuracy study establishes that its crack accuracy equals the original. Existing steel-joint false positives and the concrete-ROI qualitative miss remain unresolved.
