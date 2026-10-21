# Road model local optimization

The same pinned five-class YOLO12s weights are exported at smaller fixed input sizes. This retains a distinct **Repair** class; the faster four-class nano alternatives remain research options because their previous diagnostic accuracy and pothole/repair semantics were insufficient.

No training, confidence fitting, dataset calibration or cloud inference is introduced. Confidence remains .25, per-class NMS .45, and four-class damage matching uses class-aware IoU .5 against original VOC XML. The original 640 export is preserved byte-for-byte. Model creation and evaluation use installed local artifacts; device-specific measurements identify CPU or the laptop's integrated GPU explicitly.

## Measurement protocol

`scripts/benchmark_roads_optimization.py` compares each configuration on the same fixed 28 public images and their original annotations. Five warmups are discarded. ORT uses three timing passes (84 observations), the initial OpenVINO CPU comparison two (56), and initial GPU comparison one (28); accuracy counts every original image only once. Profile timing includes letterbox, FP32 inference on the explicitly requested CPU or local GPU, decode, tracking and annotated-image generation; file read, session construction, camera exposure, source decoding, browser/JPEG preview and actuation are excluded. These short sequential tests do not establish sustained video throughput or physical line coverage.

The 640/two-thread control uses the prior spinning behavior. Smaller candidates use ORT sequential execution, full graph optimization and non-spinning workers. All candidate sizes/thread counts are selected before their first diagnostic scores; selecting the final preset using these results creates development selection bias. The model author has not published training membership, so the 28 images may overlap training. They are a compatibility diagnostic, not independent accuracy.

Reports include per-image SHA256, original XML SHA256, predictions, original truth, TP/FP/FN, repair observations, full timing samples and model provenance. Python socket connections are denied while evaluating. Timing is coordinated with other agents so model benchmarks do not run concurrently on this shared laptop. Browser/service inference is stopped for the exclusive measurement window.

## Measured before and after

These sequential experiments used this laptop's Intel i7-8550U (four physical/eight logical CPUs), existing Balanced power plan and charger. No power, driver or system settings changed. Other project model inference was paused. Windows background activity and thermal variation remain possible; the new 640 control differs from the historical 1.68 FPS run, so fresh comparisons use the new 1.99 FPS control.

| Configuration | Timing samples | p50 / p95 ms | Mean process FPS | TP / FP / FN | Precision / recall | Repairs |
|---|---:|---:|---:|---:|---:|---:|
| Before: ORT CPU 640, two threads, spinning | 84 | 467.0 / 872.6 | 1.99 | 29 / 8 / 6 | 78.4% / 82.9% | 2 |
| Portable default: ORT CPU 640, four threads, non-spinning | 84 | 391.6 / 543.1 | 2.53 | 29 / 8 / 6 | 78.4% / 82.9% | 2 |
| Balanced: ORT CPU 416, four threads, non-spinning | 84 | 177.7 / 258.8 | 5.73 | 29 / 5 / 6 | 85.3% / 82.9% | 2 |
| Fast tradeoff: ORT CPU 320, two threads | 84 | 111.5 / 160.6 | 9.04 | 22 / 4 / 13 | 84.6% / 62.9% | 2 |
| Fast tradeoff: ORT CPU 320, four threads | 84 | 81.6 / 115.7 | 12.16 | 22 / 4 / 13 | 84.6% / 62.9% | 2 |
| OpenVINO CPU 640, four threads, FP32 | 56 | 443.5 / 678.6 | 2.15 | 29 / 8 / 6 | 78.4% / 82.9% | 2 |
| OpenVINO CPU 416, four threads, FP32 | 56 | 158.4 / 272.3 | 6.21 | 29 / 5 / 6 | 85.3% / 82.9% | 2 |
| Laptop Intel GPU 416, OpenVINO FP32 | 28 | 64.5 / 70.8 | 15.44 | 29 / 5 / 6 | 85.3% / 82.9% | 2 |
| Normal installed app, Intel GPU 416, OpenVINO FP32 | 84 | 61.3 / 65.0 | 16.31 | 29 / 5 / 6 | 85.3% / 82.9% | 2 |

Accuracy always counts **28 images and 35 original damage boxes**, once per configuration, regardless of timing repetitions. All variants retain a separately named Repair class; the two observed repair detections are excluded from damage TP/FP/FN and are not independently scored as correct repairs. No repair-ground-truth precision/recall is claimed.

At 416, every image retained its 640 TP and FN counts on this diagnostic, while FP changed: China_Drone_000109 increased from two to three, China_MotorBike_001563 dropped from one to zero, Czech_001476 dropped from one to zero, and United_States_000420 dropped from two to zero. This is **observed development parity, not an assurance that reducing resolution preserves field recall**. The 320 option missed seven additional labeled defects, so it is never chosen as a silent optimization. The portable no-options default stays 640/four-thread CPU, and explicit balanced/GPU presets name their 416 input.

The portable default improves mean process throughput **1.27×** over the fresh control with unchanged diagnostic counts; explicit CPU416 is **2.88×**, while the normally installed GPU416 profile is **8.20×** on 84 observations (the initial isolated-module comparison was 7.76× on 28). Sequential measurement and thermal/background variation mean these are observed comparisons, not a sustained speed guarantee. OpenVINO CPU640 did not beat ORT CPU640/four-thread performance, and its CPU416 tail was slightly worse than ORT; it is not adopted as the CPU default.

Initial GPU station construction, including first graph compilation in the isolated comparison, took **21.93 seconds**, excluded from per-frame timing. Capture starts after model construction. No first-frame or camera exposure claim follows from this warm profile benchmark. The installed normal-app validation and actual runtime stress are recorded separately below.

## Installed application and runtime verification

The normal application environment repeated the same 28-image comparison over three timing passes. The report records `.venv/Scripts/python.exe`, `.venv/Lib/site-packages/openvino/__init__.py`, **no isolated package-path override**, OpenVINO build `2026.1.0-21367-63e31528c62-releases/2026/1`, and execution device `GPU.0`. Counts stayed 29 TP / 5 FP / 6 FN plus two separately named repair observations. Normal-app profile p50/p95 were 61.25/64.96 ms, mean throughput 16.31 FPS. Model initialization took 21.14 seconds separately. This confirms the integrated package path rather than relying only on the export environment experiment.

`scripts/benchmark_roads_runtime.py` then exercised actual `EdgeRuntime` capture pacing, latest-frame buffer, model, asynchronous previews and local event storage for **30 seconds at 12 FPS with a 150 ms deadline**. It cycles the same 28 hash-verified real public images as predecoded arrays, capped to the runtime's 1,920-pixel long side and copied for every capture. No model-result cache or synthetic pixels are substituted. This is varying software replay, not a real moving road, physical acquisition benchmark, tracking validation, or additional independent accuracy data.

| Runtime measurement | Observed |
|---|---:|
| Captured / processed | 360 / 360 |
| Dropped / late / inspection errors | 0 / 0 / 0 |
| Coverage gaps / pending queue | 0 / 0 |
| Host receipt to decision p50 / p95 / p99 | 66.8 / 79.5 / 85.9 ms |
| Maximum host receipt to decision | 95.6 ms |
| Session average capture / process | 11.9 / 11.9 FPS, including 0.2-second drain |
| Encoded previews / preview errors | 139 / 0, at a 6 FPS maximum |
| Saved evidence images | 0 |

The **first model inference is included** in these runtime timing/deadline counts; runtime warmup frames were not discarded. Model construction is reported separately and occurs before capture. Local source copies occur before the host-receipt timestamp, but still contend for this laptop's memory/CPU during the test. JPEG encoding runs asynchronously and is not part of the host-to-decision timer; its concurrent load is present. Python `socket.socket.connect` was denied for the entire process, including model construction and inference. No physical firewall/network settings changed and native-library network access was not audited separately. Runtime state records cloud off, raw media sync off and recording off.

The 12 FPS/150 ms GPU preset is retained because this run processed all 360 scheduled frames within the deadline. It is a short measured workload, **not a deterministic deadline guarantee**, camera-to-decision measurement or field qualification. Other programs, thermal throttling and camera decoding can change results. The code still reports missed/late/dropped work rather than concealing it.

Evidence: [full normal-app runtime report](../output/optimization/roads-runtime-gpu12.json). Repeated dataset frames are not new accuracy samples. Camera behavior and factory stress are documented separately by the shared runtime verification.

Full evidence, including misses and non-favorable cases:

- [ORT control and four candidates](../output/evaluation/roads-optimization-ort/report.json)
- [OpenVINO CPU comparison](../output/evaluation/roads-optimization-openvino-cpu/report.json)
- [Initial Intel GPU comparison](../output/evaluation/roads-optimization-openvino-gpu/report.json)
- [Normally installed app GPU comparison](../output/evaluation/roads-optimization-openvino-app/report.json)

Each report folder also preserves annotated pothole and repair examples for that configuration. The original 640 reports and images remain unchanged in `output/evaluation/roads-real-fiveclass`.

## Exact artifacts

| Export | ONNX SHA256 |
|---|---|
| 640 baseline | `72d63019c84fadfc9777cd56734d3f79a12ca40f18dfccf352b8c3d7ad810d37` |
| 416 candidate | `e6634f04c1965dda75930af5a4d9ec2b090e4e013ff183ca53dc490aa040e034` |
| 320 candidate | `34ef7efd2fce0af8a017c8d2fe8c839afdd7aee4894f071604026905beb3ee4c` |

All derive from checkpoint SHA256 `ccf91e3199d81ee761f4a3a012da752939c9d72e9b1eaa242e508db3163270a0`. Candidate manifests beside the model retain the existing source, tool, class and license provenance and state their spatial input. FP32 model size stays roughly 37 MB; reducing resolution reduces spatial computation, not parameter count. Runtime validates hashes and fixed input/output contracts and never silently resizes a fixed 640 graph into a smaller graph.

## Reproduce

```powershell
.venv-roads-export/Scripts/python.exe -m scripts.optimize_roads_exports --sizes 320,416
.venv/Scripts/python.exe -m scripts.benchmark_roads_optimization --configs 640:2,640:4,320:2,320:4,416:4 --repeats 3
.venv/Scripts/python.exe -m scripts.benchmark_roads_optimization --configs 640:4:openvino,416:4:openvino --repeats 2 --name roads-optimization-openvino-cpu
.venv/Scripts/python.exe -m scripts.benchmark_roads_optimization --configs 416:4:openvino:GPU --repeats 3 --name roads-optimization-openvino-app
.venv/Scripts/python.exe -m scripts.benchmark_roads_runtime --seconds 30 --fps 12 --deadline 150 --output output/optimization/roads-runtime-gpu12.json
.venv/Scripts/python.exe -m pytest tests/test_roads_model.py tests/test_roads.py -q
```

Use `profile_options` keys `input_size` (320, 416, 512 or 640), `cpu_threads` (1–8), and `cpu_spinning` (boolean). A matching validated export/manifest must exist; 512 is an explicit export option, not an installed artifact. Only the five-class model supports smaller input sizes. Missing exports fail with an actionable error.

The OpenVINO comparison initially loaded only its module from the isolated export environment while using the application's Python, NumPy 2.2.6 and OpenCV 5.0.0, making the existing ORT data path comparable. Its configuration requests explicit FP32 precision, latency mode, and one CPU stream, following the [official CPU scheduling documentation](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device/performance-hint-and-thread-scheduling.html). One reusable infer request executes synchronously. GPU uses the latency hint and one in-flight request. Enumeration found both `CPU` (i7-8550U) and `GPU` (Intel UHD Graphics 620); actual GPU inference and the returned `GPU.0` execution device were then verified on the 28 cases.

The official PyPI `openvino==2026.1.0` wheel is now installed normally in `.venv`, pinned in `requirements.txt` and `requirements-lock.txt`; NumPy stayed 2.2.6. `pip check` is clean. Its resolved dependency `openvino-telemetry==2025.2.0` is recorded in the lock; application code invokes no telemetry API. Offline benchmarks deny socket connections. The OpenVINO wheel's Apache-2.0 license/notices are installed with the package; existing model and dataset license obligations remain unchanged. Application setup does not install Torch, Ultralytics or ONNX export tools.

`engine: openvino` defaults to `openvino_device: CPU` and rejects unavailable requested devices. GPU must be selected explicitly. There is no `AUTO` device choice, driver installation, or silent cloud fallback. If a GPU start fails, choose the separately configured ORT CPU preset; the runtime does not conceal a failed accelerator behind different timing behavior.

This laptop's GPU plugin rejected the CPU-style integer `NUM_STREAMS` option before inference during the first integration attempt. The final GPU path uses its supported latency hint and one reusable synchronous request; only CPU receives an explicit one-stream property. The succeeding 28-image report uses that corrected path. Thread-count options bound CPU model workers and do not configure GPU compute threads. Performance is measured with one station active; simultaneous factory, OCR and road workloads are not covered.

Explicit laptop GPU settings:

```json
{"engine":"openvino","openvino_device":"GPU","model_variant":"fiveclass","input_size":416,"cpu_threads":4}
```

Portable balanced fallback settings:

```json
{"engine":"onnxruntime","model_variant":"fiveclass","input_size":416,"cpu_threads":4}
```

Use 4 FPS and a 500 ms deadline for the balanced CPU preset. The explicit laptop GPU preset uses **12 FPS and a 150 ms deadline**, verified by the normal-app runtime stress above. Camera exposure, source decoding and display remain outside host-to-decision timing. Neither preset guarantees every physical defect is visible long enough.

Quantization is not represented as completed: uncalibrated INT8 on this attention/convolution graph can fail CPU operator support or damage small-crack accuracy. A calibrated quantized model requires a disjoint calibration set and a fresh held-out evaluation. Neither the 28 diagnostic images nor their labels are used to calibrate weights.
