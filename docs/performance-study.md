# Laptop real-time optimization study

This iteration targets faster local inference, especially factory inspection. The laboratory stress target is **30 input frames/second with a 100 ms host decision budget**. Those are development targets, not a specification for a physical production line. Actual item rate, camera exposure/transport, trigger association and rejection timing still require measurement on that line.

## Fair comparison

The package, scripts and configuration were snapshotted under `output/optimization/before` before shared-runtime edits. Parallel module preparation had already started, so this folder is not claimed to be a pristine copy of every original module. Factory comparison explicitly selects the retained original `sklearn` execution path; road controls explicitly use the original 640-pixel/two-thread settings and unchanged weight hash. Dataset images, factory learned parameters and decision thresholds were frozen. Model workers prepared changes in parallel but ran timing experiments in explicitly exclusive CPU windows. Reports distinguish experiments, the selected configuration and rejected speed/quality tradeoffs. Baseline and optimized measurements use the same inputs and timing scope within each paired comparison.

Host: Windows 11, Intel i7-8550U, four physical/eight logical cores, approximately 23.9 GiB RAM, Python 3.13.5, OpenCV 5.0.0. Windows reported a 2,001 MHz current/max clock during preparation; this is a snapshot, not continuous frequency telemetry. Power policy was Balanced, battery was charging at 64%. No OS power policy or graphics driver was changed. Installed Intel UHD 620 availability is assessed separately from CPU comparisons rather than assuming a CUDA device.

Accuracy comparisons are scoped:

- Factory uses the same 80 item-held-out Kolektor images and unchanged learned forest. Compare full profile statuses, original-coordinate boxes and every threshold pixel, not just average FPS. Below-threshold cascade score maps are lower bounds; they must not be substituted for exact score maps in ranking/AUROC studies.
- Freight uses the identical six public images, 12 selected invoice field checks and 58 annotated receipt-word substring checks. Invoice fixtures are externally authored synthetic data; three receipt photos are real but outside supported freight document types. These checks are not CER/WER or representative shipment accuracy.
- Roads uses the same 28 fixed public images with original VOC annotations, confidence .25 and class-aware IoU .5. The model author's training membership is unknown, so numbers remain compatibility diagnostics with possible overlap. A faster lower-resolution configuration must show its actual missed/false-detection tradeoff.

See [factory optimization](factory-performance.md), [freight optimization](freight-performance.md), and [road optimization](roads-optimization.md) for detailed artifacts and reproduction.

## Shared runtime changes

JPEG encoding no longer runs in the capture or inspection thread. A dedicated media worker owns one replaceable preview slot and at most four queued event images, with a 32 MiB raw-image budget including in-flight data. The existing bounded event writer remains responsible for SQLite/media storage.

Preview sampling defaults to an upper limit of six images/second and a 960-pixel long side. It can be set to 0–15 FPS independently of inspection input rate. This sampling does **not** skip model inference. Preview replacements/throttling are separate counters from captured inspection drops, late frames and missed units. API state exposes preview sequence/age and media queue/encoding state. Displayed preview pixels can lag the latest decision; the overlay belongs to that preview's frame.

Selected event images retain their original annotated resolution and immutable payload/session identity. A full media queue retains the finding metadata and marks the absent image explicitly; encoding errors also retain metadata. Stop/switch waits for the bounded media worker, so an old session cannot publish an image into a new one. Camera waiting-for-trigger preview uses the same worker.

Decision timing already excluded JPEG encoding in the earlier build. Moving JPEG work therefore chiefly removes interference and improves total processing capacity; it does not obtain a misleading latency gain by changing the timer endpoint. Encoding time and media state are reported separately. Source decoding, camera exposure/backend buffering and physical actuation remain outside the host-receipt-to-decision timer.

## Runtime baseline

`scripts/benchmark_runtime.py` runs a fixed software clock over a predecoded real held-out positive surface image. It executes actual model inspection on every processed observation, without result caching. Repeated stills isolate runtime/model capacity; they are not independent accuracy cases, camera acquisition tests or proof of item coverage.

Before: eight seconds at requested 30 FPS and 100 ms deadline captured **240 frames**, processed **22**, and dropped **218**. Processed rate including drain was **2.6 FPS**, median inference **323.7 ms**, host P95 **475.5 ms**. There were 110 late counts, including stale frames rejected before inference as well as completed late decisions; these categories overlap dropped/processed counters and must not be added together. The last result was late; no timely clear outcome was manufactured.

Report: `output/optimization/runtime-before.json`. Baseline code root is recorded in the JSON. The benchmark supports `--current-profile` with the old runtime to isolate runtime changes while holding the optimized model implementation constant.

## Model results on the same inputs

| Module / scope | Before | Selected optimization | Accuracy check |
|---|---|---|---|
| Factory complete profile, 80 images | p50 315.52 ms; p95 479.99 ms | Native CPU: p50 35.02 ms; p95 53.27 ms (**9.01× p95 speedup**) | Identical threshold masks, boxes, statuses and candidate scores; 8 TP / 3 FN / 1 FP / 68 TN |
| Freight six pages, one cold + five warm | 43.319 seconds total | 19.289 seconds (**2.25×**); warm p95 11.771 → 4.856 seconds | Same 12/12 fields and 55/58 annotated-word checks |
| Roads complete profile | Original CPU640: 1.99 observations/s; p95 872.6 ms | Explicit Intel GPU416: 16.31 observations/s; p95 64.96 ms (**8.20× mean throughput**) | 29 TP / 8 FP / 6 FN → 29 TP / 5 FP / 6 FN on 28 development images |

The road GPU comparison changes execution device and spatial input resolution, using the same FP32 weights and confidence/NMS thresholds. Final validation uses the normal application environment and 84 timing observations, matching the CPU control's repetition count; the initial shorter isolated-module GPU study is also preserved. At 416 every diagnostic image retained its TP/FN counts, but reducing resolution can lose small defects on unseen data. Default unconfigured roads remains CPU640 with four non-spinning threads (2.53 observations/s, unchanged counts); CPU416 is an explicit balanced fallback. CPU320 reached 12.16 observations/s but lost seven additional annotated defects and is not selected as the default. OpenVINO CPU did not improve the preferred CPU configurations enough to replace ORT.

Factory has no resolution/model/threshold reduction. Below-threshold score-map values change to documented lower bounds; retained candidate scores and decisions remain exact. Freight retains full recognition crops and model weights; compact experiments losing annotated words were rejected. OCR still takes seconds per new page; fast frame guidance and page result latency are different measurements.

## Factory runtime stress

The matched eight-second, 30 FPS / 100 ms test improved from **22 to 191 processed frames** (2.6 → 23.0 FPS including drain), with drops reduced **218 → 49**. Host p95 fell **475.5 → 151.5 ms**, but 141 stale/completed-late counts remain. This target **failed** and is not offered as a reliable 30 FPS mode.

Holding the optimized profile constant while restoring the old runtime processed 186 frames (22.4 FPS), 54 drops, p95 152.1 ms. This small runtime-only comparison is within short-run variation; the native model provides most of the measured speedup. The media refactor primarily isolates blocking encoding and bounds its memory/work, rather than claiming a large additional model acceleration.

A separate 60-second varied-image test at **15 FPS / 100 ms** processed **900/900**, with **zero drops and zero inspection errors**, p95 **57.9 ms**, p99 **117.9 ms**, max **164.6 ms**. There were **11 late decisions**. It establishes useful processing throughput while **failing a strict 100 ms deadline**. A separately identified 200 ms budget is evaluated below; it must not be confused with meeting the initial target.

Reports:

- `output/optimization/runtime-after-30.json`
- `output/optimization/runtime-old-with-current-model.json` (hybrid module paths are recorded)
- `output/optimization/runtime-varied-15.json`

The final **15 FPS / 200 ms** varied-image run processed **900/900** in 60 seconds, with **zero drops, zero late decisions and zero inspection errors**. Host p50/p95/p99/max were **36.2 / 51.5 / 75.9 / 106.9 ms**. The factory public presets now use this explicit input rate and budget, with native `auto` selection. This is an observed short laptop run, not a guaranteed deadline on Windows or an every-item factory acceptance test.

The separate real `LocalSource` still path (initial decode followed by cached-frame copies, rather than the predecoded benchmark adapter) processed **120/120** at 15 FPS / 200 ms over eight seconds, with **zero drops/late/errors**, p95 **58.2 ms**, max **105.5 ms**. Repeating a supported still verifies integration, not motion detection accuracy or camera acquisition.

- `output/optimization/runtime-varied-15-200ms.json`
- `output/optimization/runtime-local-source-15.json`

Every run keeps preview enabled at six FPS and event-image saving disabled. There is no result caching or intentional inspection sampling. Input quality/features/forest are evaluated on each processed observation. `station_start_ms` records initialization outside the active timing clock in the new reports; percentile samples include the first processed frame. Varied runs cycle the same 80 images repeatedly, so 900 frames are not 900 independent accuracy cases.

For reproduction, run one timing job at a time:

```powershell
.venv/Scripts/python.exe scripts/benchmark_runtime.py --output output/optimization/runtime-rerun.json --seconds 60 --fps 15 --deadline 200 --varied
.venv/Scripts/python.exe scripts/benchmark_runtime.py --output output/optimization/runtime-local-rerun.json --seconds 8 --fps 15 --deadline 200 --local-source
.venv/Scripts/python.exe scripts/benchmark_camera_capture.py --seconds 5 --fps 30
```

## Actual laptop camera acquisition

The five-second transport-only probe read **153 frames at 30.28 observed receipts/s**, using MSMF. First-frame startup took **15.03 seconds**. The driver accepted a 30 FPS request and reported 30 FPS. It accepted width/height setter calls but actually negotiated **640×480**, despite the requested 960×540. Setter success is therefore not proof of the requested mode.

Read p50/p95/p99 were **32.11 / 47.30 / 49.22 ms**. Buffer-size-one was requested but rejected; the driver's reported buffer property is not a verified exposure/queue latency. No camera images were encoded, returned or saved. This probe tests transport only; it does not apply the factory model to a qualified product view, measure camera exposure-to-decision delay or associate physical units.

The real camera source now makes a best-effort FPS request and exposes request acceptance, driver-reported FPS/dimensions, observed first-frame geometry and buffer negotiation in `description.capture_properties`. Those values survive reset/calibration. See [camera property contract](camera-capture-properties.md), [raw probe](../output/optimization/camera-capture.json) and [media lifecycle regression](runtime-media-verification.md).

## Installed road GPU runtime

The ordinary `.venv` OpenVINO installation was validated on all 28 diagnostic images, with 84 warm timing observations. Counts stayed **29 TP / 5 FP / 6 FN**, with two separately reported repair observations. Full-profile p50/p95/p99/max were **61.25 / 64.96 / 65.78 / 66.15 ms**, mean serial capacity **16.31 observations/s**. Report: `output/evaluation/roads-optimization-openvino-app/report.json`.

The actual shared runtime then cycled the 28 public images for 30 seconds at **12 FPS / 150 ms**, with previews enabled and local event storage active. It captured/processed **360/360**, with **zero drops, zero late decisions, zero coverage-gap counts and zero inspection errors**. Host p50/p95/p99/max were **66.8 / 79.5 / 85.9 / 95.6 ms**. There were 139 preview encodes, zero media errors, and no event-image saving. The clock-average rate including drain was 11.9 FPS. First inference was included; model construction/graph compilation took **20.54 seconds before capture** and is reported separately.

The report verifies `.venv/Lib/site-packages/openvino`, actual `GPU.0` execution, the 416 export hash and `socket.connect` denial. There is no isolated-module path override in this final validation. These are clocked still-image stress runs, not moving-road tracking accuracy or vehicle-speed coverage. The explicit Intel GPU public preset uses **12 FPS / 150 ms**; the separately selected CPU416 fallback uses **4 FPS / 500 ms**. If the requested GPU is unavailable, startup reports an error rather than silently substituting a slower device.

Evidence: [GPU runtime report](../output/optimization/roads-runtime-gpu12.json). Reproduce with `.venv/Scripts/python.exe scripts/benchmark_roads_runtime.py`; run it without other active inference jobs.

## Regression verification

`python -m pytest -q` passed **112 tests in 28.34 seconds**, with one existing third-party Starlette/AnyIO deprecation warning. Two newly added mocked GPU configuration/unavailable-device checks were also explicitly rerun and passed. `pip check`, Python compilation of runtime/scripts and JavaScript syntax checking passed.

Tests cover native/Python/baseline feature and candidate parity, preserved reference mode, real OCR and asynchronous lifecycle, model shapes/hashes, road decoding/repair semantics, preview/evidence memory bounds, blocked JPEG isolation, stop/switch ownership, terminal trigger outcomes and camera property/reset races. Installed-model network checks deny Python connections; no physical network setting was changed. These functional checks supplement the scoped dataset accuracy measurements rather than replacing them.

Module detail: [factory](factory-performance.md), [freight](freight-performance.md), [roads](roads-optimization.md), [media](runtime-media-verification.md), [camera properties](camera-capture-properties.md). Raw reports preserve non-favorable candidates and failed targets as well as the selected results.

## Dashboard verification and final state

The refreshed local dashboard loaded the factory public preset with 15 input FPS, 200 ms and native `auto`; API evidence confirmed `native_cascade` and the expected native-code hash. A real interactive run processed **285/285** in **19.0 seconds**, with **zero drops/late/errors**, p95 **82.4 ms**, p99 **166.0 ms** and maximum **197.8 ms**. This higher GUI-run tail is preserved alongside the quieter software-clock measurements, not replaced by them. It illustrates why a short isolated p95 is not a deadline guarantee.

Setting Preview FPS to zero showed **Preview disabled** while inspection continued (51 processed observations, zero drops/late at the observed UI check). Preview was restored to six FPS and the final station was stopped explicitly. No browser console errors appeared during this verification. The server remains available on loopback; camera, inspection worker and image saving are off. The final preview/result and frozen timing metrics remain for review.

Evidence: `output/optimization/dashboard-state-stopped.json`, [dashboard screenshot](../output/optimized-edge-dashboard.png). The image is a public held-out still; it is not a factory camera feed. These short replay tests do not establish sustained thermal stability, physical trigger/reject timing or an acceptable customer defect miss rate. Factory still misses 3/11 defective holdout views; speed changes did not improve that recall.
