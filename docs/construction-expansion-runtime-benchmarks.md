# Construction expansion: real-video runtime benchmark

21 October 2026. These measurements concern video delivery and inspected-frame latency. **They do not establish better model accuracy or a working humidity/fastener detector.**

## Fixed input and comparison

Local owner recording `C:/Users/IAM/Documents/hackathon/opencv/videos/VID-20261020-WA0000.mp4`, 269 original decoded frames. No generated damage, looped input, or slideshow. Its SHA256, complete profile options, samples and counters are in `output/evaluation/construction-guided-expansion/runtime-wall-crack/report.json`.

Before: legacy crack profile, existing ONNX crack model, ONNX Runtime CPU, one inference thread. After: `walls_finishes`, explicitly wrapping the **same** crack model as research, with capture-quality gating and independent raw preview. Both requested 30 input/preview fps with a 1,000 ms decision deadline. The clip ends naturally; acquisition clocks exclude model preparation. These consecutive runs happened during development with other desktop services present, so they are not thermally controlled capacity measurements.

| Metric | Before | After |
|---|---:|---:|
| Encoded preview fps, measured delivery timestamps | 2.54 | **17.61** |
| Inspected frames/s over session | 2.4 | 2.6 |
| Original source frames captured | 269 | 269 |
| Inspected frames | 24 | 26 |
| Queue drops / coverage gaps | 245 | 243 |
| Late results / inspection errors | 0 / 0 | 0 / 0 |
| Host-receipt-to-decision p95 | 503.5 ms | 471.1 ms |
| Sampled displayed-evidence age p95 | Not separately available | **847.97 ms** |
| Sampled process working-set peak | 216.32 MiB | 320.01 MiB |

The after run passes the **encoded preview ≥15 fps** and **sampled evidence age ≤1 second p95** targets. It does not inspect every frame at 15 fps. Higher memory includes two bounded JPEG workers, separate images and model allocations within the same benchmark process; it is not an isolated model memory delta. Runtime evidence age measures host receipt, not camera exposure or browser rendering. Browser delivery is checked separately.

## Behavior and bounds

- Guided capture submits raw frames directly to the preview encoder, while inspection retains the existing latest-frame queue. Every dropped inspection frame is counted.
- Annotated images use a separate encoder and `/api/inspection.jpg`. `inspection_result` is published with its exact image sequence/session. The app labels the inspected still separately from live video.
- A maximum 32 recent image/result pairs, with an 8 MiB JPEG budget, protects the state-request/image-request race. A missing pair returns HTTP409; the API never substitutes another frame's boxes. At least the newest image is retained if one individual JPEG exceeds the budget. Metadata is additionally bounded by 32 records.
- Preview supports 0–30 fps; zero disables live preview without disabling inspections/evidence. Construction UI guided defaults request 30 fps to leave timing margin above 15 delivered fps.
- Reopening a session clears the evidence cache; closing joins both media workers.

## Reproduce

```powershell
.venv\Scripts\python.exe scripts\benchmark_construction_guided.py --video C:\Users\IAM\Documents\hackathon\opencv\videos\VID-20261020-WA0000.mp4 --output output\evaluation\construction-guided-expansion\runtime-wall-crack --seconds 12
.venv\Scripts\python.exe -m pytest tests/test_construction_guided_runtime.py tests/test_construction_release_gate.py tests/test_construction_guided_tasks.py -q
```

The benchmark's final JPEGs are inspected-frame snapshots, potentially at different times. They are **not** advertised as matched defect comparisons. Model accuracy requires independently annotated masks/boxes and real-video event truth. The four owner recordings remain diagnostic-only and are not used for training or threshold selection.

## Sustained delivery under deliberate training load

A later stress run uses the actual public-domain NIOSH framer recording,120seconds per condition from the same original source, with no loop or invented frames. The4-thread SDNET fit remains active at BelowNormal priority and desktop services remain running. This is deliberately contended delivery evidence, not isolated detector capacity or wall-defect validation. The source's native rate is about23.98fps. Script/runtime/media hashes, UTC start, all samples and options are in `output/evaluation/construction-guided-expansion/runtime-sustained-context-under-training/report.json`.

| Metric | Legacy crack path | Guided wrapper, same crack weights |
|---|---:|---:|
| Elapsed acquisition |120.1s|120.1s|
| Encoded preview fps |2.03|**22.63**|
| Captured original frames |2,877|2,877|
| Processed profile calls |243|1,198|
| Actual inspected frames |243|**9**|
| Quality/guidance frames |0|**1,189**|
| Actual inspected fps |2.02|**0.07**|
| Queue drops |2,634|1,679|
| Source skips |1|2|
| Late / errors |2 /0|0 /0|
| All profile-call host latency p95 |702.7ms|156.5ms|
| Sampled displayed backend-result age p95 |Unavailable|314.65ms|
| Process working-set peak |287.17MiB|447.29MiB|

**Most guided views were rejected as unsuitable.** Its314.65ms age is predominantly fresh capture-guidance evidence, not continuous neural inspection evidence. The low inspected count is retained; a fast guidance loop cannot establish15fps detector throughput or sufficient coverage. Legacy models do not expose the new guided model-call flag, so their model-specific counter is not used to infer absence of inference.

This run includes the sampling-cadence correction: ordinary source timing jitter no longer repeatedly resets the preview timer to `now+period`, which previously discarded roughly every second nominal30fps frame. A long idle gap resets the cadence to avoid catch-up bursts. Mechanical rate/jitter tests cover both cases.

Further telemetry separates actual inspected-frame latency and model-call latency percentiles from all profile-call percentiles. When no model ran, those neural percentiles are unavailable, not zero-speed success. This was added after the sustained receipt; its existing all-profile timings are not retroactively relabelled neural timings.

## Browser delivery and displayed age

The real PHELE SSDLite owner damp-wall replay under active training produced162unique image loads in6.466s (**25.05fps**) before the displayed-age correction. Backend full-clip counts were330captured,147actual inspected/model frames,183queue drops,zero source skips/errors. Backend preview was29.72fps and actual model rate12.92fps. Cache stayed within32frames and466,539JPEGbytes. This is a research-model execution benchmark, not evidence that the owner's fault is correctly detected.

The age display originally used the latest backend result while the browser could still show an earlier paired image. The corrected UI ages the **shown** image and decodes its JPEG before swapping image/caption. A subsequent contended sample measured131unique loads in6.379s (**20.54fps**), with19valid displayed-age samples. A server-monotonic/browser-RTT bound puts displayed-age p95 at**1,007.98–1,040.22ms**, whereas latest-backend age p95 was282.3ms. Thus the≤1-second **displayed** age target narrowly **fails** that sample. The earlier backend-only age does not override this failure.

Receipts are versioned under `output/playwright/construction-browser-displayed-age-*.json` and the [UI record](construction-expansion-ui.md). Later scheduling/idle measurements are reported separately, with their workload context; these contended samples are preserved.

The guided UI now requests status every200ms with one request in flight and immediately fetches the corresponding retained still, rather than waiting for independent500ms status and250ms image timers. Legacy status remains500ms; hidden pages do not use the fast path. An older sequence cannot replace a newer shown image. A subsequent real PHELE SSDLite replay measured**18.49unique preview fps** (117loads/6.328s) and20displayed-age samples with p95 bounds**606–668ms**, within both requested targets. The cache stayed at32frames/462,048bytes with no409 or JavaScript exceptions. The stress run had ended while v2 training remained active, so this is **not a controlled attribution of the entire improvement to polling**. Its separate receipts are `output/playwright/construction-browser-after-poll-optimization*.json`.

Hardware receipt: `output/construction-expansion-hardware.json`: IntelCorei7-8550U,4physical/8logical cores,25,682,677,760bytes physical RAM, IntelUHD620 and AMDRadeonR7M340. All new construction adapters use one-thread CPU-ONNX; GPU presence does not imply these models used it.

## Final serial checks after project training/export stopped

All eight verified research checkpoints were run serially on the actual owner damp-wall or fastening clips, each source once per model. All project training, export, numerical parity and photo-audit processes had exited; the app was stopped and the browser was waiting. Unrelated desktop services remained active, and there was no thermal reset after fitting. The process receipt is `output/construction-expansion-quiescence-before-final.json`; pinned models/sources/code and full samples are in `output/evaluation/construction-guided-expansion/runtime-all-models-final-v1/`.

| Research checkpoint | Encoded preview fps | Actual model frames/s | Model frames / captured | Queue drops / source skips | Model-frame profile p95 ms | Backend result-age p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| PHELE SSDLite320 |29.02|21.88|248 /324|76 /6|75.4|218.55|
| PHELE YOLOv8n384 |29.69|14.73|168 /330|162 /0|81.6|169.50|
| SDNET LR-ASPP320 |29.58|13.87|102 /207|105 /0|88.6|194.16|
| SDNET YOLOv8n-seg320 |29.61|15.13|110 /207|97 /0|80.5|158.61|
| SDNET SSDLite320 |29.58|25.91|188 /207|19 /0|45.2|114.11|
| SDNET YOLOv8n320 |29.59|18.89|137 /207|70 /0|60.6|136.28|
| Extended SDNET YOLOv8n512 |29.53|**7.23**|53 /207|154 /0|157.6|294.00|
| MBDD stain SSDLite320 |29.70|22.32|253 /330|77 /0|55.0|128.66|

All eight completed without inspection errors or late results, and all actual processed frames in these clips ran the selected model. Encoded preview meets15fps in every short run; actual model throughput ranges**7.23–25.91fps**, so some models do **not** meet15fps inspection. These are7–11second clips, not sustained field capacity. The separate120second test above mostly produced capture guidance and cannot fill that qualification gap.

The first PHELE replay skipped six source frames and all models dropped some pending inspection frames. Those coverage gaps remain recorded; neither smooth preview nor a low result age establishes every-frame coverage. This benchmark introduces no owner ground truth or model/threshold selection. All accuracy gates remain failed.

Peak process working sets range153.52–362.24MiB across the same serial process; native allocator retention prevents interpreting differences as isolated model memory. Model-frame profile latency includes quality checks, inference, postprocessing and annotation, excludes external decoding/JPEG, and is distinct from browser age. Every model's startup, frame counters, latency percentiles, media queues, exact model provenance and final inspected image are retained in its report. The human table is also saved as `runtime-all-models-final-v1/README.md`.

## Final same-model before/after delivery check

After all fits and the browser measurement stopped, the original269-frame crack clip was replayed with the same legacy crack weights: legacy delivery versus the current guided wrapper. This isolates the delivery configuration more closely than comparing different task models, though consecutive desktop runs are not a controlled thermal experiment. Source/code hashes, all samples and exact options are in `runtime-wall-crack-final-v2/report.json` under the same evaluation directory.

| Metric | Legacy delivery | Guided delivery, identical weights |
|---|---:|---:|
| Encoded preview fps |3.86|**29.40**|
| Captured frames |269|269|
| Actual inspected frames |36|29|
| Guidance-only frames |0|5|
| Actual inspected fps |3.75|**2.99**|
| Queue drops / source skips |233 /0|235 /0|
| Late / errors |0 /0|0 /0|
| Host-receipt-to-decision p95, inspected frames |328.8ms|417.7ms|
| Model-frame profile p95 |Legacy flag uninstrumented|391.9ms|
| Sampled backend-evidence age p95 |Unavailable|686.46ms|
| Process working-set peak |218.84MiB|322.33MiB|

The wrapper improves preview delivery, **not neural throughput or accuracy**. It spends additional work on capture gating, raw preview and the separate paired still; five views are rejected and actual inspected throughput is lower. The two extra queue drops remain visible. Backend-evidence age meets the one-second target on this short clip; no every-frame or sustained detector qualification follows.

## Final browser check after project jobs stopped

Using the actual owner damp-wall source and explicitly selected PHELE SSDLite, the final real-browser window delivers**136unique preview frames /5.724seconds =23.76fps**. Twenty displayed-age samples put p95 at**425–442ms**, including a measured transport-delay bound. Backend latest-result age p95 is183ms and cannot replace the displayed-image value.

The app measures**12.88actual inspected/model fps**,85inferences,110decoded queue drops,zero source skips/errors, model-frame profile p95115.6ms and complete inspected latency p95139.3ms. Browser cache peaks at32frames/467,846bytes; no live sequence conflict or JavaScript exception occurs. Startup preview503s and favicon404 are separate. This short UI diagnostic meets preview/display-age goals while **inspection remains below15fps**, and the replay yields no findings: it is an unassessed owner damp-wall input, not validated humidity detection.

Exact samples, timebase/RTT method and nested graph/threshold provenance are in `output/playwright/construction-browser-final-idle*.json`; the [UI record](construction-expansion-ui.md) retains a legible screenshot. The app is left stopped with that source and explicit research model, without opening a camera.
