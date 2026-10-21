# Guided construction inspection interface

The Construction profile now offers **Walls and finishes**, **Brackets and railings**, and **Tile installation**, alongside the existing construction model selector. Selecting a task does not establish that its detectors have passed validation. Each task displays the category states and reasons returned by `GET /api/construction/tasks`; if that endpoint is unavailable the fallback explicitly identifies unverified capabilities.

## Capture and references

Guided tasks use a normalized material ROI (`roi: [left, top, right, bottom]`) in full-frame coordinates. Left must be less than right and top less than bottom, with all values between zero and one. Tile specifications are optional `nominal_joint_mm` and `tolerance_mm`; absence is preserved rather than filled with an invented installation tolerance.

Advanced reference inputs merge into `profile_options` as `calibration_reference` and `expected_fasteners`. The task, ROI and reference controls take precedence over matching advanced JSON keys when a guided task is selected. Existing model options and legacy presets remain available through **Existing construction models**.

The **Local research model** selector lists only the compatible installed manifests returned in `research_models` by `/api/construction/tasks`. It defaults to **No research model selected**. A selection sends `profile_options.research_model_manifest`; task changes, legacy selection and incompatible catalog entries clear that choice. State and preset restoration can recover a compatible installed selection, including a stopped guided session on page reload. The selector takes precedence over the same key in advanced JSON, and shows the model's source classes and unreleased status. Guided controls always remove older `research_model` options; an empty selection removes both research-model keys, so advanced JSON cannot silently run a detector behind “No research model selected.” The backend's direct API compatibility remains separate. Installing weights never automatically selects or validates them.

A scale reference supplies the measured DICT_4X4_50 marker size/id, `same_plane`, `image_size: [width, height]`, camera matrix and distortion coefficients. The camera calibration must match the analysis resolution. The UI supplies no fabricated camera calibration. Expected fastening positions are an array of `{id, bbox}` records with normalized full-frame bounding boxes; an expected position alone does not prove a missing fastening.

## Preview and inspected evidence

Guided selection defaults to 30 input FPS, 30 preview FPS and a 1,000 ms deadline. These are requested settings, not achieved performance claims. The raw preview is separate from the last inspected image. Preview FPS and inspected FPS are displayed individually using runtime metrics. The age indicator estimates the age of the still actually shown, rather than borrowing the newer backend result age. When a task is unavailable, its processing rate is explicitly labelled **Guidance · no detector**. Preview updates are deduplicated using the atomic `X-Frame-Sequence` and `X-Session-Id` response headers; repeated JPEG delivery does not count as another displayed source frame.

The evidence image is requested using the sequence/session from `state.inspection_result`:

```text
/api/inspection.jpg?sequence=<sequence>&session_id=<session>
```

An unavailable or mismatched image is not replaced with another frame. The displayed evidence timestamp/status come from the same captured result that requested that image, rather than a newer `last_result`. The inspected still is labelled separately from the continuing raw preview. Session changes clear both images and revoke their object URLs. A fetched JPEG is decoded before its image and caption are published together; late replies cannot replace a newer sequence or another session. Running guided tasks poll state every approximately 200 ms with a request-overlap guard and immediately request the corresponding still after receiving state. Legacy and hidden-page state polling stays at 500 ms. The independent 250 ms evidence fallback retains its single-request guard.

Findings display task, assessment status, source-video evidence timestamp and optional measurements when supplied. No-result text remains an observation rather than construction acceptance.

## Verification

- `node --check edge_runtime/static/app.js` passed after the changes.
- Real-browser Edge verification against the actual app at `http://127.0.0.1:8765/` exercised task switching, backend catalog statuses, source selection, guided start, live paired evidence and EOF. No camera was accessed.
- A 6.011-second observation window on the supplied damp-wall video produced **99 unique preview image loads = 16.47 FPS**, after sequence/session deduplication, and **12 paired inspected-image loads**. This was the unavailable wall-task guidance path, **not a trained dampness detector benchmark**. The runtime's encoded-preview rate remains a separate measurement. A preliminary 30.29 image-load FPS result counted duplicate JPEG responses and was rejected.
- Before the preview polling change, the frontend polled every 170 ms, imposing a theoretical 5.88 updates/second ceiling. This is a code-derived ceiling, not a measured before-run result.
- Browser QA exposed a live sequence/image race. The runtime now keeps a bounded inspected-image cache so a fetched state snapshot can retrieve its exact image. The final UI test observed live evidence advancing to frame 198 at video time 6.7 seconds; an earlier run had returned only conflicts until EOF. Missing/mismatched sequences continue to be rejected.
- Reversed material ROI bounds were rejected before starting another session, with the message “Material area must have left < right and top < bottom, within 0–1.” The prior session identifier stayed unchanged. Controls were restored afterward.
- An inspected screenshot is stored at `output/playwright/construction-guided-preview-evidence.png`. Startup preview requests can return 503 until the first local frame is encoded; no JavaScript exceptions were observed.

## Final research-model browser verification

The installed catalog returned seven compatible research models during this check: two PHELE wall detectors, a wall-stain/leakage SSDLite detector, two SDNET box detectors and the two genuine-polygon SDNET segmentation candidates. Walls-to-brackets and brackets-to-walls switching cleared incompatible model choices. The browser restored an explicitly selected PHELE SSDLite320 manifest and the supplied local damp-wall video after page reload, including a stopped session. The selected model remained research-only.

A separate no-model replay counted **71 processed guidance frames, 71 guidance frames, zero inspected frames and zero model inferences**. Its guidance rate was never presented as detector throughput. The real PHELE replay produced no findings on the supplied damp-wall clip; the UI displayed **unassessed**, not a humidity-detection success. Private owner footage was used only as a diagnostic replay.

### Displayed-frame age correction and polling optimization

The earlier age label reported the backend's newest inspected result while the browser could still be showing an older still. It now adds the host monotonic timestamp difference between the paired result and the displayed image, plus browser time elapsed since receiving state; a 100 ms tick continues to age a stopped image. The label is explicitly approximate. It excludes camera exposure/buffering and cannot establish sensor-to-display latency.

For independent browser verification, each measurement fetch recorded its round-trip time and used the server's snapshot monotonic time minus the displayed image's capture timestamp. This bounds the displayed host-receipt age at response completion between that difference and that difference plus the fetch RTT. Percentiles below are descriptive bounds from small samples, with no synchronized browser/server clocks assumed.

| Version, same real damp-wall clip and PHELE SSDLite | Unique displayed preview FPS | Displayed-age p95 lower-upper | Valid age samples |
|---|---:|---:|---:|
| Correct age measurement; 500 ms state and independent 250 ms still polling | 20.54 | 1,008-1,040 ms | 19 of 20; one awaiting initial evidence |
| Guided 200 ms state polling plus immediate paired still request | 18.49 | 606-668 ms | 20 of 20 |

The first run **failed the one-second displayed-age target**. The optimized run met it in this observation window and retained more than 15 unique displayed preview FPS. CPU load changed: the earlier run overlapped stress work, which had ended before the second run; the separate SDNET v2 fit remained active. This is **not a controlled isolated performance comparison or a production qualification**. The optimized run's newest-backend age p95 was 313 ms, which is substantially less than displayed-image age and cannot substitute for it. Maximum measurement fetch RTT was 173 ms.

The optimized unique preview count was **117 images over 6.328 seconds**; duplicate frame/session header pairs do not update the image. Inspection throughput remained separate from preview throughput. The evidence cache peaked at 32 frames and 462,048 JPEG bytes, within its 32-frame / 8 MiB bounds. No live sequence conflicts or JavaScript exceptions occurred; six startup preview requests returned 503 before the first image became available.

Versioned receipts preserve both the initial misleading-age run and the corrected measurement:

- output/playwright/construction-browser-before-age-fix.json preserves the earlier 25.05 unique-preview-FPS observation, before the age-label correction.
- output/playwright/construction-browser-before-poll-optimization.json and its summary preserve the corrected-age failure.
- output/playwright/construction-browser-after-poll-optimization.json and its summary contain the final timed samples.
- output/playwright/construction-final-after-poll-optimization.png is the final full-page screenshot.
- output/playwright/construction-final-evidence-detail.png was inspected at readable resolution: raw video and timestamped evidence are separate, the selected walls task is unassessed, and no detections were fabricated.

The application was left stopped on the real owner damp-wall video with PHELE SSDLite explicitly selected. No camera was opened. Final JavaScript syntax verification passed.

## Final idle verification after all model fitting

A fresh page on the final server selected **Walls and finishes** with a local-video source, **30 input FPS / 30 preview FPS / 1,000 ms deadline**, and **No research model selected**. Source-specific category explanations distinguish MBDD stain/leakage appearance, PHELE Mold and peeling labels, and genuinely unavailable tasks. This fresh default check occurred before starting any inference.

The final catalog contains **eight research-only models**, including SDNET v2. Brackets-and-railings options include the newly published v2 manifest alongside the earlier detector and mask comparisons. Selecting another task clears incompatible choices. No category was promoted to a validated release. After checking these choices, the UI returned to walls and finishes with PHELE SSDLite selected.

With all training, exports, audits and parent model benchmarks idle, the final real owner damp-wall replay yielded:

| Measurement | Observed result |
|---|---:|
| Unique browser preview frames / window | 136 / 5.724 s |
| Unique displayed preview FPS | **23.76** |
| Runtime encoded-preview FPS | 29.65 |
| Model / inspected FPS | **12.88** |
| Displayed host-receipt age p95, 20 valid samples | **425-442 ms bounds** |
| Newest backend result age p95 | 183 ms |
| Model inference p95 | 115.6 ms |
| Complete inspection p95 | 139.3 ms |
| Model inferences / dropped decoded frames | 85 / 110 |
| Source skips / inspection errors | 0 / 0 |
| Maximum evidence cache | 32 frames / 467,846 JPEG bytes |

The preview and displayed-age goals were met in this finite diagnostic window; inspection itself remained below 15 FPS. Dropped decoded frames mean every-frame coverage was not achieved. The replay produced no findings, and remains an unassessed damp-wall diagnostic rather than a validated humidity success. The 5.724-second measured window is reported exactly rather than described as a six-second sustained qualification.

The paired result includes persisted model provenance under inspection_result.model.guided_model_provenance: model/manifest paths, SHA-256, thresholds, input size, source/license, ONNX Runtime CPU engine, one CPU thread, and research-only validation state. The inspected image and caption stayed paired. Only initial preview requests returned 503 before the first available frame; no live 409 or JavaScript exception was observed.

The existing damp diagnostic ID now resolves to all **330 original source frames**, with original variable presentation intervals relative to the first frame. Its evidence metadata reports a maximum encoded timestamp error of **0.000444 ms** against a 1 ms tolerance. This metadata was read without playing another video during the idle benchmark.

Additional receipts:
- output/playwright/construction-idle-preparation.json records fresh defaults and native-PTS metadata.
- output/playwright/construction-browser-final-idle.json contains the exact timing samples and paired server state.
- output/playwright/construction-browser-final-idle-summary.json summarizes the bounds, metrics and nested model provenance.
- output/playwright/construction-final-catalog-check.json records all eight installed manifests and the five bracket model choices.
- output/playwright/construction-final-idle-evidence-detail.png was visually inspected: the real source preview, separate exact inspected frame and unassessed status are legible.

The app was left stopped with the owner damp-wall source and PHELE SSDLite selected. No camera was opened.
