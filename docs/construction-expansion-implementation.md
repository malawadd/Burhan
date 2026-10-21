# Construction inspection expansion

21 October 2026. This is the implementation/evidence index for the approved three-mode plan. The application runs locally at `http://127.0.0.1:8765`. **Research models and measurements are not released construction inspection capabilities.**

Follow-through after the failed delivery: [what was implemented, why work stopped too early, actual training/data repairs and measured results](construction-resumption.md). The later record distinguishes working application behavior from the still-unfinished fault detectors.

## Implemented application behavior

- Guided **walls and finishes**, **brackets and railings**, and **tile installation** modes, with 30 tracked checks. Walls/finishes is the fresh Construction selection; previous model presets remain available.
- Normalized material ROI, expected fastening positions, optional printed-marker/camera calibration, and specified tile joint width/tolerance. Invalid or incompatible input is rejected. Calibration must match the actual frame resolution, including rotation/resizing.
- Task-specific local CPU-ONNX research model selection. Model files are SHA256 checked, fixed float32 input geometry is checked, and all adapters use one CPU inference thread. Empty model selection runs capture guidance, not a hidden detector.
- Independent unannotated live preview and exact timestamped inspected image. A bounded paired-image cache resolves metadata/image request races; session changes invalidate old evidence. Preview images include frame/session identity so repeated JPEG loads do not inflate unique browser FPS.
- Separate counters for processed guidance, actual inspected frames, and neural inference. A rejected/unavailable frame cannot inflate inspected FPS. Geometry processing is identified separately from neural inspection.
- Actual inspected/model-frame latency percentiles exclude guidance calls. Persisted guided findings retain the selected model's checksum, class operating points and task provenance rather than relying on the station's transient description.
- Actual native masks, when supplied by a mask model, are rendered at their real crop-local coordinates and exported as measured pixel regions/contours. Box-only models remain identified as box localization.
- Real-frame recording comparisons and original-input presets are explicitly diagnostic; no ground-truth match is invented for the four private owner videos.

## Public interfaces

`GET /api/construction/tasks` returns three task families, category statuses/reasons, unsupported physical checks and hash-verified installed research-model choices. No current category is promoted automatically from a trained checkpoint.

`POST /api/start` keeps the existing schema and accepts additive `profile_options`: `inspection_task`, `roi`, `research_model_manifest`, `calibration_reference`, `nominal_joint_mm`, `tolerance_mm`, `expected_fasteners`. Existing legacy options continue to work without `inspection_task`.

`GET /api/state` adds `inspection_result` (the exact encoded still's metadata), `preview_kind`, bounded cache statistics, actual inspection/guidance/model counters and evidence age. `GET /api/preview.jpg` supplies `X-Frame-Sequence` and `X-Session-Id`. `GET /api/inspection.jpg?sequence=N&session_id=S` returns only that retained pair, HTTP409 for a stale pair/session, or HTTP503 before an image exists.

Findings add optional `task_id`, `evidence_timestamp_ms`, `measurements` and `assessment_status`. Video timestamps identify source moments; camera observations also retain the runtime's host-receipt monotonic timestamp and sequence, without asserting camera exposure time.

## Data, experiments and evidence

| Workstream | Evidence record |
|---|---|
| All 30 checks, category-specific data gaps and unsupported physical conclusions | [Category catalog](construction-expansion-category-catalog.md) |
| All local model recipes, frozen budgets, selection and CPU continuation | [Training recipes](construction-expansion-training-recipes.md) |
| Per-category native localization counts and fixed-point before/after tables | [Benchmark overview](construction-expansion-benchmark-summary.md) |
| 826 real SDNET connection photos; actual boxes/polygons; no corrosion labels | [Public-data audit](construction-expansion-data-audit.md) |
| Actual Batalha efflorescence mask/photo acquisition; installed-tile, sealant and paint-blister source gaps | [Additional primary-source audit](construction-expansion-additional-data-leads.md) |
| Real installed-floor runtime crash repaired; image-space band measurements and unchanged damage-model transfer failure | [Tile geometry resumption](construction-resumption-geometry.md) |
| YOLOv8n versus SSDLite fastening fits, before/after localization, Fixed-only negatives, stronger continuation | [Fastening benchmarks](construction-expansion-sdnet-benchmarks.md) |
| Genuine polygon-mask YOLO segmentation versus LR-ASPP, native masks and export verification | [Mask comparison](construction-expansion-sdnet-masks.md) |
| 169 real PHELE photos; peeling and publisher Mold kept separate from damp-only inspection | [Wall-finish experiment](construction-expansion-phele.md) |
| Direct leakage/stain appearance, with new photo evaluation and protected diagnostics excluded | [Wall-stain experiment](construction-expansion-wall-stain.md) |
| Retained crack/spalling results and actual before/after videos; no accuracy transfer to new categories | [Round4 evaluation](construction-round4-measured-results.md), [owner-video review](new-construction-video-review-round4.md) |
| Shared runtime before/after and unique browser preview measurement | [Runtime benchmarks](construction-expansion-runtime-benchmarks.md) |
| Capture/calibration, conservative fastener logic and geometry limitations | [Guided backend](construction-guided-tasks.md) |
| Controls, research-model selection, state restoration and browser checks | [Interface](construction-expansion-ui.md) |

The initial owner fastening replay is a real H.264 original/model comparison: `output/evaluation/construction-guided-expansion/owner-fastening-ssd-v1/original-vs-research.mp4`. All207original frames are decoded, inspected and re-encoded; 25 frames contain unscored candidates. This is not 25 correct detections. The fixed midpoint snapshot still misses the visibly separated plate. No physical severity, fastening adequacy or success percentage is assigned. The recording is in the app's **failure diagnostics**, with the matching original source preset.

The additional owner damp-wall replay, `output/evaluation/construction-guided-expansion/owner-damp-stain-ssd-v1-native-pts/original-vs-research.mp4`, contains all 330 actual frames with their original variable presentation intervals. The new stain model at its frozen CAL-selected 0.25 threshold produces candidates on 330/330 frames. The fixed midpoint is cluttered with overlapping proposals instead of a useful localized assessment. This is a failed diagnostic, not proof that dampness was detected correctly. Both the recording and matching original-input preset are available under failure diagnostics; neither owner's clip was used to tune the model. Its timing-correction receipt verifies all input hashes and predictions equal the preserved nominal-rate version; maximum encoded timestamp error is0.000445ms. The initial fastening replay used nominal frame-rate timing; the updated renderer preserves original presentation timestamps for further comparisons.

The final [three-panel fastening comparison](../output/evaluation/construction-guided-expansion/owner-fastening-ssd-v1-vs-yolo-v2-native-pts/original-vs-research.mp4) contains every207original frame with timestamp error below0.000445ms: original / SSD v1 / extended YOLO512. All207views pass capture gating. The earlier model gives candidates on25frames; the extended model gives none. The owner's visible mounting-plate gap is still missed/unassessed, so this is explicitly a **failed diagnostic**, not a false-alarm success. The app includes its matching original-source preset. Full numbers and the inspected midpoint are in the [extended-fit record](construction-expansion-sdnet-v2-results.md).

A [printable ArUco reference](../assets/construction-reference/aruco-4x4-50-id0-50mm.svg) supports the optional calibration workflow. It does not calibrate any camera by itself, provide measurement validation or retrospectively establish millimetres in the private videos.

## Release boundaries

`edge_runtime/construction_validation.py` computes separate per-category gates: localization precision ≥95%, recall ≥90%, ≥50 independent positive and negative originals, verified source-group split, complete negative labels, appropriate rights, real-video localization/event validation, and measured runtime targets. Metric tasks additionally need independent physical measurement validation. A photo accuracy score cannot grant video release.

Initial public-data fits fail quality and/or data qualification. Publisher annotations may not exhaustively mark the opposite fastening class; annotation-unmatched predictions are distinct from verified physical false alarms. Publisher Fixed-only views are therefore reported separately. No model is silently installed as a successful default.

Tile geometry currently finds parallel visible edges. It does not identify grout or prove a joint defect. Valid calibration permits explicitly **experimental** marker-plane millimetres, without an acceptance alarm. Missing grout, crooked installation, chipped installed tiles, sealant, paint coverage and reveals stay unavailable until suitable labels and independent evaluation exist.

Humidity percentage, active hidden leakage, mold species, torque, embedment, concealed anchorage, hollow tiles, adhesion, floor slope and lippage remain outside RGB inference. The construction application does not issue Saudi-code acceptance.

## Reproduction

Run source acquisition/preparation before training; the linked experiment documents preserve source hashes, fixed splits, initializer provenance, budgets, threshold selection, export checks, all unsuccessful results and exact commands. Training uses `.venv-roads-export`; the deployed runtime uses `.venv` and requires no Torch or cloud API. Media rendering uses the existing export environment's PyAV/H.264 encoder and does not turn photographs into claimed real videos.

Verify software with `.venv\Scripts\python.exe -m pytest -q` and JavaScript with `node --check edge_runtime/static/app.js`. Synthetic unit fixtures verify software mechanics only, never construction accuracy. Final measured verification is recorded below after all workstreams complete.

The first complete integration run passed **566 tests, 9 skipped** in 91.69 seconds. One existing Starlette/AnyIO deprecation warning remains. The JavaScript syntax check and Python compilation also passed. Subsequent reviewed changes and the final runtime/UI measurements are recorded in the linked documents; these software checks do not qualify any detection category.

After loader, task-isolation, telemetry and provenance review, the full suite passed **580 tests, 9 skipped** in 79.71 seconds. The log is `output/construction-expansion-pytest-final-reviewed.log`; the same existing deprecation warning remains. Model-training completion and final quiescent short-clip measurements are additional evidence, not replacements for failed accuracy gates.

Final delivery checks after project fitting/export ended: eight checkpoints each deliver29.02–29.70encoded preview fps on short actual owner clips, while actual model rates range7.23–25.91fps. A real browser window measures**23.76unique preview fps**,**12.88inspected fps** and**425–442ms displayed-age p95 bounds**; it drops110decoded inspection frames and produces no findings on that unassessed damp-wall input. The same-weight legacy/guided runtime comparison raises preview3.86→29.40fps but reduces actual inspected throughput3.75→2.99fps after quality gating/additional delivery work. These are measured delivery results, not proof of reliable fault detection or sustained15fps inspection. The [runtime record](construction-expansion-runtime-benchmarks.md) preserves all failures, samples and workload context.

After registering the final recordings, the targeted recording/preset regression checks passed3tests. Current JavaScript syntax, selected Python compilation and all local links in the construction-expansion Markdown records also pass. `output/construction-expansion-final-verification.json` binds the test logs, current runtime/config hashes, all eight graph/manifest hashes, hardware, final runtime/browser receipts and stopped application state. It records30tracked checks and **zero released categories**. The app remains available locally with the owner damp-wall file and an explicitly selected PHELE research model; no camera is opened.
