# Verification record

**Performance update:** measurements below record the earlier implementation. Current model/runtime changes, before/after accuracy checks, measured presets and regression results are in [the laptop performance study](performance-study.md). Historical results are preserved for comparison.

Build date: 20 October 2026, Asia/Riyadh. Machine-readable reports retain UTC timestamps.

The initial synthetic-runtime record below is retained as a baseline. The later real-input implementation adds local models, OCR and public-data evaluation; see [real-inputs.md](real-inputs.md), the per-module model reports, and [real-input verification](real-input-verification.md) for the current checks. Synthetic timings do not describe model inference.

## Environment

| Component | Observed |
|---|---|
| Development host | Windows 11, OS build 26200 |
| CPU | Intel Core i7-8550U at 1.80 GHz, 8 logical processors |
| Installed memory | Approximately 23.9 GiB |
| GPUs reported by Windows | Intel UHD Graphics 620 and AMD Radeon R7 M340 |
| Actual inference path | CPU; GPU acceleration is not configured |
| Python | 3.13.5 in project-local .venv |
| OpenCV | cv2 reports 5.0.0; opencv-python package pinned to 5.0.0.93 |
| Camera | HP TrueVision HD; frame capture succeeded through MSMF |
| Dashboard | http://127.0.0.1:8765 |

DirectShow failed to open the camera; Media Foundation succeeded. The standalone probe read a 640x480 frame in roughly 4.4 seconds including device initialization. That startup time is not per-frame inference latency. Neither the probe nor the integrated camera check saved images.

## Automated test suite

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Recorded result: **49 passed**, approximately 11 seconds. One third-party Starlette/AnyIO deprecation warning remains; it did not fail the tests.

Coverage includes:

- Factory clean and changed pixels, missing parts/scratches/displacement, calibration, image quality, unsupported registration and event behavior.
- Freight boundaries, motion/crop/quality rejection, fictional-template identification, duplicate suppression, full packet completion, manual capture and reset.
- Roads compact dark candidates, ROI/shape rejection, temporal confirmation, duplicate suppression, timeout and blur rejection. A compact-shadow false positive is deliberately documented.
- Runtime latest/FIFO overload policies, all profiles running with socket connections denied, current JPEG output, deadline-aware trigger confirmation, explicit late/missed trigger handling and missing source errors.
- Local API validation, cross-origin control rejection, export and stopped timing behavior.
- Immutable event snapshots, no-I/O enqueue, bounded byte/event/media retention, orphan cleanup, post-close rejection and SQLite handle release on Windows.

The socket-denial tests establish that the tested local processing paths do not require network connections. They do not represent physically disabling the laptop's network adapter.

## Profile functional and CPU baseline

Command:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_profiles.py --output output\evaluation\profile-baselines.json
```

[Recorded report](../output/evaluation/profile-baselines.json) and [detailed evaluation notes](evaluation-baselines.md).

All 10 authored functional checks passed. The benchmark processed 240 synthetic frames per profile at 960x540, sequentially. Source timestamps simulated a clock; processing was not paced to that rate. The numbers below are process-only observations from that run, excluding source capture, queueing, JPEG encoding, event I/O and browser rendering.

| Profile | p50 ms | p95 ms | p99 ms | Maximum ms |
|---|---|---|---|---|
| Factory reference baseline | 50.19 | 90.48 | 105.26 | 116.46 |
| Freight geometry baseline | 10.09 | 17.76 | 20.26 | 22.68 |
| Experimental road candidates | 14.20 | 32.37 | 37.75 | 41.40 |

Factory mean process-only throughput was about 16.95 FPS for that run. This does not establish 30 FPS, industrial deadlines, field recall or defect escapes. Road track events do not establish unique physical defects. Background load, CPU thread limits, scene and runtime configuration change timing.

## Real service integration check

With the local service running:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_local_api.py --camera
```

[Recorded integration report](../output/evaluation/integration-smoke.json).

All five integration checks passed:

1. Factory synthetic source processed through the actual HTTP-controlled local runtime.
2. Freight synthetic source processed through the same runtime.
3. Roads synthetic source processed through the same runtime.
4. The laptop camera processed three real frames through FreightProfile using MSMF, with event-image saving off.
5. JSON export succeeded and cross-origin device control was rejected.

The short synthetic streams each processed 12 frames at a configured 8 FPS. Observed host-receipt-to-decision p95 was 143.7 ms factory, 39.8 ms freight and 49.0 ms roads. No drops, late decisions or inspection exceptions occurred in those short samples. These are small smoke samples, not a sustained-load benchmark or a timing guarantee.

The service was left on an explicitly labelled synthetic factory source, not an active private camera. The evidence directory contained zero JPEGs after the default-off media checks. Event metadata remains in the local database.

## Browser verification

The local dashboard was opened in the Codex browser. Its displayed processing results, source labels, timing and storage state were inspected. It shows the distinction between synthetic inputs, live camera and local file replay. Profile selection and stream start are separate actions; camera activation is explicit.

The UI uses only local CSS/JavaScript and system fonts. Static checks verified JavaScript syntax and HTML control references. The dashboard's P95 label is HOST-TO-DECISION, with camera exposure/backend delay explicitly excluded. Factory buffering is labelled FIFO; freight and roads use latest-frame behavior. Browser verification also observed both fictional freight pages marked captured, subsequent duplicate suppression, and successful switching to the running road profile. A screenshot of the synthetic dashboard is saved with the build output.

## Outstanding validation

- No public production/road dataset has been downloaded or evaluated by this build.
- No trained road detector, OCR engine or generic document classifier is configured.
- No representative factory line, actuator, vehicle or GNSS hardware has been validated.
- No thermal soak, unattended multi-hour run or industrial timing guarantee exists.
- Reference calibration and accepted packet data do not yet persist across runtime restarts.
- Source loss is surfaced, but automatic camera reconnection and subprocess isolation of blocked native reads remain open.
- AWS management, update verification/rollback and cloud event synchronization are not implemented. Their UI state is disabled; they are not claimed as complete.

The initial implementation demonstrates local edge plumbing and narrow classical-vision tasks. Production accuracy, installation economics and meaningful AWS competition compliance require the remaining work in the [backlog](implementation-backlog.md).


## 20 October 2026 final local research checkpoint

The initial-build limitations above are historical. The installed models, acquired datasets and current measurements are summarized in [current model status](current-model-status.md). The factory accuracy target remains unmet; construction wall/tile categories are weak research, not complete building inspection.

Final verification passes447 runtime-environment tests, with2 skips and1 upstream AnyIO deprecation warning, in37.15s. The independent Torch-only arithmetic file runs separately in `.venv-roads-export`:5 pass in.041s test time, excluding interpreter/import cost. JavaScript syntax passes. All23 registered real comparison videos return HTTP200 and their reports exactly match saved evidence;96 local links across the principal result documents resolve. Actual browser verification shows the new crack-mask video, complete38/33/3 counts and sourceframe3 localized boxIoU.83. Matching source loading selects Local video file and the actual failed research model; the station remains stopped.

The first full collection exposed an optional Torch dependency. The next run had443passes/5failures/1skip: four failures needed the ONNX graph-test dependency and one was a historical preparation-only assertion that inference outputs did not yet exist. ONNX1.23.2/ml_dtypes0.6.0 are now installed and pinned; the original pre-NN test source and successful15-test receipt remain unchanged, with that specific assertion explicitly skipped once execution has occurred. Its skip does not waive an accuracy check. The other skip is an optional Torch construction fixture. [Exact counts, changes and source hashes](../output/evaluation/round3-final-verification/software-checks.json).

These checks verify software and evidence integration. They do not establish factory production recall, real-line item coverage, video pixel accuracy, quantitative humidity, installed-tile geometry or construction acceptance.
