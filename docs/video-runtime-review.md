# Shared video/runtime correctness review

Reviewed 20 October 2026 with lightweight timing/metadata fixtures; no model inference benchmark was run during the other agents' exclusive measurement windows.

## Corrections

- Capture scheduling now starts after opening the camera/codec. A slow open no longer leaves an initial scheduler backlog and immediate catch-up read.
- Original video frame indices travel in each immutable `FrameContext` and result payload. Later capture metadata cannot relabel an earlier result as a different source frame.
- Invalid, negative or repeated container timestamps fall back to nominal original-frame time when source FPS exists. Results preserve the timestamp semantics, so fallback values are not presented as measured exposure times.
- Source skips increment coverage gaps for factory and freight as well as road/construction profiles. Source skips, inference-queue drops and inspection errors remain distinct counters.
- Results persist actual construction preprocessing, `ppe_preprocess`, supported capabilities, model source/license and validation context. The configuration hash describes submitted settings; effective defaults are recorded separately in model metadata.
- Unknown construction stages are rejected before loading a model or stopping a valid session. Static catalog entries cannot restore stage/row approvals: rows are `not_assessed`, their default remains `unassessed`, and code compliance is `not_determined`.
- PPE and crack catalog rows identify the narrow available visual candidates without marking the broader inspection family complete.
- Dashboard stage choices survive catalog/state/preset response ordering. Public presets apply their video playback mode and preview rate.

## Playback and coverage limits

`realtime` uses the video's nominal reported FPS for its replay clock and skips old source frames explicitly. This is not a complete variable-frame-rate presentation-timestamp scheduler. A missing FPS uses the requested input rate and exposes that fallback in metadata. Camera exposure latency remains unknown.

The API's existing `every_frame` value means **sequential source decoding without source skipping**. It does not guarantee every decoded frame reaches inference: guidance profiles retain a bounded latest-frame queue, and stale/overflow processing remains visible. The UI now names the control “Sequential source” and explicitly explains this distinction. A future all-frame offline inspection mode would require backpressure and separate deadline semantics; no such guarantee is implied here.

Video dimensions and coordinate scaling remain explicit: findings refer to the analysis input frame; original source and bounded analysis dimensions are included separately. A 4K original may be resized to a 1920-pixel analysis bound before the selected model's own preprocessing.

## Verification

```powershell
.venv\Scripts\python.exe -X utf8 -m pytest tests\test_video_playback.py -q
node --check edge_runtime\static\app.js
```

The first review's **12 tests passed in 1.49 seconds**; JavaScript syntax passed. Regressions cover real-time source skipping versus sequential decoding, coordinate metadata, unusable timestamp fallback, delayed source opening, factory coverage gaps, stable per-packet frame identity, preprocessing provenance, invalid-stage rejection and prevention of static approval reuse. Controlled fixture frames verify code contracts only; they are not represented as real model/video evidence.

## Final recorded-video and lag review

The recorded-video player uses same-origin allowlisted H.264 MP4 artifacts and remains separate from the active inspection stream. The page explicitly identifies these as actual model outputs, including errors. Playing a recording does not start a model, change an inspection stage, or approve any checklist item. Notes and attribution are inserted as text, not executable HTML.

Two additional correctness fixes were made:

- A repeated/invalid video timestamp after a nonzero container start now preserves that origin. For example, 1000 ms followed by a fallback at source frame 3 / 10 FPS yields **1300 ms**, not 300 ms. The replay-lag calculation therefore compares consistent origins. Lag measures host replay timing against source-relative presentation time, not camera exposure latency or model decision latency.
- Recording inventory and delivery now use the same resolved-path predicate: the artifact must be an MP4 file under the resolved workspace `output` directory. An existing file outside that directory, traversal path or wrong file type is unavailable in inventory and rejected by delivery. An arbitrary filename cannot bypass the registry's identifier allowlist.

The existing CSP's `default-src 'self'` already permitted same-origin media. An explicit `media-src 'self'` now records the intended policy. Starlette `FileResponse` retains browser byte-range support; the allowlisted route also explicitly supports HEAD without exposing it as a separate documented operation.

```powershell
.venv\Scripts\python.exe -X utf8 -m pytest tests\test_video_playback.py tests\test_recorded_video.py -q
```

**15 focused tests passed in 2.20 seconds**, with one dependency deprecation warning. Byte fixtures verify normal MP4 delivery, 206 partial and suffix ranges, 416 out-of-range handling, HEAD length/body behavior, CSP and path/identifier restrictions. A timing fixture verifies nonzero source origins, actual source skips and a 50 ms replay lag. These tests do not decode a purported real video, run inference or establish H.264 playback compatibility; the actual exported-video browser check remains separate.

## Actual browser and full-suite verification

The updated loopback server was restarted with current code. The in-app browser successfully played the actual H.264 NIOSH annotated excerpt to its endpoint: `readyState=4`, 1280×720, duration/current time 10.01001 seconds, ended/paused true, no video error. This is real codec/playback evidence, separate from the byte-fixture HTTP tests. A labelled Play/Pause recording button provides an accessible control; native video controls remain available. Viewing a recording did not start an inspection station or camera.

The browser also opened the preconstruction checklist and visibly showed separate low-rise residential/commercial applicability notes and official links; every item remained NOT ASSESSED. A JSON audit verified all89 topics/16 stages include both nonempty building-scope notes and retain research_only/unassessed state.

Full Python suite: **140 passed, one dependency deprecation warning,67.31 seconds** while accuracy-only CV/encoding could share CPU. This is regression evidence, not a throughput measurement. JavaScript syntax and runtime `pip check` passed. Actual speed benchmarks used the separate exclusive timing windows described in the update.

The actual dashboard Start/Stop workflow was also exercised on the original NIOSH video. Its40.2-second interactive session captured603 observations and processed595, with8 queue drops,360 skipped source frames, zero late decisions/errors,14.8 processed FPS and93.0ms host-decision P95. This includes GUI load and cold initial inference effects, so it does not replace the separate exclusive225/225 measurement or imply zero-loss UI operation. It also produced a spurious thin image-edge machinery candidate on a title card; outputs remain unassessed and experimental. Raw state: `output/evaluation/construction-ui-session.json`.

The browser's final error log was empty. The inspection station is stopped and no camera is active; the loopback server remains available. Screenshot `output/construction-recorded-result.jpg` shows a paused real annotated recording, attribution, known-miss note and the89-topic unassessed plan. A separate stopped-stream screenshot is retained at `output/construction-video-dashboard.png`.
