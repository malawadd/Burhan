# Real video inspection on the laptop

The application accepts local videos and cameras; inference and event storage stay local. Construction has no synthetic/demo input. Public video examples use actual recorded frames with source attribution and hashes. An annotated output video is an inference artifact, not newly generated evidence of construction work.

## Playback and coverage

`video_mode=realtime` is the default for video files. The capture worker follows elapsed source time at no more than the requested analysis rate or the video's stated frame rate. If a 30 FPS file is inspected at 15 FPS, the reader grabs past intervening source frames to keep playback on its original timeline. Those frames are counted as `metrics.source_skipped`; they are not represented as inspected. Captured observations subsequently rejected by the bounded inference queue remain a separate `dropped` counter. Roads/construction coverage-gap counts include both kinds of uninspected frames.

`video_mode=every_frame` reads every sequential source frame and paces playback at the requested input FPS. A 30 FPS video read at 15 FPS consequently takes about twice its original duration. This mode supports deliberate review; its processing rate must not be called source-speed video performance.

The API exposes `video.source_fps`, `frame_count`, `duration_ms`, `position_ms`, `source_frame_index`, playback mode and skipped counts. The dashboard displays the source timeline and skip count. Container timestamps are not camera exposure timestamps. Unsupported/unknown source frame rates remain explicitly unavailable; the configured analysis clock is then used.

`replay_lag_ms` compares source timeline progress with the wall clock. It exposes slow decoding/capture even when model-only latency is low; a video backend too slow to keep up must not be described as real time solely because model FPS is high. This is a nominal-FPS replay clock, not a full variable-frame-rate presentation scheduler. Invalid or repeated codec timestamps use an explicitly labelled nominal-frame fallback.

Decoded video is bounded to a 1,920-pixel long side while preserving aspect ratio. Each result records source and analysis dimensions from its own captured frame; bounding boxes refer to that analysis frame. Video metadata exposes the scale back to source pixels. This bounds annotation/preview memory on 4K inputs without pretending the detector sees full-resolution detail. Model-specific resizing occurs afterward. These pixels are not calibrated physical dimensions.

Construction uses the latest pending observation and reports replaced frames. A missing finding, skipped view or unsupported checklist topic never becomes an inspection pass. Original source files remain on disk; event image saving is off by default. Preview sampling at six FPS is independent of model inspection rate.

## Construction models and stage plan

Select the Construction profile, a real video/camera/image, and a named model. PPE and surface-crack models run separately so their actual computation and scope are explicit. A PPE observation does not certify suitable PPE; a crack mask does not measure physical crack width, depth, cause or structural acceptance.

The [Saudi study](saudi-construction-inspection-study.md) and [89-topic checklist](saudi-construction-checklist.md) cover 16 researched stages. The dashboard's stage selector filters that plan and records the selected stage in observation metadata. All topics remain unassessed. Available visual models are listed separately from the evidence needed for tests, calibrated measurements, design checks and authority decisions. The catalog is a research planning aid, not an exhaustive code-compliance engine.

Model setup, actual public image/video evaluations, licenses and annotation limits are in [construction vision](construction-vision.md). Item-level factory accuracy and sustained factory performance are measured separately; a construction video cannot validate the commutator-specific factory model.

## Verification

`tests/test_video_playback.py` controls a fake transport clock to check exact frame skipping and preservation of every-frame review. These are adapter tests, not simulated model evidence. It also checks that catalog data cannot issue/reuse an approval or conceal a row with an unknown stage. Initial focused result: four tests passed. Final Python suite:140 passed; see [runtime/UI verification](video-runtime-review.md) for actual browser playback and separate fixture scopes.

`scripts/benchmark_construction_runtime.py` uses an actual local video and the real shared runtime, with source-clock pacing, preview and local event storage. It can limit an excerpt by wall time without substituting or repeating source pixels. Reports preserve the full source hash, original media properties, selected time limit, initialization time, capture/processing counts, skipped source frames, drops, late decisions, model/device metadata and errors. Python socket connections are denied during installed-model processing. It measures throughput; event accuracy needs separately annotated original video frames/events.

Example (after explicit model and public-video setup):

```powershell
.venv/Scripts/python.exe scripts/benchmark_construction_runtime.py --video datasets/construction/framer-fall-protection-public-domain.webm --mode ppe --seconds 30 --fps 15 --deadline 200 --output output/construction-runtime-ppe.json
```

Image dataset compatibility diagnostics, full-video offline export rates, source-paced runtime throughput and Saudi project acceptance are different measurements. Each must retain its own stated scope.

## Recorded results in the dashboard

Construction has a separate **Recorded public-video results** viewer. Select an available recording, then use **Play recording** or the native controls. Actual NIOSH, site-PPE, bridge full-frame and bridge-ROI inference artifacts are available with source/license attribution and prominent known-miss/false-candidate notes. They are precomputed real-data outputs; playing one does not run inference or populate a code approval. H.264 viewing copies preserve the decoded count, nominal FPS, duration and dimensions of their original annotated exports and are separately hashed. Original public source videos remain unmodified.
