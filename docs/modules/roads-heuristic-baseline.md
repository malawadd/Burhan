> Synthetic/legacy heuristic baseline only. Real inputs now default to the [local trained model](roads.md).

# Experimental road candidate review

This module is a functioning, offline CPU baseline for **candidate-region review**. It is not a trained pothole detector and has no measured field accuracy. It works best with a fixed camera looking at a dry, textured road and a manually calibrated visible-road ROI. Both annotated video and result titles label it experimental.

`RoadsProfile(config=None).process(bgr_frame, FrameContext(...))` returns the shared `InspectionResult`. `reset()` clears tracking. `describe()` exposes scope and configuration. `demo_frame(index, width=960, height=540)` creates synthetic scenes with textured asphalt, an irregular moving dark region and an elongated shadow. Detection reads actual pixels; it receives no demo labels or scene index. These generated images contain no external imagery or dataset assets.

## What runs

OpenCV converts to grayscale and checks ROI brightness and Laplacian texture/focus. Underexposed, overexposed or textureless frames return `rejected` and `request_retake`; these frames never imply a clear road. It thresholds compact dark regions relative to the road median, removes noise, then filters by ROI-relative area, aspect ratio, extent, solidity and contact with the ROI boundary. Dark contrast divided by 255 is a method-specific score, **not probability or confidence of damage**.

Three consecutive overlapping detections confirm a candidate by default. It then requests human review and emits one event for that short-lived track. One-to-one greedy box matching uses intersection-over-union; tracks expire after one second without detection. Overlap is mandatory. There is no geographical deduplication, camera-motion compensation or guarantee that two separate tracks represent different real defects. A repeated region after timeout creates a new event. Sequence/time rollback, resolution change or source/session change resets tracking. Rejected frames break unconfirmed streaks. No road coordinate, GPS location, depth, engineering severity or repair urgency is calculated.

`domain_state.tracks` contains visible IDs, pixel boxes, hit count, contrast score and confirmation. `new_track_ids` lists newly confirmed tracks; a frame with multiple new tracks emits one event containing their IDs. `event_key` combines session, source and IDs. IDs increase during the instance lifetime, including resets. Root runtime owns session recording, retention, coverage and delivery; this module does not send messages or dispatch workers.

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `roi` | `[[.39,.36],[.61,.36],[.94,.96],[.06,.96]]` | Four normalized polygon points; recalibrate for each real camera |
| `min_contrast` | 32 | Required grayscale darkness below median |
| `min_area_fraction` / `max_area_fraction` | .0015 / .065 | Contour area relative to ROI |
| `min_focus` | 16 | Minimum Laplacian variance; texture affects this score |
| `confirm_frames` | 3 | Consecutive frames before review |
| `track_timeout_seconds` | 1.0 | Maximum gap retaining local identity |
| `match_iou` | .12 | Minimum bounding-box overlap for identity |

Thresholds are demo starting points, not field calibration. This heuristic itself has no ONNX path. Real inputs use the separately documented trained model. No network, paid API or model download occurs at runtime. AWS integration can store evidence and review records through the parent application; the detection itself is local OpenCV.

## Limitations and evaluation

The shape filter rejects the demonstrated broad elongated shadow, but compact shadows **still trigger candidates**. Tests explicitly preserve that known failure so the UI does not silently imply semantic shadow discrimination. Dark paint, asphalt repairs, wet surfaces, leaves and vehicles can also trigger. Light-colored defects and thin cracks may be missed. A uniform healthy road can fail the texture check. Automatic ROI inference, robust motion compensation and production distress classification are absent.

The synthetic suite verifies pixel-driven operation, empty ROI, broad shadow filtering, known compact-shadow false positives, three-frame confirmation, event deduplication, bounded timeout, source reset and rejected-frame behavior. These are behavioral checks, not accuracy claims. Before any real deployment, acquire licensed representative footage, calibrate ROI, evaluate by camera/weather with human ground truth and compare review time against current tools. Hold out entire sources/routes to prevent adjacent-frame leakage. Use defect recall, false review events per minute, rejected-frame rate and review time; never report a detection benchmark on these generated images as field performance.

From the repository root run `.venv/Scripts/python.exe -m unittest discover -s tests -p test_roads.py -v`.
