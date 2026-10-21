# Local prototype build notes

20 October 2026.

## Real-input implementation

Three parallel workers researched, implemented and measured factory surface inspection, freight OCR and road condition detection. The root integrated still-image/folder/PDF sources, model settings, public-data presets, model attribution and async lifecycle/source completion. [Real-input verification](real-input-verification.md) is the current combined record; initial synthetic measurements below remain historical baselines.

The final delivered modules use a locally trained Kolektor surface Random Forest, PP-OCRv4 English mobile models, and a locally exported five-class YOLO12s model through CPU ONNX Runtime. Data and model rights, rejected candidates and measured limits are documented separately for each module. No private footage was uploaded or retained during development. One station remains active at a time.

The real-data audit fixed premature EOF timing, idle/EOF factory trigger settlement, bounded OCR shutdown ownership, stale OCR rejection and final-page completion after source EOF. Asynchronous document completions preserve original frame identity and are excluded from guidance timing percentiles. The dashboard visibly labels real files and exposes separate OCR job time.

## Parallel work

The root agent implemented shared contracts, environment setup, local capture, buffering, runtime supervision, API, integration tests and documentation. Three subagents implemented factory, freight and roads independently against the shared contract. Follow-up work assigned the dashboard, evaluator and asynchronous storage review/fixes in parallel.

The implementation uses one active local station, selected by profile. It does not run three independent cameras simultaneously. The same host and codebase can exercise each supported profile through the same dashboard and runtime.

## Decisions made

- OpenCV 5.0.0.93 was available from PyPI and installed in an isolated Python 3.13 environment. cv2 reports 5.0.0.
- CPU inference was selected for this laptop; no CUDA assumptions were made from its GPU inventory.
- Media Foundation was selected after a DirectShow camera-open failure and successful MSMF frame capture.
- Synthetic, self-authored scenes provide an immediately reproducible local demo. Analysis inspects actual image pixels; scenario indices and benchmark labels are not runtime inputs to the detector.
- Factory reference differencing was implemented before introducing a large learned model. Camera mode requires explicit calibration.
- Freight color markers are an explicit supported fictional-template mechanism, not a generic document recognizer.
- Roads uses an experimental compact-dark-region heuristic. It is intentionally labelled a candidate baseline because shadows and camera conditions can cause false positives.
- The initial station defaults to 8 input FPS to leave CPU headroom. Users can configure the rate and deadline; measured lateness and drops remain visible.
- The browser never submits footage to an external inspection service. Preview requests are loopback requests to an in-memory JPEG.
- No cloud deployment or paid service was invoked. AWS work remains in the target plan.

## Issues found and corrected

- Storage initially held mutable payload references, could leave failed-write JPEGs, accepted post-close events and did not explicitly close SQLite handles. The storage rewrite adds bounded immutable snapshots, cleanup, lifecycle state and Windows regression tests.
- Triggered inspection initially treated pending temporal confirmation as an immediate missed item. It now preserves a unit ID within an absolute budget and records terminal results or missed/late outcomes.
- The dashboard initially labelled factory buffering as latest-frame and described timing too broadly. It now distinguishes FIFO and reports host-receipt-to-decision latency with camera delay excluded.
- Calibration errors are returned to the caller instead of reporting an invalid reference as accepted.
- Stopped session-average rates use a frozen elapsed time rather than decaying after the station stops.
- Non-default synthetic factory resolution now generates a matching demo reference instead of silently comparing incompatible shapes.
- Local controls reject mismatched browser origins and non-JSON mutation requests.

## Reproduction artifacts

- requirements.txt and requirements-lock.txt: direct and resolved dependency pins.
- tests/: detector, runtime/API and event-store regression tests.
- scripts/evaluate_profiles.py: synthetic functional checks and unpaced process timing.
- scripts/smoke_local_api.py: actual local-service and optional laptop-camera check.
- output/evaluation/: recorded machine-readable results.
- docs/modules/: method details, config and known failure cases per module.

See [verification](verification.md) for actual counts and measured limits. The [product plan](product-plan.md) and [backlog](implementation-backlog.md) describe intended extensions; their presence is not evidence that those extensions have been implemented.

## Browser observations

The actual dashboard displayed the factory stream with real pixel-difference overlays and populated timing counters. Switching to Freight and starting its local synthetic source eventually marked both delivery note and packing list captured; continued viewing of the same page reported duplicate rather than accepting it repeatedly. Switching to Roads started a new session through the same runtime and displayed its experimental scope. The source badge remained synthetic throughout these checks. The camera integration check ran separately and was stopped; no camera screenshot was saved.
