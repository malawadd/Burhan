# Real-input integration verification

**Performance update:** measurements below record the earlier implementation. Current model/runtime changes, before/after accuracy checks, measured presets and regression results are in [the laptop performance study](performance-study.md). Historical results are preserved for comparison.

Date: 20 October 2026. This extends the [initial runtime verification](verification.md). All model execution is local CPU work on the same Windows laptop, Intel i7-8550U, Python 3.13.5 and OpenCV 5.0.0. No supported CUDA device is configured.

## Public-data evidence

- [Factory model/data and complete held-out results](models/factory-surface.md): 399 real industrial images; physical items separated into train/calibration/test. The full profile tested 80 untouched images: 8 true positives, 3 missed defects, 1 false positive, 68 true negatives. All 80 passed the acquisition guards; this is narrow controlled surface inspection, not general machinery diagnosis. The UI includes a known missed defect rather than only successful examples.
- [Freight models/data and results](freight-public-evaluation.md): six public inputs, including three genuine photographed receipts and three externally authored synthetic invoice fixtures. All 12 selected field checks passed on those three invoices; receipts remained unknown freight types. No real bill-of-lading/delivery-note/packing-list benchmark accuracy is established.
- [Road model selection, original annotations and measured results](roads-real-validation.md): the initial India model's failed Czech check is retained. Original and mirrored annotation sources are identified separately. See the report for the replacement's results and evaluation limits.

Model hashes, data source URLs, licenses, split identity, setup commands and actual outputs are in the linked Markdown reports and their JSON evidence. New documentation is Markdown; no PDF report was generated.

## Installation and camera integration

The runtime keeps regular `opencv-python==5.0.0.93`. Testing the headless wheel removed its working MSMF camera backend on this laptop; DirectShow also failed. RapidOCR 3.9.0 supports regular OpenCV, avoiding two conflicting packages that own the same `cv2` files. The selected dependency environment passes `pip check`.

After restoring the regular wheel, a fresh camera probe opened MSMF and read three 640×480 frames, saving no images. This probe took about **22.1 seconds including initialization under concurrent development load**; the earlier build's shorter initialization is not a startup guarantee. Camera exposure/backend buffering is not measured. No private camera footage was retained.

The separate `.venv-roads-export` environment is for one-time model export; PyTorch and Ultralytics are not runtime dependencies. The normal `.venv` uses compact installed artifacts and ONNX Runtime/OpenCV/scikit-learn CPU execution. Model/dataset acquisition requires network access during explicit setup, while inference uses local files.

## Source completion and deadlines

- Finite input transitions through `draining` until queued inspection completes. The elapsed-time endpoint freezes after draining, not merely after the final decoded frame.
- Accepted factory triggers expire during idle inference polling even if capture blocks. Source completion settles accepted triggers with no final inspection outcome. Tests verify explicit missed events without requiring the operator to press Stop.
- Freight uses one bounded background OCR job. Reset/source/session changes invalidate stale results. Stop/switch waits boundedly and refuses another station while a native OCR worker remains active.
- At EOF, freight can finish the retained actual last frame, without inventing a new camera capture or incrementing processed-frame counts. The runtime waits up to 45 seconds, can be stopped, and records an incomplete source if OCR remains unfinished.
- Such document completions are marked `async_source_completion`, preserve original sequence/capture timestamp and carry OCR job timing. Their per-frame latency/late fields are null and they do not enter guidance P95. This explicitly separates OCR readiness from frame-guidance deadlines.
- Folder/PDF hold time defaults to 30 seconds and is configurable. Time-based replay still does not guarantee every page was inspected if processing cannot keep up; use one-image input or evaluation scripts for deliberate per-image checks.

## Reproduce

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe scripts/smoke_real_inputs.py
```

The real-model runtime smoke denies Python outbound socket connections, uses downloaded public stills, saves no event images, and checks actual local results/previews. Repeated stills test runtime integration, not additional independent accuracy samples. It does not modify Windows network settings or constitute a physical network-disconnection field test.

## Combined real-model runtime smoke

`scripts/smoke_real_inputs.py` completed with Python outbound socket connections denied. It used one real industrial surface image, one public synthetic invoice fixture and one real annotated road photograph. All three used learned local model paths, not the authored synthetic demo handlers. Event-image saving stayed off.

| Profile | Repeated processed observations | Latest result | Host guidance/decision P95 |
|---|---:|---|---:|
| Factory surface model | 4 | Fault candidate | 500.2 ms |
| Freight asynchronous OCR | 27 | Invoice accepted for review | 178.1 ms guidance only |
| Roads five-class model | 4 | Review candidates | 698.9 ms |

These short repeated-still checks are not accuracy samples, sustained thermal tests or guaranteed deadlines. Raw configuration, exact model hashes, domain output, drops/late counts and timings are saved in `output/evaluation/real-runtime-smoke.json`. The separate module reports provide broader public-data results.

The combined smoke had zero inspection errors and zero late processed decisions across all profiles. Freight discarded one pending guidance frame under its documented latest-frame policy; factory and roads discarded none. Frame discards are not equivalent to lost documents or measured physical coverage.

The actual EOF test `scripts/smoke_document_eof.py` placed the public invoice in a one-image folder with only a one-second hold. The source captured and processed **exactly two frames**, then drained the real background OCR job. It completed the invoice with **one separate asynchronous completion**, retained source sequence 2, preserved the original capture timestamp and added no latency sample or fabricated frame. OCR job time was **10,092.26 ms**; guidance P95 was **117.3 ms**. The expected invoice reference was read from pixels. See `output/evaluation/document-eof-smoke.json`. Networking was blocked by a Python test patch and image saving was off.

The final integrated command `python -m pytest -q` passed **83 tests in 39.09 seconds**. One third-party Starlette/AnyIO deprecation warning remains. The suite covers bounded storage/failures, queue overload, real local adapters, model contracts/hashes, OCR stale-result invalidation, shutdown ownership, source completion and factory trigger outcomes. `pip check`, Python compile checks and JavaScript syntax checks also passed.

Browser verification used the actual local service: the factory public positive produced one surface candidate, invoice OCR populated `INV-2026-001` with a completed invoice checklist and duplicate suppression, and the five-class road model produced two pothole candidates on the annotated motorbike photo. The dashboard remains on the public road image, with camera capture and event-image saving off. [Screenshot](../output/real-input-dashboard.png).
