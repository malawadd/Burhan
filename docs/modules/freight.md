# Freight module: real local document input

Implemented 2026-10-20. This module now reads actual printed pixels with offline OCR. Its default workflow expects a delivery note and a packing list; the required document types are configurable. It also recognizes English invoice and bill of lading headings.

## Run it

The parent project setup installs `rapidocr==3.9.0` and CPU `onnxruntime==1.23.2`. Keep regular `opencv-python==5.0.0.93`: the headless Windows wheel lost the laptop's working Media Foundation camera backend during verification. RapidOCR 3.9 uses the regular package; 3.10 declares headless. Do not install both OpenCV wheels together.

One-time explicit model and public sample setup:

```powershell
.venv/Scripts/python.exe scripts/fetch_freight_models.py
.venv/Scripts/python.exe scripts/fetch_freight_assets.py
```

Both scripts need internet while installing assets. Runtime processing never calls either script or downloads missing files. Missing files or SHA256 mismatch produce a visible OCR error rather than a demo result.

Start the local dashboard. Select Freight and use `camera:0`, a local video, `image:<absolute-path>`, `images:<absolute-folder>`, or `pdf:<absolute-path>`. A digital page uses its image boundaries; a camera page needs visible corners and a contrasting background. A camera photo with tiny printed text cannot be recovered by upscaling. Use a closer, focused camera view or an original scan.

A ready public image is:

```text
image:C:\Users\IAM\Documents\hackathon\opencv\project1\datasets\freight\invoice-pack\01-simple-english\invoice.png
```

For its one-document packet, use profile options:

```json
{"required_pages":["invoice"],"stable_frames":2}
```

Default packet types remain `["delivery_note", "packing_list"]`. Other supported required keys are `invoice` and `bill_of_lading`. Reset the packet between shipments. An invoice does not satisfy a missing packing list.

## Actual behavior

1. OpenCV finds the camera page quadrilateral, rectifies it, and checks its size, focus, printed detail and motion. Scans and PDF page renders are treated as complete digital pages.
2. Once stable, one bounded background job runs PP-OCR locally through ONNX Runtime CPU. Page quality and the live preview continue while OCR runs. The UI says **Reading page locally** and exposes `ocr_pending`.
3. The module consumes a result only when its source, session, reset generation and perceptual page hash match the current page. Changing the page or resetting cannot attach an old result to a new packet.
4. English heading rules recognize `PACKING LIST/SLIP`, `DELIVERY NOTE/ORDER`, `PROOF OF DELIVERY`, `BILL OF LADING`, and `INVOICE`. Ambiguous combinations and unknown headings require review. Mean OCR score must reach 0.65 before a type is accepted; this is a recognition score, not a calibrated document confidence.
5. Label-based regular expressions suggest reference, date, currency, total, gross/net weight, and package count when explicit labels and values are present. Every extracted value has verbatim evidence, method and `verified:false`. OCR line boxes use rectified document pixels (`coordinate_space`); the main page finding uses original input coordinates. It does not invent absent values or infer totals from a table.
6. Accepted document types are captured once per packet. The perceptual hash suppresses repeated views. A different second page of an already captured type is flagged for review rather than replacing the first.

The former colored markers are recognized **only for the explicitly labelled `demo` source**. A real camera image containing a blue or green square cannot use that shortcut.

## Model and lifecycle

Profile version: `ppocr_v4_english_mobile_document_rules_v2_cpu`.

- Detector: RapidAI export of PaddleOCR `ch_PP-OCRv4_det_mobile.onnx`, 4,745,517 bytes.
- Recognizer: `en_PP-OCRv4_rec_mobile.onnx`, 7,653,044 bytes.
- Orientation: bundled `ch_ppocr_mobile_v2.0_cls_mobile.onnx`.
- Backend: ONNX Runtime CPU, 4 intra-op threads and 1 inter-op thread, CPU arena on, helper spinning off, one text crop per recognition batch; CUDA and DirectML off. See [before/after CPU measurements](../freight-performance.md).
- Runtime explicitly selects existing local paths. All three model hashes are verified against the official registry before use. The installed OCR wheel contains unused PP-OCRv6 small models; these are not the selected runtime models.
- No OCR requests, images or extracted text leave the laptop. OCR text and suggested fields are included in local event records/export, so selected event metadata contains document content even when image saving is off.
- Exactly one OCR job can run per profile, without a queued collection of camera frames. Results are reused while that same page remains in view.
- `finish_source(timeout=3)` waits for an in-flight final-page job and returns a result against the retained original frame/context, or `None` while pending. `source_completion_state()` exposes EOF/pending state. It does not create new captures or repeatedly start OCR at EOF.
- `close(timeout=3)` invalidates results and waits up to the requested bound. It returns `False` if the native OCR operation is still running. The shared runtime must wait/refuse a station switch until it exits, preventing accumulating abandoned workers. Reset alone invalidates the observation but does not interrupt a native inference call.
- Live per-frame deadline metrics describe preview/quality/decision frames; OCR engine duration is reported in `ocr_ms`, and full job duration including initialization in `ocr_job_ms`. They do not mean that a complete page is OCRed within the frame deadline.

## Public datasets, provenance and licensing

See [public asset/model study and measurements](../freight-public-evaluation.md). Downloads are stored under ignored `datasets/freight` and `models/freight`; manifests include source URLs, sizes and SHA256 values. License copies are included beside downloaded assets/models.

The downloaded evaluation set is **three genuine photographed CORD receipts** and **three externally authored synthetic invoice fixtures**. The receipts test real paper-image OCR and conservative out-of-scope handling. The invoice fixtures test invoice headings/fields without exposing private shipments. They are not a representative real freight document benchmark. No real bill of lading, delivery note or packing-list holdout has been validated yet.

## Limits

Page type presence is not shipment correctness, authenticity, signature validity, goods matching, legal clearance, or a verified document packet. No generic learned freight document classifier, table parser or trained key-information extractor is claimed. English headings and labelled fields are the supported extraction scope; other languages/scripts need dedicated model and rule selection. Difficult shadows, glare, landscape pages, handwriting, arbitrary layouts, missing titles, rotated 90-degree pages, and poor images need review. The orientation classifier handles 0/180 degrees, not arbitrary document rotation.

The current HP laptop still needs seconds per complete page. The selected CPU configuration reduced the identical six-page test from43.319 s to19.289 s while retaining all12 exact field checks and55/58 annotated word hits. See [measurement protocol, rejected variants and limits](../freight-performance.md). The UI continues separately while the bounded OCR job runs; this is an interactive document capture workflow, not full OCR at video frame rate. Synthetic demos remain available for regression testing.

## Verification

```powershell
.venv/Scripts/python.exe -m pytest tests/test_freight.py tests/test_freight_real.py -q
.venv/Scripts/python.exe scripts/evaluate_freight_public.py
.venv/Scripts/python.exe scripts/smoke_freight_async.py
```

Tests cover camera quality gates, source-specific demo markers, real mode heading extraction, bounded async jobs, stale session/reset invalidation, close behavior, low-confidence and missing-model handling, duplicate suppression, and actual public-image OCR with Python socket connections blocked. This tests no-network inference behavior without modifying the laptop's network settings.
