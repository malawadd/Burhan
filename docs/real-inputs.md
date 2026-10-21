# Real local inputs

The laptop remains the edge device. Cameras, files, model inference, OCR, previews and event storage stay local. Downloading public datasets/model packages during setup needs the internet; processing installed artifacts does not. No application startup or inference path downloads models automatically.

## Use the dashboard

Run `./.venv/Scripts/python.exe -m edge_runtime.cli serve`, then open <http://127.0.0.1:8765/>. The **Public-data example** selector fills a real downloaded sample and its explicit model settings. Press **Start inspection** to apply it. Samples are public data, not a live camera or synthetic demo.

For your own input, select a module and input type, enter its absolute local path if needed, set the model options, and press Start. **Local model settings** accepts a JSON object passed to that module's constructor. Changing a field does not change a running station until Start is pressed. The development runtime operates one station at a time.

| Input | API `source` | Behavior |
|---|---|---|
| Laptop camera | `camera:0` | Local Media Foundation capture on Windows; no image saving by default. |
| Video | `file:C:\data\line.mp4` | Source-timeline replay with explicit skipped-frame counts; optional sequential review; explicit end-of-file. |
| Single image | `image:C:\data\product.png` | Repeats the same image until stopped, allowing stability/confirmation controls to operate. |
| Image folder | `images:C:\data\pages` | Sorted supported files in the immediate folder; each is held for 30 seconds by default, then advances; stops at the end. Maximum 10,000 files. |
| PDF | `pdf:C:\data\packet.pdf` | Renders each page locally, holds it for 30 seconds by default, then advances; stops at the end. |

Repeated still frames are **not independent observations or dataset samples**. Frame counts and temporal confirmation in these modes test the interactive workflow; use module evaluation scripts for image-level measurements. Set `source_hold_seconds` (1–300 seconds) through the API or Local model settings to allow enough time for stability and OCR. Folder/PDF replay is time-based and can skip an item if processing remains slower than its hold time; use single-image input or the evaluation scripts for deliberate per-image checks. Folder replay does not recurse into subfolders. Still images are bounded to 1,920 pixels on the long side before inspection; PDF rendering is capped at 1,800 pixels before any profile-specific resizing. Full-image document inputs and photographed paper may use different page-quality paths.

PDF rendering uses [pypdfium2](https://pypdfium2.readthedocs.io/en/stable/python_api.html) 5.14.0. Document, page and bitmap handles are explicitly closed. One capture thread owns the PDF; it is not rendered concurrently. Its [package licensing](https://pypi.org/project/pypdfium2/) lists Apache-2.0/BSD-3-Clause for the wrapper and bundled PDFium dependency notices. Keep the wheel's notices when redistributing it. This is input rendering, not a PDF report generator; project documentation remains Markdown.

## Local API

`POST /api/start` accepts `profile_options` in addition to source/FPS/deadline controls. For example:

```json
{
  "profile": "factory",
  "source": "camera:0",
  "input_fps": 4,
  "deadline_ms": 2000,
  "save_evidence": false,
  "profile_options": {}
}
```

Use an approved normal reference for general fixed-view factory comparison, or explicitly choose the trained surface-defect model for compatible surfaces. A trained model's dataset is a narrow domain, not evidence it works on every product or camera view. Read each module guide for supported settings and licenses:

- [Factory](modules/factory.md)
- [Freight](modules/freight.md)
- [Roads](modules/roads.md)
- [Construction models and real footage](construction-vision.md)
- [Video timing, geometry and coverage](video-inspection.md)
- [Saudi inspection study and scope](saudi-construction-inspection-study.md)

Results preserve input provenance, runtime/profile versions, configuration hash and available model hashes. Local document text may appear in event JSON, exports and the dashboard even with image saving off. Keep sensitive documents on a controlled laptop and use synthetic/public inputs for shared screenshots.

## Timing and coverage

Capture, inference, JPEG encoding and storage run in separate bounded workers. A heavier model can lower the processing rate and cause skipped/late frames. The dashboard reports these; a configurable input FPS is not a throughput guarantee. Factory remains FIFO, while freight, roads and construction use the latest pending frame. Preview encoding has an independent 0–15 FPS limit (default six); zero disables only preview. See the [before/after performance study](performance-study.md) for measured settings.

Host-receipt-to-decision timing starts **after** decoding/camera read/PDF rendering returns. It excludes camera exposure, backend buffers, source decoding/rendering, preview JPEG/browser rendering and physical actuation. Warm initialization and steady-state processing must be reported separately. Factory and freight use CPU; roads and construction also offer a separately measured OpenVINO path on this laptop's Intel UHD 620 GPU. It does not require CUDA. CPU fallback settings remain documented explicitly. Real dataset inference and narrow holdout evaluation do not establish production accuracy, safety or line coverage.

The camera is activated only when selected and started. Save event images is off by default. The service binds to loopback and does not upload footage.
