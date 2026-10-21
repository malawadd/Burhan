# InspectFlow Edge

Latest results: [rounds4–5 training, real-video and benchmark report](docs/inspection-improvement-round4.md). More real training data and thin-structure training still yield10/11 factory localizations (90.9%) with1/69 unaffected-view alarms, despite19.645FPS fresh capacity; sustained replay still drops frames. Thin cracks/spalling remain unreliable. All [four owner before/after videos](docs/new-construction-video-review-round4.md) and [five public construction comparisons](docs/construction-round4-measured-results.md) retain actual masks, misses and false foreground. No new model is promoted.

A local OpenCV 5 runtime for factory visual inspection, freight document OCR, road-damage review and construction video observations. Your laptop is the edge device. Detection, OCR, preview, decisions and event storage run locally; no footage is sent to a remote inspection service. The installed real-input models and public datasets are documented in [the real-input guide](docs/real-inputs.md).

Read [current model status](docs/current-model-status.md) for measured factory recall/false alarms, construction limits, rejected candidates and experiments still in progress. The detailed Markdown benchmark records preserve the failures alongside improvements.

## Run on this laptop

The environment is already installed in .venv. From the project directory:

```powershell
.\.venv\Scripts\python.exe -m edge_runtime.cli serve --port 8765
```

Open [the local dashboard](http://127.0.0.1:8765). Alternatively run .\scripts\start.ps1. Use Ctrl+C in the server terminal to shut down the service; the dashboard Stop button stops the inspection station.

The server binds to 127.0.0.1. It does not automatically activate the camera. Choose a profile and source, then press Start inspection. Four profile adapters are installed; this laptop prototype runs one active station at a time, with explicit profile switching.

### Try the profiles

| Profile | Synthetic demo | Laptop camera |
|---|---|---|
| Factory | Authored reference comparison with missing components and scratches. | Approve a focused reference for a fixed-view product, or explicitly select the locally trained Kolektor surface-defect model for compatible industrial surface crops. |
| Freight | Fictional color-marker pages exercise capture and duplicate suppression. | Offline OCR reads page text; conservative English heading rules identify delivery notes, packing lists, invoices and bills of lading. Extracted fields require review. |
| Roads | Moving dark regions exercise the older heuristic and tracking. | Real inputs use a local YOLO12s ONNX model for cracks, pothole candidates and separately labelled repairs. Domain shift, missed damage and false detections remain material limits. |
| Construction | Disabled: actual local footage only. | Local PPE/crack models, fourteen visible-condition photo diagnostics, recorded damage evidence and a Saudi lifecycle inspection plan. Per-category weak localization and unsupported checks remain explicit. |

Start with a **Public-data example** to inspect an actual local input using its recorded model/settings. The older authored demos remain functional regression tools. Camera resolution and a compatible product view must be configured for the selected model; the learned factory surface model is measured on portrait commutator-surface crops.

For real public-data examples, use the dashboard's **Public-data example** selector and press Start. It fills the source, model options and measured laptop settings. Cameras, videos, images, image folders and PDFs are supported. Read [real input usage and timing](docs/real-inputs.md) and the [before/after performance study](docs/performance-study.md) before applying a model to your own data.

The preserved factory spatial model's recorded original **16 input FPS /100ms** 60-second replay processed960/960 observations with zero drops/late/errors, P9538.6ms. Its later curvature-paired CAL replay processed959/960 with1 drop/8 late; a subsequent separate density-paired baseline processed960/960 with0 drops/late. All runs remain recorded: average FPS does not establish a hard deadline. The phase-max candidate reduces normal-view alarms5/69→1/69 but still localizes10/11 defects; its16-input-FPS replay processes960/960 with0 drops/late. New curvature and density candidates do not exceed90.9% recall. These are software-clock replays of80 predecoded public stills, not physical production-line coverage. Read the [current paired results and failed candidates](docs/inspection-improvement-round3.md); older experiments remain preserved. The explicit Intel GPU road preset passed **12 FPS /150ms** for30seconds (360/360, zero drops/late/errors, P9579.5ms); CPU presets remain available. Freight new-page OCR still takes seconds.

Construction now includes [real before/after camera videos with complete misses and unmatched detections](docs/real-before-after-video-results.md), a separate tile photo research specialist, and a [finish-expansion study](docs/construction-finishes-expansion.md). Thin cracks and spalling remain unsolved. The author segmentation candidate improves one full daytime crack clip from 1/71 to 38/71 matches, but still fails thin-crack/night/spalling checks; all five new mask recordings and 23 comparison videos are available. Wall damp/stain appearance, detachment, corrosion and bulging have separately measured weak photo models. Installed-tile geometry, paint alignment, lippage and quantitative humidity remain unsupported. The 24 earlier diagnostic video presets now correctly select the video reader.

**Factory accuracy and false alarms:** the spatial model introduced in round two localizes 10/11 exposed comparison defects (90.9%) at foreground IoU ≥0.10, but flags 5/69 normal views. It is a manual-review candidate. The [current accuracy, false-alarm and construction-damage study](docs/inspection-improvement-round3.md) preserves this before measurement and records the next experiments. The older 72.7% RF and failed review-policy experiments remain documented as historical baselines.

Factory manual-trigger mode accepts local trigger requests and associates a result or explicit missed/late outcome with each accepted unit. It is a software integration test, not verified PLC triggering or reject actuation. Untriggered observation does not establish every-item coverage.

Save event images defaults OFF. Preview remains in memory. Enabling the checkbox saves selected annotated event images locally; continuous video is never recorded by this application. The current factory reference and freight accepted-page state are session-local and are lost on restart. Recalibrate/rebuild the packet after restart.

## Fresh setup

Tested on Windows with Python 3.13.5. Python 3.11 or newer is declared; other OS/hardware combinations have not been validated.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m edge_runtime.cli probe-camera
.\.venv\Scripts\python.exe -m edge_runtime.cli serve
```

requirements.txt pins direct runtime dependencies. requirements-lock.txt records the runtime/test environment. Model and dataset setup scripts are explicit separate steps described in the module guides; copying the code alone does not include ignored local model/data folders. Inference and the interface run locally without a cloud account, Node build, CDN or external font. CPU/GPU requirements are documented per model; the new spatial factory variant explicitly requires this laptop's Intel GPU.

The factory native accelerator is already built on this laptop. On a fresh machine, explicitly run `./.venv/Scripts/python.exe scripts/build_factory_native.py` with a local Clang compiler. Startup validates its source/artifact hashes and falls back visibly to the slower threshold cascade when unavailable. See [factory performance](docs/factory-performance.md) for build and score semantics.

This laptop's HP TrueVision camera was successfully opened with Media Foundation. The DirectShow backend failed, so Windows capture explicitly uses MSMF. A camera probe reads frames without saving or returning image contents. OS camera permission settings may differ on another machine.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\evaluate_profiles.py --output output\evaluation\profile-baselines.json
```

See [recorded verification](docs/verification.md) and [profile evaluation](docs/evaluation-baselines.md). Tests include real pixel analysis of authored inputs, temporal/event behavior, local runtime operation with socket connections denied, trigger deadlines, local API controls, and storage failure/retention behavior. Synthetic functional checks are not field accuracy measurements.

Recorded process-only profile timing and integrated host-receipt-to-decision timing measure different paths. The dashboard excludes camera exposure, hidden backend buffering, physical actuation and browser rendering from latency. It reports drops, lateness and missed units separately; do not infer a production deadline from average FPS.

## What is implemented

- Shared local capture/replay, bounded FIFO/latest-frame buffering and local OpenCV inference.
- Four profile adapters with real local model inference/OCR, quality handling, overlays and domain results. Construction rejects simulation; older synthetic fixtures remain regression tools for the original profiles.
- A local live dashboard, manual controls, event history and JSON export.
- Bounded asynchronous SQLite/event-image storage with immutable event snapshots and explicit loss/flush state.
- A separate bounded JPEG worker, with independently sampled previews and original-resolution selected event images.
- Host timing percentiles, frame age, dropped/late frames, source errors, missed triggers and road coverage-gap counters.
- No network requirement for detection or the local interface; no raw media synchronization.

## What remains

- Representative customer data evaluation beyond the documented public-data checks. Dataset/model license restrictions must be respected; the industrial dataset is noncommercial, and the road model has upstream YOLO license obligations.
- Real line triggers, lighting/optics qualification, multi-camera operation and actuation integration.
- Persistent approved calibration, packet state and deployment configuration.
- Native phone deployment, real GNSS timing, long-duration/thermal profiling and camera reconnection supervision.
- AWS fleet management/package delivery and updates. Cloud is visibly disabled, not simulated as deployed; the hackathon's meaningful AWS requirement is still outstanding.

Do not expose this development server to the LAN or internet as a production management endpoint. It has loopback host/origin checks, but no multi-user authentication or fleet credentials.

## Documentation

- [Round three: factory false alarms, thin cracks, spalling and added damage categories](docs/inspection-improvement-round3.md)
- [Factory round 2: spatial segmentation, additional industrial data and calibration](docs/factory-accuracy-round2.md)
- [Factory model/data research and independent localization protocol](docs/factory-model-data-research-round2.md)
- [Construction: actual damage videos, annotation matching and measured failures](docs/construction-actionable-inspection.md)
- [Current factory and construction verification](docs/inspection-improvement-round2.md)

- [Factory recall and construction video update: measured results and remaining requirements](docs/factory-construction-update.md)

- [Real video playback, coverage and construction workflow](docs/video-inspection.md)
- [Saudi construction code research](docs/saudi-construction-inspection-study.md)
- [All 89 researched construction inspection topics](docs/saudi-construction-checklist.md)
- [Construction models, actual footage, measured accuracy and failures](docs/construction-vision.md)
- [Construction data/model availability and gaps](docs/construction-data-gap-study.md)
- [Factory accuracy improvement experiments](docs/factory-accuracy-improvement.md)
- [Implemented architecture and API](docs/runtime-architecture.md)
- [Real local inputs, datasets and model setup](docs/real-inputs.md)
- [Verification and laptop measurements](docs/verification.md)
- [Real-model integration checks and limits](docs/real-input-verification.md)
- [Before/after model and runtime performance study](docs/performance-study.md)
- [Factory implementation and limits](docs/modules/factory.md)
- [Freight implementation and limits](docs/modules/freight.md)
- [Road implementation and limits](docs/modules/roads.md)
- [Storage and retention](docs/runtime-storage.md)
- [Dashboard behavior](docs/ui.md)
- [Target product architecture](docs/product-plan.md)
- [Implementation backlog](docs/implementation-backlog.md)
- [Market research](research/market-study-2026-10-20/market-study.md)

Synthetic demo scenes and fictional demo documents are generated by this repository. Real-input presets use explicitly downloaded public datasets and locally installed models; their provenance, licenses and evaluation limits are documented in the module guides. Those licenses do not become licenses for new customer footage.
