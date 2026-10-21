# Local road condition inspection

Real images, folders, videos and cameras use **YOLO12s locally**, with a portable ONNX Runtime CPU default and an explicit OpenVINO preset for this laptop's Intel UHD 620 GPU. It identifies longitudinal, transverse and alligator crack candidates, pothole candidates, and separately identifies repaired surfaces. Detection and evidence stay local. These are human-review suggestions, not engineering assessments.

See [measured CPU/GPU optimization and quality tradeoffs](../roads-optimization.md). Fresh fixed28 comparisons measured 1.99 FPS at the old 640/two-thread CPU settings, 2.53 FPS at 640/four threads, 5.73 FPS at 416/four threads, and 15.44 FPS at 416 on the laptop's integrated GPU. The 416 exports retained the same 29 TP and 6 FN on this development diagnostic, with 5 FP instead of 8; training overlap and preset selection bias prevent an independent accuracy claim. The faster 320 export lost seven additional true detections and is an explicit tradeoff, never the default. These are profile timings; runtime stress is recorded separately.

The normally installed application subsequently measured **16.31 process FPS** over 84 timing observations with the same diagnostic counts. Actual `EdgeRuntime` varying-public-image replay at **12 FPS/150 ms for 30 seconds** processed **360/360 frames with zero drops, late frames or errors**, host-to-decision p95 **79.5 ms**. Python socket connections were denied. Select the explicitly named **laptop Intel GPU 416px** public preset for that configuration. GPU startup took about 21 seconds before capture; steady frame timing does not include this initialization. The **CPU balanced 416px** presets use 4 FPS/500 ms; the no-options portable default preserves 640 resolution and four CPU threads.

**The model author does not publish its training/test membership. Our 28-image result has possible training overlap: it is a compatibility diagnostic, not independent accuracy or generalization evidence.**

## Installed default model

[rezzzq/yolo12s-road-damage-rdd2022](https://huggingface.co/rezzzq/yolo12s-road-damage-rdd2022), pinned revision `59b19d21f6ee0c5ad44a6455b36204a0e283b34b`, provides a five-class RDD2022 checkpoint. We verified its class order and exported it locally.

| Item | Value |
|---|---|
| Source | `models/roads/yolo12s-rdd2022-fiveclass.pt`, 18,966,618 bytes |
| Source SHA256 | `ccf91e3199d81ee761f4a3a012da752939c9d72e9b1eaa242e508db3163270a0` |
| ONNX | `models/roads/yolo12s-rdd2022-fiveclass.onnx`, 37,222,294 bytes |
| Current ONNX SHA256 | `72d63019c84fadfc9777cd56734d3f79a12ca40f18dfccf352b8c3d7ad810d37` |
| Contract | Float32, opset 17, fixed input `[1,3,640,640]`, output `[1,9,8400]` |
| Class order | D00, D10, D20, D40, Repair |
| Portable default runtime | ONNX Runtime CPU, four intra-op threads, one inter-op thread, sequential graph, non-spinning workers |
| Export tools | Python 3.13.5, Torch 2.14.1+cpu, Ultralytics 8.4.182, ONNX 1.23.2 |

Setup verifies the pinned source checkpoint SHA. Export records its artifact SHA in `models/roads/fiveclass-manifest.json`; runtime validates that hash before loading. Regeneration may change export metadata and hence the artifact hash; the manifest captures the actual export. Tools live in the separate ignored `.venv-roads-export`; the application `.venv` needs neither Torch nor Ultralytics. The tool lock is `output/evaluation/roads-export-tools-lock.txt`.

## Public data and evidence

[RDD2022's original authors](https://github.com/sekilab/RoadDamageDetector) provide road images and VOC XML from six countries with several capture setups. The original [Figshare record](https://doi.org/10.6084/m9.figshare.21431547) declares CC BY 4.0. Attribution: Arya, Deeksha; Maeda, Hiroya; Ghosh, Sanjay Kumar; Toshniwal, Durga; Sekimoto, Yoshihide, *RDD2022: A multi-national image dataset for automatic Road Damage Detection* (dataset 2022; article 2024).

`datasets/roads-rdd2022-test-balanced` contains 28 real images: four SHA256(filename)-ranked cases from each of China Drone, China MotorBike, Czech, India, Japan, Norway and United States. Selection was fixed before scoring the five-class model. Images came from [DetectionBench's provider test split](https://huggingface.co/datasets/dronefreak/RDD2022), pinned at `72b01f94c2e5bfe430861afd73995f2d412b8855`. This is neither the official CRDDC test nor a verified holdout for the separate YOLO12s author.

We found mirror-label discrepancies, including repaired areas labeled as potholes. We therefore retrieved **original VOC XML for all 28 images**, verified image CRCs against the original archive, and used original labels as the primary diagnostic. The audit fetched only 7,766,648 range bytes from the 13 GB archive. Source image SHA256s, original XML, CRC checks and both annotation sources are retained. The mirror declares CC BY-SA 4.0; both its notice and original attribution are preserved.

The original pre-optimization 640/two-thread run at confidence .25 and class-aware IoU .5 gave **29 true positives, 8 false positives and 6 false negatives** (precision 78.4%, recall 82.9%). Possible training overlap prevents interpreting these as deployment accuracy. Repaired-surface observations are excluded from four-class damage matching. That historical profile-process p50 was **593 ms**, p95 **758 ms**, maximum **772 ms**, averaging **1.68 FPS**; its former 1 FPS recommendation is superseded by the explicit measured presets in the optimization guide. File decoding, camera exposure and browser display are excluded. They do not establish vehicle-speed inspection coverage.

See [validation and rejected alternatives](../roads-real-validation.md), `output/evaluation/roads-real-fiveclass/report.json`, and annotated images beside it. Socket connections were denied during evaluation; physical network settings were not changed.

## Use real inputs

Select Roads, choose an image/image-folder/video/camera source, and Start. Real inputs automatically select the five-class model. Useful installed images are `datasets/roads-rdd2022-test-balanced/images/China_MotorBike_001578.jpg` (potholes) and `China_MotorBike_001563.jpg` (repairs and cracks). Their annotated outputs were visually checked.

Source API values are `image:<absolute path>`, `images:<absolute folder>`, `file:<absolute video>` and `camera:0`. Direct callers use `RoadsProfile({"demo": False})`. Synthetic sources and legacy direct `RoadsProfile()` retain the separately labeled [heuristic baseline](roads-heuristic-baseline.md).

| `profile_options` key | Default | Meaning |
|---|---|---|
| `model_variant` | `fiveclass` | `india` and `multicountry` are retained research alternatives |
| `engine` | `onnxruntime` | CPU inference; installed export also works with `opencv` |
| `input_size` | 640 | Installed fixed exports: 320/416/640; 512 requires explicit export. Smaller spatial inputs can lose small defects. |
| `cpu_threads` | 4 | CPU inference worker bound; 1–8 supported. |
| `cpu_spinning` | false | ONNX Runtime worker spin between calls; disabled to avoid idle CPU contention. |
| `openvino_device` | `CPU` | With `engine: openvino`, explicitly select `CPU` or verified local `GPU`; missing devices fail clearly. |
| `confidence` | .25 | Uncalibrated model-score threshold |
| `nms_iou` | .45 | Per-class duplicate-box suppression |
| `confirm_frames` | 2 | Matching observations before an event |
| `track_timeout_seconds` | 1.5 | Short image-space identity window |
| `model_path`, `model_sha256` | installed manifest | Exact-contract replacement; no automatic downloads |

Missing assets, hash mismatches and incompatible output shapes fail explicitly. Runtime never downloads weights.

## Processing and limits

Frames are letterboxed with value 114, converted BGR to RGB and divided by 255. The model output is decoded into original-image XYWH boxes with per-class NMS. Scores are not calibrated probabilities. Repaired surfaces have kind `road_surface_repair` and are not described as new damage. No detections returns `no_detection`, not damage-free assurance.

The tracker emits an event per short-lived image-space track with exact model SHA, and is bounded to 512 tracks. It does not supply physical defect IDs, geographic deduplication, motion compensation, depth, engineering severity or automatic repair decisions. Moving footage can skip small or fast-passing defects at any rate; actual camera geometry and vehicle speed determine coverage. Single-image evaluation uses `confirm_frames=1`. Repeated gallery images are software inputs, not independent video observations.

Country, viewpoint, lighting, weather and pavement differences remain unvalidated. The model can miss defects or confuse paint, shadows, repairs and damage. Use representative route-held-out footage with human ground truth before operational deployment.

## Reproduce setup and export

From the repository root in PowerShell:

```powershell
python -m venv .venv-roads-export
.venv-roads-export/Scripts/python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1+cpu torchvision==0.29.1+cpu
.venv-roads-export/Scripts/python.exe -m pip install ultralytics==8.4.182 onnx==1.23.2 onnxscript==0.7.2
.venv-roads-export/Scripts/python.exe -m scripts.setup_roads_fiveclass
.venv/Scripts/python.exe -c "from scripts.setup_roads_multicountry import dataset; dataset()"
.venv/Scripts/python.exe -m scripts.fetch_original_roads_annotations
.venv/Scripts/python.exe -m scripts.evaluate_roads_multicountry --variant fiveclass --engines onnxruntime
.venv/Scripts/python.exe -m pytest tests/test_roads.py tests/test_roads_model.py -q
```

The export script verifies source SHA and class names, exports float32 at opset 17 with `dynamic=False`, `simplify=False`, `half=False`, `device=cpu`, `nms=False`, runs ONNX's checker and writes tool/hash provenance. Setup uses internet; inspection uses installed assets locally.

## Licensing

The model card declares MIT. Its YOLOv12/Ultralytics derivation has an [AGPL-3.0 upstream license](https://github.com/sunsmarterjie/yolov12/blob/2abab7153a065fb2925e8088e9ca2b19016ab7d6/LICENSE); Ultralytics also documents [AGPL and enterprise terms](https://www.ultralytics.com/license). The pinned upstream license and model card are saved beside the weights. We preserve these facts and do not treat the card's MIT label as clearance for proprietary redistribution. Resolve applicable model terms or replace the weights before proprietary distribution. Dataset and model terms are separate.
