> Archived first integration. The default is now the five-class model documented in [roads.md](roads.md). Values below describe the retained India-only research option.

# Local road damage inspection

Real image, image-folder, video and camera sources now use a downloaded **YOLO11n road damage ONNX model on this laptop CPU**. Synthetic demo mode retains the separate [dark-region heuristic](roads-heuristic-baseline.md), so its authored animations are not represented as neural model validation.

## Model and public data

The [model author](https://github.com/VivekVagale/road-hazard) reports fine-tuning YOLO11n on 7,706 RDD2022 India images. The export has input `1x3x640x640` and output `1x8x8400`: XYWH boxes and scores for longitudinal crack, transverse crack, alligator crack and pothole. OpenCV DNN runs the exact export with its CPU backend. No PyTorch, GPU, cloud inference or paid endpoint is needed.

The file `models/roads/yolo11n-rdd2022-india.onnx` is 10,607,022 bytes. SHA256: `a93a8e1735aaf6cd502d13be54684b6335eb8cca1eec270d053e43550531e1f9`. The author hosts weights outside Git, so setup pins the bytes and rejects changed downloads. Source metadata and upstream README/license are retained beside it. Code/sample revision: `84673b89d7def3dcb2211ef00a10e3eb0318d0dd`.

[RDD2022's original authors](https://github.com/sekilab/RoadDamageDetector) provide labeled train images and unlabeled test images from six countries. The [Figshare record](https://doi.org/10.6084/m9.figshare.21431547) reports **CC BY 4.0**. Attribution: Arya, Deeksha; Maeda, Hiroya; Ghosh, Sanjay Kumar; Toshniwal, Durga; Sekimoto, Yoshihide, *RDD2022: A multi-national image dataset for automatic Road Damage Detection* (dataset 2022; article 2024).

Two small real-image sets are installed:

- `datasets/roads-rdd2022-samples`: 12 real images the model author identifies as held-out RDD2022 test examples. These copies have no accompanying annotations; they are inference smoke cases only.
- `datasets/roads-rdd2022-czech-subset`: first 16 lexicographic Czech train images and original VOC XML annotations, read from the original Figshare archive using bounded HTTP byte ranges. ZIP CRCs are verified and individual SHA256s recorded. The 13 GB archive is not downloaded. This is a convenience slice, not representative sampling or a new benchmark. Country separation follows the author's India-only training report; we did not audit training. Adjacent images may share a route.

## Install and reproduce

From the repository root:

```powershell
.venv/Scripts/python.exe scripts/setup_roads_assets.py --labeled-subset --count 16
.venv/Scripts/python.exe scripts/evaluate_roads_real.py
.venv/Scripts/python.exe -m pytest tests/test_roads.py tests/test_roads_model.py -q
```

Asset acquisition is an explicit setup command. Running the detector never uses the network. Missing or changed weights fail explicitly. Runtime passes `demo=False` for real inputs, selecting the model automatically. Direct callers use `RoadsProfile({"demo": False})` or `{"backend": "model"}`. Legacy `RoadsProfile()` defaults to the synthetic heuristic for compatibility.

Select Roads in the dashboard, choose an image/image-folder/video/camera source, then Start. The installed labeled folder is `datasets/roads-rdd2022-czech-subset/images`. Source API values are `images:<absolute folder path>` and `image:<absolute image path>`. `profile_options` accepts `confidence` (default .25), `nms_iou` (.45), `confirm_frames` (2), `track_timeout_seconds` (1.5), `model_path`, and `model_sha256`. Replacement weights must match the documented export/class contract. These thresholds are working defaults, not calibrated field acceptance criteria.

## Inference and events

The detector letterboxes with value 114, converts BGR to RGB, scales by 1/255, forwards the CPU model, decodes the exact output contract, clips boxes to original input coordinates, and applies per-class NMS. Scores are **uncalibrated model scores**, not probabilities of real damage. Results request human review. No detections returns `no_detection`, never a damage-free assurance.

Two overlapping observations of the same class confirm a candidate by default. An IoU tracker suppresses repeat events within 1.5 seconds. Tracks are local image-space identities, not unique physical road defects; no geographic deduplication is claimed. Reappearing after timeout can create a new event. Source/session/resolution/time rollback resets tracking. Independent still-image evaluation uses `confirm_frames=1`. `describe()` exposes model identity, exact SHA256, classes, thresholds, input shape and licensing status. Events include the model SHA256. No GPS, depth, engineering severity or repair decision is calculated.

## Validation

Decoder tests cover coordinate reversal, per-class NMS, incompatible exports, nonfinite outputs, explicit missing-model errors and automatic real-input selection. Legacy synthetic tests remain separate. Public-image timings, class-aware IoU matching and annotated images are written to `output/evaluation/roads-real/`. See [real-input evidence](../roads-real-validation.md) for measured results. The evaluation uses IoU at least .5 and confidence at least .25. It does not calculate AP or re-use the author's GPU timing as a laptop result.

## Limits and licensing

The model learned mostly paved Indian roads and has documented weaknesses for rough rural roads and transverse cracks. Weather, countries, night footage, viewpoints and tiny defects can differ substantially. Damage can be missed; markings, repairs and shadows can produce false alarms. A laptop webcam can verify software input but is not representative road inspection. Meaningful route-held-out validation remains necessary.

The upstream repository declares MIT, but the model derives from Ultralytics YOLO11 pretrained weights. [Ultralytics describes AGPL-3.0 and enterprise licensing](https://www.ultralytics.com/license) for its models and derived works. We preserve both provenance facts and **do not treat the MIT file as clearance for proprietary weight redistribution or commercial deployment**. This is a local research/hackathon integration. Resolve applicable model terms or replace with independently cleared weights before proprietary distribution. Dataset CC BY 4.0 attribution is separate from model terms.
