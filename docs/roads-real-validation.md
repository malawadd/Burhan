# Road real-input validation

This preserves the pre-optimization model-selection evidence. Current CPU thread settings, explicit 416 input/GPU presets, unchanged-weight accuracy comparisons and 12 FPS normal-app runtime results are in [road model local optimization](roads-optimization.md). The original reports below remain unchanged.

**Performance update:** measurements below record the earlier implementation. Current model/runtime changes, before/after accuracy checks, measured presets and regression results are in [the laptop performance study](performance-study.md). Historical results are preserved for comparison.

This records actual local inference and known failures. It is not a field qualification or a claim of production road accuracy.

## First candidate rejected for general road inspection

The first integration used [VivekVagale/road-hazard](https://github.com/VivekVagale/road-hazard), a 10.6 MB YOLO11n ONNX model trained on RDD2022 India. The exact model and source are retained as the explicitly selectable `model_variant: india` research option. It is no longer the intended default for general road inputs.

Original [RDD2022 Figshare](https://doi.org/10.6084/m9.figshare.21431547) VOC XML and images were downloaded for the first 16 lexicographic Czech train images using only **1,645,954 HTTP range bytes**, avoiding a full 13 GB archive download. Twelve images had no in-taxonomy boxes; six annotated boxes covered D00 (1), D10 (4) and D40 (1). D20 was absent.

At confidence .25 and class-aware IoU .5, the model had **0 true positives, 1 false positive and 6 false negatives**: precision 0, recall 0 on this tiny slice. This rules out claiming that this India-trained model is validated for Czech or global road inspection. The implementation was checked for class order (D00/D10/D20/D40), VOC one-based inclusive to zero-based XYWH conversion, and original-coordinate decoding. A separate India sample's model boxes aligned with visible road regions when inspected. The negative result was retained rather than concealed by changing thresholds on the scored examples.

The 12 author-provided real India sample images produced candidates on 11 of 12 images, but have no accompanying public annotations. That is an **inference smoke result**, not precision, recall or successful road classification.

OpenCV DNN on all 28 initial real images (three separate warmups; two CPU threads) measured profile-process p50 **329.18 ms**, p95 **430.72 ms**, p99 **435.36 ms**, maximum **436.37 ms**, mean process throughput **3.18 FPS**. Processing includes model letterbox/inference/decoding/annotation, excluding image load, capture and browser rendering. It is a short observed workload, not a sustained latency guarantee. A subsequent five-measurement same-image check found ONNX Runtime median **201.86 ms** against OpenCV **433.90 ms** under the then-current laptop load; this motivated testing ONNX Runtime for the replacement.

Evidence: `output/evaluation/roads-real/report-india.json`, annotated images in the same folder, and dataset SHA256 manifests. The evaluation blocked Python `socket.connect` while inference ran; networking settings were not physically disabled.

## Replacement selection protocol

The next candidate was [DetectionBench's multi-country YOLOv8n](https://huggingface.co/dronefreak/rdd2022-yolov8n), investigated for its documented multi-country training, explicit AGPL-3.0 model terms, small 6.25 MB checkpoint and CPU-compatible export. Its author's benchmark scores are not laptop results. It was not selected as the final default because of the training-label discrepancies below.

Before scoring, four images per seven RDD capture partitions were selected from the provider's test split by ascending SHA256 of filename: 28 cases across China Drone, China MotorBike, Czech, India, Japan, Norway and United States. Selection used neither labels nor model confidence. This is a small stratified convenience diagnostic, distinct from the first Czech16 slice, although seeking a replacement followed that slice's failure. We did not audit training or route independence. The provider claims a holdout for its YOLOv8n; that does not establish a holdout for the separate five-class model author. It is not official unlabeled CRDDC test data.

The mirror's `Czech_000010` retains a D10 box but omits an original D40 box; that case is outside the fixed 28. Within the 28, `China_MotorBike_001563` maps two original Repair boxes to class 3 (named pothole by that model), and `China_MotorBike_001578` omits two original D40 boxes. The global YOLOv8n option therefore labels class 3 as a disputed training-label surface anomaly, not a trustworthy pothole classification.

Original VOC XML was retrieved for **all 28 cases** using 7,766,648 byte-range bytes. All 28 mirror images' CRC32s matched the original archive. Original and mirror annotations are preserved. Original Figshare metadata declares CC BY 4.0; the mirror declares CC BY-SA 4.0. Both notices and attribution are retained. VOC decimal coordinates in Norway were retained as floats; one-based inclusive boxes are converted to zero-based XYWH. The primary diagnostic uses original explicit D00/D10/D20/D40 labels. Repair and other country codes such as D44 are excluded from the four-class damage assessment rather than silently remapped.

## Final default: five-class YOLO12s

[rezzzq/yolo12s-road-damage-rdd2022](https://huggingface.co/rezzzq/yolo12s-road-damage-rdd2022) supplies a model with explicit D00/D10/D20/D40/Repair classes. Its checkpoint metadata confirmed that class order. It was exported locally to 37,222,294-byte ONNX and selected as the default real-input model. Repaired-surface observations remain distinct from damage candidates.

**Its author does not publish the training split. All numbers below are compatibility diagnostics with possible training overlap, not independent accuracy.** The 28 fixed cases were selected before this model's inference, but choosing a larger model followed the initial model's poor results and the second candidate's label issues. No threshold was tuned on the diagnostic images. No AP, official challenge or field-performance claim is made.

| Model / same 28 cases | Original VOC TP / FP / FN | Diagnostic precision / recall | Profile p50 / p95 | Mean process FPS |
|---|---|---|---|---|
| YOLOv8n multi-country, research option | 18 / 19 / 17 | 48.6% / 51.4% | 220 / 243 ms | 4.75 |
| YOLO12s five-class, default | 29 / 8 / 6 | 78.4% / 82.9% | 593 / 758 ms | 1.68 |

The original four-class ground truth has 35 boxes. Matching uses confidence .25, per-class NMS .45 and class-aware IoU .5, greedy by score. Repair observations are excluded from damage counts. Against the mirror's differing labels, the five-class model produced TP27/FP10/FN9 on 36 boxes; these are recorded for comparison, not preferred over the original labels. Mirror annotation bounds were checked, with zero invalid boxes in the 28 selected rows.

Timing used ONNX Runtime 1.23.2 CPU provider, two intra-op threads, one inter-op thread and three warmups. Profile timing includes model letterbox, inference, decoding and annotation; source decode/1920-pixel cap, exposure and UI rendering are excluded. No sustained or production latency guarantee follows. Recommended laptop source rate is 1 FPS with a 2,000 ms deadline.

A separate same-image engine check (three warmups, five measurements) found ONNX Runtime median 460.54 ms, p95 528.55 ms versus OpenCV DNN median 604.02 ms, p95 1036.03 ms. All five resulting boxes, labels and scores matched to four decimals on that case. This is limited equivalence evidence, not an exhaustive export-accuracy test. See `output/evaluation/roads-real-fiveclass/engine-comparison.json`.

Visually checked outputs: `China_MotorBike_001578.jpg` localizes two potholes beside a crossing; `China_MotorBike_001563.jpg` separates repaired surfaces from longitudinal cracks. The original VOC labels support these examples. The evaluation itself is stronger than these two favorable illustrations because it includes all fixed 28 cases and negative results.

`models/roads/fiveclass-manifest.json` records exact source/export SHA256s, class contract, tool versions, export arguments and licensing provenance. `output/evaluation/roads-real-fiveclass/report.json` contains complete per-image predictions and counts. `output/evaluation/roads-real-multicountry/report.json` retains the rejected candidate's results. [The module guide](modules/roads.md) gives exact regeneration commands and license links. The 21 road tests passed after the final model/default changes.
