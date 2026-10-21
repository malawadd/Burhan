# Frozen R4head thin-box diagnostic: detections are absent or wrong class

The failed R4head result is **not explained by nearby crack boxes narrowly missing the IoU cutoff**. None of the62 geometrically thin crack annotations has any crack prediction anywhere in its frame at the frozen operating point. Primary localization remains **0/62 at IoU.50**. No confidence, box, class, ROI, decoder or acceptance rule was changed.

The [read-only audit script](C:/Users/IAM/Documents/hackathon/opencv/project1/scripts/audit_construction_frozen_thin_geometry.py) uses actual saved122-frame calibration serving records only; it opens no raw NN cache, image, mask or heldout data. The [complete diagnostic JSON](C:/Users/IAM/Documents/hackathon/opencv/project1/output/evaluation/construction-r4head-frozen-thin-geometry-v2.json), SHA256 `4dc7cffe2e19c5896cae40988e5cfce02c76a43a808fe037f0c2f5ecfea36f4c`, retains per-annotation geometry and an earlier diagnostic hash. Source runtime-record SHA is `33f58c2480b48d7e798170ab08d618cc631b52718a0605eb18c2f3ad2fd28559`; frozen operating-point SHA is `dafef4b77079e01902dbb2b44059b8d167dc3349657482c088d5b1088af89183`. Actual model identity is recorded from that operating point; no model was loaded.

Frozen confidence is **crack.45 / spalling.10**, NMS.45, primary class-aware one-to-one matching IoU.50. The original matching IDs and frozen score-cutoff consistency were independently reproduced. Thin uses the original publisher-box criterion: native short side<=30px and long side>=twice short side. These are62 annotations in44 correlated calibration frames, not62 independent defects or a physical crack-width measurement.

| Frozen prediction situation | Thin annotations | Distinct frames |
|---|---:|---:|
| No predictions of either class |40|32|
| Spalling predictions only |22|12|
| Any crack prediction, anywhere in frame |0|0|

Of the22 annotations with spalling-only predictions,21 intersect a spalling prediction. Those are wrong-class box overlaps, not crack localization. Their best-overlap spalling box is typically much larger than the crack annotation:

| Geometry, conditional on21 wrong-class intersections | Median | Range |
|---|---:|---:|
| Box IoU |.0423|.0261–.2584|
| Publisher GT **box** coverage |100%|97.14–100%|
| Prediction/GT box area ratio |23.65×|3.77–38.25×|
| Center offset, native pixels |53.27|29.27–62.54|
| Center offset / GT box diagonal |.543|.254–.989|

Same-class IoU/area/offset/coverage distributions are **null because no crack prediction exists**, not zero-valued measurements. Covering a publisher rectangle with a large spalling rectangle is neither accurate crack localization nor pixel-mask coverage. Annotation scope is partial, so wrong-class overlap suggests confusion/background firing at this frozen point without proving verified material false alarms or a training cause.

The descriptive one-to-one rerun uses the **same actual predictions and original saved order**, competing against all original crack/spalling GT:

| Matching IoU used only for this diagnostic | Thin crack matches /62 | All crack matches /125 |
|---|---:|---:|
|.10|0|8|
|.25|0|8|
| Primary.50|0|5|

These supplementary.10/.25 counts are not new success criteria, confidence selection or model promotion. By source, missing/only-wrong-class thin annotations are crack12912/1, crack1309/1, nightcrack217/0 and nightcrack2212/20. Absence of frozen crack predictions dominates; source22 also has wrong-class box firing. These records cannot establish whether lower raw confidences contain useful cracks, why fitting failed, or whether different training will fix it. No alternate confidence outputs were opened. YOLO's continuous decoded boxes and this camera-derived box criterion do not establish a stride-imposed minimum crack width or any structural/Saudi-code conclusion.
