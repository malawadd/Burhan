# Extended fastening fit: measured results

Date: 2026-10-21. This is a local research comparison on previously exposed development photographs. **No category is released.**

The final model completes eight logical epochs at 512px, batch 4, freeze first 4 layers: the first epoch used one thread and contiguous layout; seven remaining epochs used four threads and channels_last. Saved EMA model and optimizer values restored exactly. RNG, sampler and uninterrupted raw-model trajectories are not restored; this is a distinct candidate. See the [full experiment record](construction-expansion-sdnet-benchmarks.md).

## Before and after at their previously frozen calibration points

| Candidate / category | TP | Annotation-unmatched predictions | FN | Annotation-relative precision | Recall |
|---|---:|---:|---:|---:|---:|
| YOLO320 two-epoch / visibly_unseated_connection | 0 | 0 | 65 | — | 0.00% |
| YOLO320 two-epoch / fastener_absence_appearance | 4 | 0 | 67 | 100.00% | 5.63% |
| SSDLite320 200-step / visibly_unseated_connection | 19 | 160 | 46 | 10.61% | 29.23% |
| SSDLite320 200-step / fastener_absence_appearance | 6 | 67 | 65 | 8.22% | 8.45% |
| Extended YOLO512 actual served / visibly_unseated_connection | 24 | 79 | 41 | 23.30% | 36.92% |
| Extended YOLO512 actual served / fastener_absence_appearance | 4 | 0 | 67 | 100.00% | 5.63% |

The 98 photographs are the same exposed v1 TEST group, now explicitly **reused development**. Thresholds were selected on CAL before final development inference. The first two rows per category are earlier candidate outputs; “before” does not imply that an earlier fastening model was reliable. Available annotations are not verified cross-class exhaustive, so unmatched predictions are not proven physical false alarms.

## Actual localization and publisher Fixed-only views

| Category | Frozen threshold | Mean IoU of matched served boxes | Flagged Fixed photographs | Boxes on Fixed photographs |
|---|---:|---:|---:|---:|
| visibly_unseated_connection | 0.025 | 0.740 | 9 / 30 | 15 |
| fastener_absence_appearance | 0.900 | 0.880 | 0 / 30 | 0 |

Matches require correct-category box IoU ≥0.5. This detector produces boxes; actual-mask experiments have separate region/pixel reports. Only 30 publisher Fixed photographs are present, with unknown physical-site independence; normal-view performance remains exploratory.

## Training and execution evidence

- Resume fit duration: 2026.3s for seven resumed epochs, including framework setup/final validation. Initial epoch timing is retained separately.
- The original first-epoch checkpoint, before-resume protocol, model/optimizer restoration, actual layout/thread receipts, RNG limitation note, CAL cuts and CPU-ONNX parity are retained in the experiment directories.
- The actual deployed adapter report binds model, manifest, dataset and adapter hashes; its per-photograph predictions are retained. Concurrent observations are not isolated inference qualifications.
- All inference uses the laptop. No owner recordings entered TRAIN/CAL or threshold selection. Public training photographs remain identified as photographs.
- The required independent 50-positive/50-negative per-category validation is absent. This report establishes neither production fastening correctness nor torque, concealed embedment or code compliance.

Machine artifacts:

- `output/training/construction-expansion-sdnet-v2-performance/results.json`
- `output/training/construction-expansion-sdnet-v2-performance/served-development.json`
- `models/construction/expansion-sdnet-v2-performance/yolov8n/manifest.json`

## Real owner fastening video: unresolved plate-gap failure

The [complete three-panel comparison](../output/evaluation/construction-guided-expansion/owner-fastening-ssd-v1-vs-yolo-v2-native-pts/original-vs-research.mp4) shows the same207original decoded frames: original / SSDLite v1 / extended YOLOv8n512. Both models use their previously frozen CAL operating points. All207views pass the unchanged capture gate; there is no quality-rejection explanation for the absence of detections.

The old model produces candidates on25frames; the extended model produces **zero candidates on207/207frames**. The visible plate-to-wall separation remains unassessed in the fixed midpoint below. **This is a miss of the owner's requested case, not a demonstrated false-alarm improvement or proof of secure fastening.** Publisher bolt/nut Loosen labels do not establish an arbitrary mounting-plate-gap class. Plate-gap inspection remains unresolved/unavailable.

![Actual midpoint: original, old model, new model; the visible separated plate is unassessed](../output/evaluation/construction-guided-expansion/owner-fastening-ssd-v1-vs-yolo-v2-native-pts/frame-00103.jpg)

The output preserves original variable presentation intervals relative to the first frame; maximum decoded timestamp difference is0.000445ms and all207output frames are verified. Offline processing took47.13seconds, not a live speed benchmark. Source/video/renderer/model hashes, input frame hashes and every before/after prediction are in the adjacent `report.json`. The private input never entered training, calibration or threshold selection; no expert frame-level ground truth or fastening reference is invented.

The app records this as **Owner fastening before-after / SSD v1 vs YOLO v2 - FAILED research**, under failure diagnostics, with the matching original source preset. [Final runtime measurements](construction-expansion-runtime-benchmarks.md) separately measure the selected extended model at29.53encoded preview fps but only7.23actual model fps on this input. Smooth preview does not repair the miss.
