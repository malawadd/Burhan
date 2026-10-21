# Construction expansion: benchmark overview

Generated from preserved local JSON results. All models remain research; **zero released categories**. No threshold or winner is selected by this summary.

## Frozen CAL-selected points

TP requires native localization IoU≥0.5. Mask rows require actual mask overlap, not box overlap. Counts are objects/components, while positive/negative counts are photographs. Different datasets and fitting budgets must not be pooled or ranked as one accuracy score.

| Dataset | Architecture | Category | Match | CAL point | TP | Unmatched | FN | Precision | Recall | Positive / target-negative photos |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| SDNET boxes | YOLOv8n320 | `visibly_unseated_connection` | box | 0.5 | 0 | 0 | 65 | — (no predictions) | 0.00% | 40 / 58 |
| SDNET boxes | YOLOv8n320 | `fastener_absence_appearance` | box | 0.5 | 4 | 0 | 67 | 100.00% | 5.63% | 28 / 70 |
| SDNET boxes | SSDLite320 | `visibly_unseated_connection` | box | 0.25 | 19 | 160 | 46 | 10.61% | 29.23% | 40 / 58 |
| SDNET boxes | SSDLite320 | `fastener_absence_appearance` | box | 0.25 | 6 | 67 | 65 | 8.22% | 8.45% | 28 / 70 |
| SDNET masks | LR-ASPP320 | `visibly_unseated_connection` | mask | 0.5 | 14 | 84 | 51 | 14.29% | 21.54% | 40 / 58 |
| SDNET masks | LR-ASPP320 | `fastener_absence_appearance` | mask | 0.5 | 10 | 64 | 61 | 13.51% | 14.08% | 28 / 70 |
| SDNET masks | YOLOv8n-seg320 | `visibly_unseated_connection` | mask | 0.1 | 17 | 72 | 48 | 19.10% | 26.15% | 40 / 58 |
| SDNET masks | YOLOv8n-seg320 | `fastener_absence_appearance` | mask | 0.35 | 0 | 0 | 71 | — (no predictions) | 0.00% | 28 / 70 |
| PHELE boxes | YOLOv8n384 | `peeling_paint` | box | 0.05 | 6 | 40 | 16 | 13.04% | 27.27% | 11 / 8 |
| PHELE boxes | YOLOv8n384 | `mold_appearance` | box | 0.05 | 0 | 1 | 3 | 0.00% | 0.00% | 3 / 16 |
| PHELE boxes | SSDLite320 | `peeling_paint` | box | 0.35 | 4 | 9 | 18 | 30.77% | 18.18% | 11 / 8 |
| PHELE boxes | SSDLite320 | `mold_appearance` | box | 0.35 | 1 | 0 | 2 | 100.00% | 33.33% | 3 / 16 |
| MBDD stain boxes | SSDLite320 | `damp_stained_appearance` | box | 0.25 | 53 | 1918 | 72 | 2.69% | 42.40% | 47 / 51 |
| SDNET reused development | YOLOv8n512 continuation | `visibly_unseated_connection` | box | 0.025 | 24 | 79 | 41 | 23.30% | 36.92% | 40 / 58 |
| SDNET reused development | YOLOv8n512 continuation | `fastener_absence_appearance` | box | 0.9 | 4 | 0 | 67 | 100.00% | 5.63% | 28 / 70 |

SDNET opposite-class annotations are not verified exhaustive: unmatched predictions are not proven physical false alarms, and the target-negative photograph column includes opposite-class views. Publisher-Fixed-only flags are reported separately in the [fastening](construction-expansion-sdnet-benchmarks.md) and [mask](construction-expansion-sdnet-masks.md) records. Site independence is unknown. The stronger fit reuses the exposed 98-photo set as development.

PHELE has only three Mold-positive TEST photographs; one matched region and zero unmatched regions does not establish dependable 100% precision. MBDD negatives contain other source defects, not independently confirmed healthy walls. Its native float-box counts are shown here; served integer rounding moves one borderline IoU across 0.5, documented in the [stain audit](construction-expansion-wall-stain.md).

## Fixed 0.5 before and after training

Before means a fresh task head on the official initializer, not a previously working installed inspection model. Both columns use the same 0.5 confidence point on the same exposed source set. CAL-selected operating points above are separate.

| Dataset | Architecture | Category | Before TP / unmatched / FN | After TP / unmatched / FN |
|---|---|---|---:|---:|
| SDNET boxes | YOLOv8n320 | `visibly_unseated_connection` | 3 / 41 / 62 | 0 / 0 / 65 |
| SDNET boxes | YOLOv8n320 | `fastener_absence_appearance` | 2 / 28 / 69 | 4 / 0 / 67 |
| SDNET boxes | SSDLite320 | `visibly_unseated_connection` | 0 / 0 / 65 | 0 / 0 / 65 |
| SDNET boxes | SSDLite320 | `fastener_absence_appearance` | 0 / 0 / 71 | 0 / 0 / 71 |
| SDNET masks | LR-ASPP320 | `visibly_unseated_connection` | 0 / 3213 / 65 | 14 / 84 / 51 |
| SDNET masks | LR-ASPP320 | `fastener_absence_appearance` | 1 / 1202 / 70 | 10 / 64 / 61 |
| SDNET masks | YOLOv8n-seg320 | `visibly_unseated_connection` | 0 / 28 / 65 | 0 / 0 / 65 |
| SDNET masks | YOLOv8n-seg320 | `fastener_absence_appearance` | 0 / 4 / 71 | 0 / 0 / 71 |
| PHELE boxes | YOLOv8n384 | `peeling_paint` | 1 / 3 / 21 | 0 / 0 / 22 |
| PHELE boxes | YOLOv8n384 | `mold_appearance` | 0 / 9 / 3 | 0 / 0 / 3 |
| PHELE boxes | SSDLite320 | `peeling_paint` | 0 / 0 / 22 | 0 / 1 / 22 |
| PHELE boxes | SSDLite320 | `mold_appearance` | 0 / 0 / 3 | 0 / 0 / 3 |
| MBDD stain boxes | SSDLite320 | `damp_stained_appearance` | 22 / 15960 / 103 | 8 / 4 / 117 |

The source records retain pixel IoU/precision/recall, individual matches, failed exports and corrected numerical verification, source sample counts, original protocols, licenses and exact training budgets. Successful parity means a model was exported faithfully, including faithfully reproducing its failures.

## Real video, runtime and release

[Runtime benchmarks](construction-expansion-runtime-benchmarks.md) distinguish encoded preview, unique browser frames, guidance, actual inspected/model frames, dropped/skipped frames and displayed-result age. A smooth preview is not 15 fps inspection. [Real owner video records](construction-expansion-implementation.md) retain timestamp-matched original/prediction footage and known misses/clutter; those private videos never enter fitting or calibration and have no invented frame-level ground truth.

The [category catalog](construction-expansion-category-catalog.md) documents missing or unsuitable data. No fit meets the entire per-category precision≥95%, recall≥90%, 50 independent positives and negatives, real-video/runtime/measurement release protocol.

## Source result receipts

- `output/training/construction-expansion-sdnet-v1/yolov8n-results.json` — SHA256 `64a6b9e674d17ec436044822cf0775735c240418ab8c8e52c2e98aca92fa2f18`
- `output/training/construction-expansion-sdnet-v1/ssdlite320-results.json` — SHA256 `fd0337bd76e7f1a43307dc686504b119d347aa1a284f98b213d5ff1346aacb4f`
- `output/evaluation/construction-expansion-sdnet-masks/lraspp-results.json` — SHA256 `239aa65cd89329e7f23ae2e6084429bb1324775c380385feb06e8e8532aed136`
- `output/evaluation/construction-expansion-sdnet-masks/yolov8n-seg-results.json` — SHA256 `2ebb39a5b3e6c3c86bcea233abf4c310b456b172ab0b4e8a2822907e4c3ebbc3`
- `output/evaluation/construction-expansion-phele-v1/yolov8n-results.json` — SHA256 `02f23e0331281ca320b6a5cf40399fa1a99e9b1bdf38ec6fc2205e2c4b0b3667`
- `output/evaluation/construction-expansion-phele-v1/ssdlite320-results.json` — SHA256 `04518d69a5c4f405565bb1658baf7aed3702dc378639223c2171a6041236bc1e`
- `output/evaluation/construction-expansion-wall-stain-v1/ssdlite320-results.json` — SHA256 `0fb708bd85936d1967d07d98cb5bf3dbb140e36b9c064cb46b0125f921ea7ff6`
- `output/training/construction-expansion-sdnet-v2-performance/results.json` — SHA256 `cf6d7460ead264805ca531a6d46a5d7b71143fdca46e03c9359dea56b0efe406`

Reproduce with `.venv/Scripts/python.exe scripts/summarize_construction_expansion.py`. It only reads results and writes this overview plus a timestamped hash receipt; no model is fitted or evaluated.
