# Factory recall and construction video update

20 October 2026. This report distinguishes implemented, measured behavior from requirements that remain unmet. All inference runs locally on the laptop. Real public images and actual recordings supply model evidence; unit-test fixtures are not evidence of real-world detection.

## Factory result and decision

**The requested actual factory defect-recall improvement has not been achieved.** A mask-overlap audit found that the review policy's apparent extra positive was a box away from the annotated defect. Its 81.8% figure is flagged-image classification recall; even permissive any-overlap localized defect recall stays **8/11 = 72.7%**, the same as baseline. This distinction takes precedence over earlier frame-level summaries.

The original comparison contains only **11 defective and 69 normal views from 10 physical items**. It has now been reused for development comparisons and is not a blind production acceptance set. Replaying its pixels does not add independent accuracy samples.

The original native model caught 8/11 defects (72.7% recall), with 1/69 false-positive normal views. A newly trained MobileNet texture/segmentation candidate caught the same 8/11 defects with no comparison false positives. Its paired complete-profile mean/P95 fell from 29.54/36.76 ms to 24.89/29.72 ms, excluding capture/queue/UI. It improved precision on this small reused comparison but **did not improve recall**. The candidate was frozen before comparison and is retained as an experimental alternative, not a production qualification.

The first calibration-selected recall policy uses unchanged original forest weights with threshold 0.99 and minimum component area eight. It flagged 9/11 defective images (81.8% image-classification recall), with three false-positive normal views and 75% image-classification precision. Only eight of those nine boxes overlapped a real annotated defect. The extra flagged image was a false localization, so this is **not an actual defect-detection recall improvement**. Its overall image-classification accuracy also falls from 95% to 93.75%. Further sensitivity-policy evaluation, internal item-separated validation and sustained before/after runtime reports are documented in [the factory study](factory-accuracy-improvement.md).

Several retrained forests and neural heads failed calibration or comparison. Their protocols, results and rejected artifacts are retained. No missed comparison image is relabelled, silently excluded or counted as an abstention. No operating point is chosen from comparison miss scores. Production-line acceptance remains unmet without an agreed false-alarm budget and new independent line data.

### Corrected localization objective and internal validation

The final predeclared calibration policy selected the original forest at 0.975 / area eight, with a maximum calibration normal-view false-positive rate of7/70. It maximized localized defective-view recall, then annotated foreground-pixel recall, then pixel precision. It selected the same point as the more permissive review tier. Frozen comparison: **8/11 localized defective views**, unchanged from baseline; normal false positives rose **1/69 to7/69**. Foreground pixel recall rose11.37% to19.78%, pixel precision fell96.88% to88.45%, and micro pixel IoU rose0.1133 to0.1928. These pixel changes did not recover another defective view. This is not shipped as a recall upgrade. Reports: `localized-frozen-comparison.json` and `review-box-overlap.json` under `output/optimization/factory-accuracy/`.

Four-fold internal validation refitted every learned forest on20 inner-training physical items, selected each policy on10 inner-calibration items, and evaluated10 excluded outer items; all40 original training/calibration items were covered, excluding the10 exposed comparison items. It did not reuse the original learned weights. These are internal, reused-development-data results, not external blind acceptance.

| Policy | Localized defective views / all41 | Normal false positives | Abstentions |
|---|---:|---:|---:|
| Original fixed point, fresh fold-local weights |34/41 (**82.9%**) |4/278 |0 |
| Review3/70 calibration budget |28/41 (**68.3%**) |6 normal views |11 defective +69 normal views |
| Review7/70 calibration budget |30/41 (**73.2%**) |19 normal views |11 defective +69 normal views |

One fold had no eligible operating point within the declared review budget, so its80 outer views were unassessed. Accepted-only recall must not conceal this lost coverage. The original fixed operating point remains the default. Detailed fold choices, original-mask localization counts, pixel metrics and predictions are retained in `item-crossvalidation-review/mask-localization-report.json`.

## Construction delivered

- A fourth local profile accepts actual video, camera and image inputs. Construction rejects synthetic/demo input and PDF input.
- Two explicitly selected local models provide PPE observations or surface-crack candidates. Model hashes, versions, sources and scope are recorded.
- Real video follows source time by default, with source indices, timestamps, decode/replay lag, original/analysis dimensions, source skips, queue drops and late decisions exposed. Sequential review is a separate playback mode and does not guarantee every-frame inference.
- A Saudi research catalog lists **89 inspection families across 16 stages**, from preconstruction through operation/changes, with required evidence and official references. The dashboard filters it by stage. All remain unassessed; observations never issue a code pass.
- A separate viewer serves allowlisted annotated recordings of actual public footage, including documented model failures. These saved results are distinct from the active inspection stream.
- Explicit normalized ROI and box-only rendering are available for a configured target field. Outside-ROI pixels remain unassessed. The ROI does not automatically track material or establish physical dimensions.

The complete researched list is [the checklist](saudi-construction-checklist.md); [the Saudi study](saudi-construction-inspection-study.md) explains source verification, code edition/project scope and unresolved authority issues. [The data-gap matrix](construction-data-gap-study.md) covers every visual family and the other evidence categories.

## What the construction models actually achieved

| Diagnostic | Measured outcome | Practical limit |
|---|---|---|
| PPE, 169 public images / 885 ground-truth objects | 631 TP / 185 FP / 254 FN; 71.30% recall, 77.33% precision. CPU 9.61 to GPU 20.04 serial profile FPS; predictions unchanged. | Unknown training overlap; not Saudi site or video-event validation. |
| Crack, 112 public images / 148 ground-truth objects | 101 TP / 29 FP / 47 FN; 68.24% recall, 77.69% precision; pixel IoU 0.6169. CPU 6.84 to GPU 13.78 profile FPS; predictions unchanged. | Small-image performance; not full-resolution video, material identity or structural acceptance. |
| Actual site clip, five manually labelled primary-worker frames | Original stretched geometry: person 0/5, bare head 0/5. Letterbox: person 3/5, bare head **1/5**. | Tiny correlated, reused failure diagnostic. Current bare-head recall is only **20%**; unsuitable as dependable safety monitoring. |
| Actual 4K bridge excerpt, full-frame crack masks | 423 candidate detections in 149/150 frames, many on steel joints; 6.69 serial profile FPS. | No independent video truth. This is a material false-candidate example, not evidence of hundreds of actual cracks. |

Letterbox geometry was selected using a separate square-image validation split and an aspect-preservation tie-break; it was not selected by optimizing those five video labels. All original failures remain in the recorded reports.

### Source-paced shared-runtime measurement

The PPE model inspected actual NIOSH footage at a requested 15 FPS for 15 seconds: **225/225 captured observations processed, zero queue drops, zero late decisions, zero inspection errors**, P95 **65.5 ms**, maximum 89.5 ms. Session-average rate was **14.8 FPS** including EOF/drain overhead. Source recording rate was 23.976 FPS: **134 intervening source frames were skipped**. Final replay lag was 16.7 ms; model initialization took 14.03 seconds separately. This is execution/throughput evidence, not independent PPE accuracy. Raw report: `output/evaluation/construction-runtime-ppe.json`.

The actual raw 4K bridge run with box-only rendering **failed the requested 10 FPS**: 32/32 captured observations processed in a 10.8-second session, approximately **3.0 FPS**, with 267 source frames skipped, no queue drops/late/errors, P95 **128.6 ms**, and 404.3 ms final replay lag. Initialization took 14.16 seconds separately. Skipping still requires the codec to advance through the original 4K recording; model-only speed does not overcome that source cost. Report: `output/evaluation/construction-runtime-crack-boxes.json`. ROI and public-image box parity measurements are recorded in [construction vision](construction-vision.md). Do not substitute model-only throughput for source-speed video performance.

The separately identified **1280×720 source proxy** passed a scheduled10 FPS,10-second run: **100/100 captured observations processed**, no queue drops/late/errors,197 intervening source frames skipped, session rate9.8 FPS including drain, P9584.9 ms, max141.7 ms, final replay lag28.4 ms; initialization15.58 seconds separately. Report: `output/evaluation/construction-runtime-crack-720p.json`. This is an offline downscaled, lossily encoded copy of the original real frames, with364/364 decoded frames, zero FFmpeg duplicates/drops and nominal-FPS CFR timestamp normalization explicitly recorded. The original irregular timestamps are not claimed preserved. Loss of fine detail can change crack accuracy; this is not equivalent-input accuracy parity, a live4K conversion benchmark or a15 FPS crack claim.

## Requirement that cannot be met by RGB video

The product does **not** detect or certify every Saudi construction inspection. The catalog contains nine potentially visual families and 80 families requiring other evidence; even the visual families are only partially modelled. Soil bearing/compaction, concrete strength, hidden reinforcement, exact cover/clearance/slope, pressure/leakage, earthing, fire-system performance, design compliance and authority approval require appropriate real measurements, tests, documents or qualified decisions. No public all-stage Saudi acceptance dataset was verified. Unsupported checks stay unassessed.

The matched 60-second factory runtime runs processed **900/900 observations at a scheduled 15 FPS**, with zero dropped/late/error observations in both runs: P95 **56.1 ms before / 56.4 ms review policy**, max **95.3 / 91.6 ms**. A 30-second 16 FPS acceptance run processed **480/480**, zero drops/late/errors, **15.9 session-average FPS** including EOF/drain overhead, P95 **57.8 ms**, max 88.0 ms. These are clocked, predecoded real public-still replays, not actual production-line video or additional independent accuracy data. Reports are in `output/optimization/factory-accuracy/runtime-*.json`.

The supplied real data consists of narrow public image sets and recorded clips, with provenance and licenses. Stock footage, reused image labels and detector overlays are not manufactured ground truth for missing inspections. The next validation dataset needs independent original site/line videos, project/item-separated splits, missed-event labels and false alerts per operating hour, plus the actual instruments and records required by each nonvisual check.

## Reproduction and verification

- [Factory protocols, artifacts and accuracy results](factory-accuracy-improvement.md).
- [Construction setup, hashes, licenses, image/video evaluations and exports](construction-vision.md).
- [Real video timing, frame coverage and benchmark commands](video-inspection.md).
- [Runtime/UI review and focused regression evidence](video-runtime-review.md).

Parallel subagents handled factory learning, construction models/data and Saudi research/runtime review. CPU/GPU speed benchmarks used exclusive measurement windows; code/research work proceeded in parallel. Accuracy-only cross-validation may run alongside unrelated file/document work and must not be cited as a capacity benchmark.

## Final software and viewing checks

The full Python suite passed **140 tests**, with one dependency deprecation warning. Runtime dependencies passed `pip check`; JavaScript syntax passed. The actual H.264 NIOSH excerpt played to its10.01001-second endpoint in the in-app browser at1280×720, with no video error. The checklist visibly showed both residential/commercial notes while retaining NOT ASSESSED. See [the detailed runtime/UI review](video-runtime-review.md). These regression and codec checks are separate from accuracy and exclusive speed measurements.

The interactive PPE UI run processed595/603 observations in40.2 seconds, with8 queue drops and360 source skips; P9593.0ms, no late decisions or errors. This is a separate GUI workflow check, not a zero-loss capacity claim. The final browser error log was empty. The station is stopped, camera off, and the local server remains available.

![Actual attributed recording and unassessed Saudi plan](../output/construction-recorded-result.jpg)
