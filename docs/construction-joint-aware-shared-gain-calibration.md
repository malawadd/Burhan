# Original crack model with shared camera gain: failed calibration comparator

Adding the existing low-light gain to the original joint-aware crack model did **not** improve localization at the original confidence threshold. Neither its selected operating point nor any tested point met the declared precision and negative-image false-positive guard. This experiment changes no application defaults, registrations or active models.

## Actual results

| Condition | Confidence | Crack TP / FN | Unmatched model boxes | Precision | Recall | Verified negative images flagged |
|---|---:|---:|---:|---:|---:|---:|
| Historical raw input, rescored with the same criterion | .25 | 8 / 117 | 10 | 44.44% | 6.40% | See original baseline report |
| Actual shared-gain input, same cutoff | .25 | 8 / 117 | 16 | 33.33% | 6.40% | 3 / 10 |
| Actual shared-gain input, predeclared failed best-F1 fallback | .10 | 13 / 112 | 42 | 23.64% | 10.40% | 4 / 10 |

No confidence in the fixed grid `.10,.15,.20,.25,.35,.45,.55,.65,.75,.85` met precision >=80% and verified-negative image FPR <=10%. The failed fallback localized 5/62 geometrically thin annotations (8.06% recall). The minimum 50% all-crack recall, 80% precision, 50% thin-box recall and <=10% negative-image FPR continuation checks all failed. Exact selected-point serving-decoder parity passed on all122 inputs.

Here an unmatched box means a model box without a class-consistent publisher-box match at IoU >=.50, under the available annotation scope. It is not automatically a verified material false alarm. Negative-image FPR comes separately from the ten complete-polygon DACL negatives. Thin means publisher-box short side <=30 native pixels and long side >= twice the short side; this is neither physical crack width nor pixel-mask accuracy.

## Preregistered, local execution

The [protocol](C:/Users/IAM/Documents/hackathon/opencv/project1/output/evaluation/construction-round3/joint-aware-shared-gain-calibration/protocol-before-forward.json) was recorded before any forward, SHA256 `069d8f5f3a5d5d964df155dae5cd2db46b4e71eec72bae79cf315c58e6f38856`. The [executable script](C:/Users/IAM/Documents/hackathon/opencv/project1/scripts/evaluate_joint_aware_shared_gain_calibration.py) pins original model SHA256 `4d48232d7b1427ac468022d2cc11a344885629cac71ad48bd2ff4c6659302246`, original class order `joint,crack`, and shared gain source SHA256 `451b3f899fee6217066b425c664e3b3b9689cf81fd67dc1450eb5d9c90ad8707`.

The external wrapper applies native-frame `camera_exposure(..., low_light_gain)` once. Internal model gain remains off; unchanged original inference performs640-square letterbox, RGB/255, original NMS(.45), global top100 including joints, and original crack decoding. Joints remain context and are hidden after decoding. No ROI, ground-truth model input, dilation, box inflation or class remapping was added.

All122 original calibration inputs were run:75 crack-annotated frames,37 unsupported-only spalling frames and10 verified negatives. Crack TP/FN/FP is scored on85 frames, excluding the37 unsupported-only frames; all38 spalling annotations remain explicitly unsupported. This comparator cannot replace spalling training. Data was already reused for calibration: there is **no independent validation credit**, no heldout opening and no claim about unseen sites.

The historical raw baseline uses saved actual predictions, independently rescored against identical annotation IDs, geometry, support and matching rules; it was not rerun or replaced by simulation. Gained inputs were freshly forwarded on local OpenVINO GPU f32, first at the minimum cutoff for raw-output caching, then at the selected point for unchanged-serving parity (244 forwards total). All122 gained native input hashes matched the preserved R4 preparation contract, and serving input-tensor SHA parity passed. Construction CPU training ran concurrently, so this is an **accuracy-only** experiment with no throughput claim.

The completed [summary](C:/Users/IAM/Documents/hackathon/opencv/project1/output/evaluation/construction-round3/joint-aware-shared-gain-calibration/summary.json) SHA256 is `60e4987d113cf99bf63b40c11578fcb2aac09c7e3d3323fe1f113261925790cc`; the [run journal](C:/Users/IAM/Documents/hackathon/opencv/project1/output/evaluation/construction-round3/joint-aware-shared-gain-calibration/run-journal.json) records completion. All122 actual raw NN output caches, all per-frame predictions, full grid and failures are preserved beside it. No still image was made into a video.

## Fixed examples, including failures

Every panel shows **original RAW | actual native gained INPUT | publisher GT | actual selected model boxes**. These are fixed first cases by declared category, not outcome-selected examples. Blue GT is crack; gray GT is unsupported spalling; green model boxes are matched and orange are unmatched/unscored under the declared scope. They are box comparisons, not segmentation-mask accuracy evidence.

- [First crack case](C:/Users/IAM/Documents/hackathon/opencv/project1/output/evaluation/construction-round3/joint-aware-shared-gain-calibration/fixed-first.jpg): visible crack and a localized model box; ordinary lighting leaves input unchanged.
- [First night case](C:/Users/IAM/Documents/hackathon/opencv/project1/output/evaluation/construction-round3/joint-aware-shared-gain-calibration/fixed-night.jpg): gain makes the actual dark crack visible to the viewer; the model still produces no crack box in this saved case.
- [First unsupported spalling case](C:/Users/IAM/Documents/hackathon/opencv/project1/output/evaluation/construction-round3/joint-aware-shared-gain-calibration/fixed-unsupported.jpg): gray GT marks material loss; model boxes a vertical feature instead. Spalling remains unsupported and this frame is unscored for crack false positives.
- [First verified negative](C:/Users/IAM/Documents/hackathon/opencv/project1/output/evaluation/construction-round3/joint-aware-shared-gain-calibration/fixed-negative.jpg): separate complete-polygon negative with no model box in this fixed example; four of all ten negatives are flagged at the selected point.

Visual inspection of all four saved panels confirms that real raw/gained inputs and distinct GT/model outputs are displayed. One successful fixed example does not override the failed aggregate benchmark. The artifact retains ev-CIVIL CC BY4.0 source data and DACL CC BY-NC4.0 negative-data limitations, original model AGPL lineage and uncleared merged training-source redistribution rights. It is experimental visible-observation evidence, not Saudi code certification or construction acceptance.
