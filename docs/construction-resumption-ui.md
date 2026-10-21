# Construction UI correction: failed models are not working inspection

Date: 2026-10-21. This change addresses what the actual application communicated after the construction expansion. It does not retrain a model, change thresholds, repair detection or establish a new qualified category.

## What the app actually showed

The running local server offered eight installed experimental checkpoints. Most selector explanations said only that no independent release was established. The current-result panel displayed **UNASSESSED**, but an empty frame said only **No findings reported for the latest frame**. Saved owner-video case descriptions were generic. These labels did not make the severe measured failures or wrong task semantics clear enough.

The observed server was stopped on the owner damp-wall video, with PHELE SSDLite selected. PHELE detects publisher peeling/Mold appearance, not a trained humidity class; its lack of findings on this source could be mistaken for a successful inspection. Having eight executable files is not having eight working construction capabilities.

## Implemented product changes

- Renamed the selector to **Experimental model Â· not qualified for inspection**, with **NOT QUALIFIED** on its choices.
- Added a prominent evaluation panel directly below the selector. It states the exact model's trained scope, concrete frozen failures, and the owner diagnostic outcome where available.
- Bound every numerical summary to the exact ONNX SHA-256 returned by the catalog. A replacement file with a new hash cannot inherit an old checkpoint's scores. Unknown hashes show no accepted performance evidence.
- No-model selection explicitly says **NO DETECTOR Â· CONDITIONS NOT ASSESSED**. Capture guidance cannot detect or rule out faults.
- Tile geometry remains distinguished: no classifier selected means geometry observations may run, not a trained tile-fault detector. Image-space edge bands do not establish missing grout or installation correctness.
- Empty construction output now says faults may have been missed and that this is not an inspection pass. Current guided results are labelled **NOT QUALIFIED**; actual error statuses remain visible.
- Research category badges say **NOT QUALIFIED**, rather than giving an ambiguous impression of usable detection.
- Added prominent failure explanations to the saved owner fastening, fastening before/after and damp-wall cases.

The selector still requires explicit model selection. No model, source, inference configuration, threshold or camera was automatically changed by this work.

## Exact evaluation summaries exposed

These are the existing frozen reports, not new evaluation. Publisher box matching and genuine-mask matching remain separate. For SDNET, cross-class annotation completeness and physical-site independence are unknown; annotation-unmatched proposals are not proven physical false alarms.

| Checkpoint | Failure made visible |
|---|---|
| PHELE SSDLite | 19 TEST photos: peeling 4/22 box matches, 9 unmatched predictions; 30.8% precision / 18.2% recall. Mold 1/3 matched; one prediction cannot establish reliability. Not a dampness model. |
| PHELE YOLO | Peeling 6/22 matched, 40 unmatched; 13.0% precision / 27.3% recall. Mold 0/3 matched. |
| SDNET SSDLite | Unseated 19/65 matched, 160 annotation-unmatched proposals; absence 6/71 matched, 67 unmatched. Publisher Fixed images flagged 21/30 and 8/30. |
| SDNET YOLO v1 | Unseated 0/65 matched; absence only 4/71 matched. Empty outputs do not establish correctness. |
| SDNET extended YOLO v2 | Reused development photos: unseated 24/65 matched, 79 unmatched, 23.3% precision / 36.9% recall; absence 4/71 matched. Owner plate-gap video: 0 candidates / 207 frames. |
| SDNET LR-ASPP masks | 14/65 unseated and 10/71 absence masks matched at mask IoU >=0.5; Fixed images flagged 6/30 and 13/30. |
| SDNET YOLO segmentation | 17/65 unseated masks matched; all 71 absence instances missed. Fixed unseated flags 10/30. |
| Wall-stain SSDLite | Actual CPU adapter: 54/125 boxes matched, 1,917 unmatched; 2.74% precision / 43.2% recall. All 51 other-defect negatives flagged. |

Sources: construction-expansion-phele.md, construction-expansion-sdnet-benchmarks.md, construction-expansion-sdnet-v2-results.md, construction-expansion-sdnet-masks.md and construction-expansion-wall-stain.md. No category satisfies independent release gates.

## Saved real clip audit: what the boxes actually locate

### Owner fastening / mounting-plate separation

Input: VID-20261020-WA0001.mp4. Both saved outputs use the actual 207 source frames; private footage never entered fitting or calibration.

The real video visibly shows the mounting plate separated from the wall. The requested inspection is plate seating/stand-off. Public SDNET labels target loose or missing bolt/nut appearance; they do not establish an arbitrary plate-to-wall-gap detector.

At source frame 18, 0.634 seconds, the old SSDLite emits fastener_absence_appearance with bbox [264,203,63,61] in the 368x496 source view, score 0.2568. Browser playback was paused at that actual frame and visually inspected. The box lies on the horizontal rail beside the connection, to the right of the separated plate. It does **not** localize the visible plate-to-wall gap or establish a required missing bolt. This is an unscored wrong-region candidate for the owner's requested issue; no expert pixel ground truth is invented.

The old model has candidates on only 25/207 frames. The extended model has none on all 207. The frame-103 midpoint shows the separated plate clearly in the original, old-model and new-model panels with no useful box. The newer empty overlay is a miss of the requested condition, not proof that the connection is secure. Torque, concealed embedment and approved connection geometry remain unassessed.

Artifacts inspected:
- output/evaluation/construction-guided-expansion/owner-fastening-ssd-v1/report.json
- output/evaluation/construction-guided-expansion/owner-fastening-ssd-v1-vs-yolo-v2-native-pts/frame-00103.jpg
- output/playwright/construction-resumption-fastening-box-audit.png, paused browser playback of source frame 18
- output/playwright/construction-resumption-plate-failure.png, prominent case explanation

### Owner damp-wall / stain appearance

Input: VID-20261020-WA0002.mp4. The owner reports humidity damage; this report does not measure wall humidity.

The dedicated stain model produces candidates on all 330 actual frames. At fixed frame 165, source 5.543 seconds, overlapping yellow boxes cover broad wall regions, the skirting/baseboard and portions of the floor. They are detector proposals for publisher leakage/stain appearance, not a validated moisture boundary, measured percentage or diagnosed leak cause. Their clutter prevents a useful localized assessment. No independent owner-frame annotation exists, so a numerical recall or false-alarm score on this clip would be unsupported.

The separate PHELE SSDLite can produce no findings on this wall because its trained source categories are peeling paint and publisher Mold. That silence also does not rule out the reported damp condition.

Artifacts inspected:
- output/evaluation/construction-guided-expansion/owner-damp-stain-ssd-v1-native-pts/report.json
- output/evaluation/construction-guided-expansion/owner-damp-stain-ssd-v1-native-pts/frame-00165.jpg
- output/playwright/construction-resumption-model-failure.png, actual stain-model evaluation panel

These clips expose unresolved failures. They are not successful construction inspection demonstrations.

## Verification

JavaScript syntax check passed. Real Edge browser QA against http://127.0.0.1:8765 confirmed the selected PHELE model shows its exact failed peeling/Mold results, the stain model shows 2.74% precision / 43.2% recall, empty output states that faults may be missed, and an empty selector explicitly says no detector is selected. Saved diagnostic case selection displays the concrete plate-gap/stain explanations. The old fastening video was played locally and paused at 0.6341 seconds for visual review; no live inference session was started.

The same local damp source and PHELE SSDLite selection were restored. The runtime remains stopped; no camera was selected or opened. These UI checks do not alter or replace the earlier speed benchmarks.
