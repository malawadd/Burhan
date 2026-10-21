# Real before/after video evidence

20 October 2026. These are complete public camera recordings with actual local model outputs, not generated damage or still photographs presented as video. **Thin cracks and spalling are not fixed.** Every comparison retains misses, unmatched detections and its development-data limitations.

## Crack semantic masks: fixed original versus final300

The pair was frozen before inference: original TUT export versus the exact final300 pixel-supervised export, both at strictly greater than .50. No failed calibration cutoff or favorable checkpoint was substituted. The input is the original camera frame, with the same documented native exposure policy applied once. Four separate panes show raw camera input, actual exposure-adjusted input, external publisher boxes and the model's real semantic trace. Yellow is the raw threshold mask; blue is the served area4/top100 mask. **Neither color asserts pixel correctness.**

| Complete recording | Before matches / misses / unmatched | After matches / misses / unmatched | Result |
|---|---:|---:|---|
| Thin crack89 |0 /8 /20|0 /8 /49|Still fails; unmatched detections increase|
| Day crack129 |8 /63 /60|7 /64 /84|Regresses|
| Night ros_crack21 |68 /374 /1,667|136 /306 /3,563|More matched boxes, substantially more unmatched detections|

All counts use unchanged class-aware, one-to-one publisher box IoU≥.50. Annotations are partial and frame instances are correlated. Unmatched detections are counted by that rule; their material correctness is not independently adjudicated. There are no video pixel masks, physical crack-width measurements or independent field qualification.

| Recording | Before video | After video |
|---|---|---|
| Thin crack89 |[Actual complete video](../output/evaluation/construction-pixel-videos/display-v2/before-crack_89.mp4)|[Actual complete video](../output/evaluation/construction-pixel-videos/display-v2/after-crack_89.mp4)|
| Day crack129 |[Actual complete video](../output/evaluation/construction-pixel-videos/display-v2/before-crack_129.mp4)|[Actual complete video](../output/evaluation/construction-pixel-videos/display-v2/after-crack_129.mp4)|
| Night ros_crack21 |[Actual complete video](../output/evaluation/construction-pixel-videos/display-v2/before-ros_crack_21.mp4)|[Actual complete video](../output/evaluation/construction-pixel-videos/display-v2/after-ros_crack_21.mp4)|

The [original paired summary](../output/evaluation/construction-pixel-videos/paired-summary.json) links every original report and SHA256. Each report retains every source frame ID and timestamp, predictions, annotations, raw/input/mask hashes, scores and provenance. The original NPZ files retain bit-packed raw and served native masks. Encoded source presentation timestamps were checked exactly in microseconds. Offline export elapsed time is not a real-time inference benchmark.

The links above use a **display-only v2**: publisher boxes and identical saved masks appear over the same recorded gain-adjusted native input so the dark material remains visible; the RAW pane is unchanged and the footer is outside all panes. `render_saved_crack_video_evidence.py` performs zero model inference calls. Every raw/input/mask hash and all700 scored records remain unchanged; exact original presentation timestamps and frame counts are verified again. Original videos/reports remain beside the immutable NPZ evidence. [All six display files and hashes](../output/evaluation/construction-pixel-videos/display-v2/all-six-presentation-manifest.json) and [registration checks](../output/evaluation/construction-pixel-videos/display-v2/registration.json) preserve the derivation. Local HTTP verification also confirms all14 comparison videos and their complete source-equal JSON reports remain available.

A separate reviewer subsequently audits all700 records across the six videos: original camera and gain-input hashes, packed mask hashes/counts, served masks contained in raw masks, component area and emitted-box agreement, complete encoded frame counts and exact source presentation timestamps all pass. The audit does not independently reproduce the original384 probability arrays, threshold decisions, mean component scores or top100 ordering: those arrays were not retained in the video NPZ, although per-frame probability hashes are recorded. This limit is retained rather than credited as a stronger parity result.

The app exposes all six under **Construction → Include failure diagnostics and context footage → Crack mask diagnostic**. Each recording has a matching raw-input preset for local inference. Input15FPS in that preset is a requested replay rate, not a measured construction capacity claim.

![Actual app video result with complete failure counts](../output/evaluation/construction-pixel-videos/ui-night-crack-after-display-v2.png)

## Spalling and the separate two-class camera fit

The earlier fixed pair compares the R4 head against the completed1171-view real low-light continuation. All four complete sources and both models are retained. The spalling75 recording has **0/77 localized material-loss annotations before and after**. It cannot truthfully be shown as repaired detection. Across the unchanged122-input camera calibration, material-loss matches rise4/38→5/38 and unmatched boxes fall28→1; these remain inadequate recall and do not transfer to success on the complete spalling75 clip.

- [Spalling75 before](../output/construction-development-evcivil-yolov8n-r4-head-640-gain-spalling_75.mp4)
- [Spalling75 after](../output/construction-development-evcivil-yolov8n-r4-real-lowlight-640-gain-spalling_75.mp4)
- [All eight whole-video results and exact registration proposal](../output/evaluation/construction-round3/four-development-video-integration-proposal.json)
- [Training, calibration and whole-video measurements](construction-accuracy-round3.md)

The historical seen-training spalling54 clip contains actual hits and misses, but is explicitly marked **seen during training**, not evidence of generalization. Missing surface material is not automatically confirmed concrete cover loss or a structural defect.

## Post-hoc known-hit full recordings

These two additional source sequences were selected **after calibration hits were known**. They show genuine damage being localized, along with every failure in the full recordings. They provide no independent improvement credit and do not override the failed fixed cases above. The pair is best epoch1 versus the exact final epoch4 from the same completed fit, with unchanged crack .10 / material-loss .25 cutoffs.

| Real complete recording | Before matches / misses / unmatched | After matches / misses / unmatched | Videos |
|---|---:|---:|---|
| Crack129,104 exposures |1 /70 /1|5 /66 /12|[Before](../output/construction-posthoc-before-crack_129.mp4) · [After](../output/construction-posthoc-after-crack_129.mp4)|
| Spalling76,249 exposures |48 /159 /4|50 /157 /35|[Before](../output/construction-posthoc-before-spalling_76.mp4) · [After](../output/construction-posthoc-after-spalling_76.mp4)|

This is a poor recall/false-alert tradeoff: whole-clip precision falls from92.31% to58.82% across all emitted classes in the spalling clip (material-loss-only precision94.12%→59.52%). The before model already localizes visible damage. The after model is not a qualified repair.

At3.400002s, actual crack129 source frame68 contains a localized thin right-hand crack segment (box IoU.5986) and a missed left-hand segment. At4.300002s, actual spalling76 frame86 shows the model localizing a visibly damaged wall surface beside windows (box IoU.7422). The substrate and whether the missing material is paint, plaster or concrete are not established. These are exact decoded video frames, not generated examples.

![Actual thin-crack hit and adjacent miss at the same time](../output/evaluation/construction-posthoc-known-hit-videos/qa/after-crack_129-actual-source-frame-68.jpg)

![Actual localized surface material loss](../output/evaluation/construction-posthoc-known-hit-videos/qa/after-spalling_76-actual-source-frame-86.jpg)

A separate root audit of all706 frame records passes: raw H5 and lossless replay pixels, independently recomputed native gain/input hashes, before/after source and annotation equality, exact encoded microsecond presentation timestamps and full frame counts, plus independently recomputed report-box IoU/counts. It performs zero NN calls and does not independently verify semantic annotation correctness, pixel GT or rerun predictions. [Audit receipt](../output/evaluation/construction-posthoc-known-hit-videos/root-independent-media-audit.json), [complete paired counts](../output/evaluation/construction-posthoc-known-hit-videos/complete-posthoc-paired-summary.json) and [explicit post-hoc protocol](../output/evaluation/construction-posthoc-known-hit-videos/two-posthoc-known-hit-protocol-before-forward.json) preserve the scope.

## New author segmentation model: full recordings, including failure

The CUBIT author nano segmentation model is now installed as **FAILED camera calibration research**. It emits actual prototype masks, rather than turning a detector rectangle into a damage mask. Its complete 122-input calibration fails: crack 12/125, spalling 0/38, thin 7/62, and 5/10 target-negative photographs flagged. It receives no qualification or default adoption.

The paired BEFORE is the exact cached best-epoch1 camera model, with its original crack .10 / spalling .25 policy; AFTER is the author model at fixed .25 for both classes. All five source selections were fixed before this candidate's inference; spalling76 remains explicitly supplementary **post-hoc** because of its historical selection. Counts below are matches / misses / unmatched boxes across every scored exposure:

| Complete original recording | Before | After | Before / after videos |
|---|---:|---:|---|
| Thin crack89 | 0 / 8 / 8 | 0 / 8 / 7 | [Before](../output/construction-development-evcivil-yolov8n-r4-real-lowlight-640-gain-crack_89.mp4) · [After](../output/construction-cubit-author-v2-crack_89.mp4) |
| Day crack129 | 1 / 70 / 1 | **38 / 33 / 3** | [Before](../output/construction-development-evcivil-yolov8n-r4-real-lowlight-640-gain-crack_129.mp4) · [After](../output/construction-cubit-author-v2-crack_129.mp4) |
| Spalling75 | 0 / 77 / 0 | 0 / 77 / 29 | [Before](../output/construction-development-evcivil-yolov8n-r4-real-lowlight-640-gain-spalling_75.mp4) · [After](../output/construction-cubit-author-v2-spalling_75.mp4) |
| Night crack21 | 10 / 432 / 412 | 0 / 442 / 177 | [Before](../output/construction-development-evcivil-yolov8n-r4-real-lowlight-640-gain-ros_crack_21.mp4) · [After](../output/construction-cubit-author-v2-ros_crack_21.mp4) |
| Post-hoc spalling76 | 48 / 159 / 4 | 0 / 207 / 1 | [Before](../output/construction-posthoc-before-spalling_76.mp4) · [After](../output/construction-cubit-author-v2-spalling_76.mp4) |

Daytime recall improves 1.41%→53.52%, with precision 50.00%→92.68%. It remains incomplete detection. The new model abstains on spalling; fewer spalling alarms cannot be credited as useful detection. Other material-loss recordings above retain genuine older-model hits and all failures.

At actual source frame3 / 0.150000s, the author model traces the visible horizontal crack, with box IoU .82947 and 1,816 predicted foreground pixels. The green trace is visibly wider than the dark crack core. There is no video pixel GT or physical width measurement, and matching a box does not establish mask accuracy. The first-annotated frame1 miss is retained separately; it was not replaced by this hit.

![Actual app recording: mask trace, complete counts and source-frame evidence](../output/evaluation/construction-cubit-author-v2/videos/ui-actual-crack-mask-and-counts.png)

An independent no-inference audit passes all **683 original AFTER exposures and 270 saved native masks**, with 0 empty masks. It checks original camera arrays/H5 and lossless replay pixels, gain/input hashes, cached BEFORE source/GT equality, saved mask geometry/hash/counts, report matching, full encoded frame counts and exact microsecond PTS. It does not rerun predictions or independently validate annotation semantics/pixel accuracy. [Audit](../output/evaluation/construction-cubit-author-v2/videos/root-independent-five-video-audit.json), [full model study](construction-cubit-author-candidate.md), and [all 23 registered comparison videos served successfully](../output/evaluation/construction-cubit-author-v2/registered-twentythree-http-audit.json).

The app exposes these under **Construction → Include failure diagnostics and context footage → CUBIT author masks**. Loading the matching source selects the actual failed research model and the lossless video reader. A separate integration repair corrected 24 earlier recording presets that incorrectly selected the image reader; [repair receipt](../output/evaluation/construction-source-type-repair/repair.json). This is a playback/source-routing fix, with no model or accuracy change. The station remains stopped until Start is pressed.

## Real pixel-ground-truth photographs

An additional fixed-pair diagnostic evaluates all102 reserved LCSD photographs after pre-inference duplicate screening:102 clean, zero quarantined and all completed. Original TUT→final300 at strict>.50 changes micro pixel precision59.47%→65.25%, recall67.64%→43.35%, IoU.46296→.35219 and Dice.63291→.52092. The fitted model regresses recall and IoU. The photographs contain crack-positive material and cannot establish healthy-wall false-alarm rate. Their role, source/pretraining overlap uncertainty, exact model binding, complete masks and all-case results are documented in [the pixel-supervision record](construction-pixel-supervision-probe.md).

## Expanded finishes

The app also runs a separately fitted normal-only tile texture specialist on real manufacturing photos. All117 test photos and the actual serving filter were evaluated: crack pixel precision31.14%, recall59.71%, IoU25.73%; derived crack-region IoU≥.50 matches0/17; normal-photo flags3/33. It detects an unclassified texture anomaly rather than a semantic crack subtype. It has no installed-floor alignment, lippage, adhesion, humidity or video qualification. [Finish datasets and all measured results](construction-finishes-expansion.md) retain the failures, actual masks and six fixed photo examples.

Wall leakage appearance, detachment, corrosion, bulging and crack supervision uses real MBDD2025 photos with complete five-class labels and split quarantine. The first four-epoch model and the separately declared twelve-additional-epoch continuation both fail every class guard. The latter localizes3/22 leakage,8/60 detachment,1/23 corrosion and1/10 bulge CAL boxes, with12/1/2/0 unmatched predictions respectively; crack remains0/19. These are measured appearance-only photo experiments, not qualified video inspection. Dataset acquisition or an inspection checklist alone is not model support. Paint coverage and tile alignment remain separate, unsupported checks until a fitted model and appropriate real evidence pass evaluation.

Sources: [ev-CIVIL author dataset](https://doi.org/10.11583/DTU.25924228), [LiteCrackSeg author code](https://github.com/1Kaleb/LiteCrackSeg), [MVTec AD publisher](https://www.mvtec.com/company/research/datasets/mvtec-ad), [MBDD2025 publisher record](https://zenodo.org/records/15622584). Individual source/model rights and restrictions remain recorded in their manifests; new data do not clear inherited model restrictions.
