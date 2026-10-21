# User construction videos: local inspection and model comparisons

20 October 2026. The four original videos stay at `C:/Users/IAM/Documents/hackathon/opencv/videos`. They are **diagnostic inputs only**, never training/calibration inputs and never uploaded. No blanket recall, precision or false-alarm estimate is possible without frame-level ground truth.

## What the owner wants inspected

The owner identified the faults after the initial visual inventory and after the baseline model/options had been frozen. This context is kept separately from the initial visual impressions and model outputs; it is not a complete set of pixel or box annotations.

| Original file | Complete exposures / duration | Owner-reported fault | Assessment boundary |
|---|---:|---|---|
| `VID-20261020-WA0000.mp4` | 269 / 9.042 s | Cracks | Visible long horizontal wall crack. Its depth, substrate and cause remain unmeasured. |
| `VID-20261020-WA0001.mp4` | 207 / 6.931 s | Not screwed/fastened correctly | Review railing-to-wall fixings. Existing crack/damage models do not verify screw engagement, torque, anchor embedment, approved arrangement or capacity. No fastening detector is claimed. |
| `VID-20261020-WA0002.mp4` | 330 / 11.040 s | Humidity in wall | Models may show wet/stained or detached-finish appearance. The owner reports humidity, but no quantitative moisture reading, active-leak confirmation or cause is inferred from RGB footage. |
| `VID_20261020_075653953.mp4` | 72 / 2.420 s | Cracks | Assess the fine features around the upper wall/cornice. Initial contact-sheet review also noted irregular finish appearance; owner context does not become automatic model confirmation. Original rotation metadata is −90°, respected by the native decoder. |

All **878 original exposures** are retained, including blurred or poorly lit moments. The [source inventory](../output/evaluation/user-construction-videos-round4/original-video-inventory.json) records original file hashes, dimensions, rotation and every presentation timestamp. Twelve deterministic original-frame samples per clip were selected before model inference; no favorable-output selection was used.

## Fixed baseline

The [initial baseline plan](../output/evaluation/user-construction-videos-round4/baseline-plan-before-inference.json) fixes three existing local models before inference: the CUBIT author crack/spalling mask model at .25; the five-class MBDD wall model at its previously frozen failed-research operating points; and DACL512 at .50 for nine named relevant damage channels. These remain research models with previously documented failures.

Each original exposure passes through actual `ConstructionProfile.process` for every model. The original OpenCV5 decoder supplies native, automatically oriented BGR frames, matching the serving decoder family. PyAV supplies original container timing metadata, not the model pixels. Its initial contact-sheet color conversion is not claimed byte-identical to OpenCV. Model output masks are saved as actual packed pixels; detector-only wall boxes are never converted to mask truth.

The complete comparison video shows separate RAW, author-mask, wall-box and broad-mask panes. All emitted predictions are retained. Every prediction is marked **unscored**, and the dashboard must show **Not evaluated** for localization/false-alarm metrics rather than displaying fictitious zero errors. An empty output is not an inspection pass. Offline multi-model export timing is not a 15-FPS benchmark.

The baseline is complete: **878 original exposures, 2,634 source-model inferences**, plus nine initialization calls. The [independent audit](user-construction-videos-round4-independent-audit.md) passed 36,701 integrity checks, including all original native frame hashes, exact presentation timestamps and 2,515 saved native masks. Candidate selection uses public calibration only; these owner videos never select a checkpoint or cutoff. Any candidate AFTER recording will be added only after that choice is frozen.

| Owner input | Complete three-model BEFORE video | Machine-readable output |
|---|---|---|
| Wall crack, WA0000 | [Play comparison](../output/evaluation/user-construction-videos-round4/VID-20261020-WA0000/before-three-models-v2.mp4) | [Full report](../output/evaluation/user-construction-videos-round4/VID-20261020-WA0000/before-three-models-v2.json) |
| Fastening issue, WA0001 | [Play comparison](../output/evaluation/user-construction-videos-round4/VID-20261020-WA0001/before-three-models-v2.mp4) | [Full report](../output/evaluation/user-construction-videos-round4/VID-20261020-WA0001/before-three-models-v2.json) |
| Wall humidity report, WA0002 | [Play comparison](../output/evaluation/user-construction-videos-round4/VID-20261020-WA0002/before-three-models-v2.mp4) | [Full report](../output/evaluation/user-construction-videos-round4/VID-20261020-WA0002/before-three-models-v2.json) |
| Upper-wall cracks, 075653953 | [Play comparison](../output/evaluation/user-construction-videos-round4/VID_20261020_075653953/before-three-models-v2.mp4) | [Full report](../output/evaluation/user-construction-videos-round4/VID_20261020_075653953/before-three-models-v2.json) |

The author model emits crack candidates in 105/269 WA0000 frames but **none in all 72 upper-wall frames**. That absence is an important diagnostic failure against the owner's stated concern; it is not a measured recall because complete visible-fault frame annotations are unavailable. On WA0002 the wall model emits three leakage boxes and DACL emits ten wetspot components; those counts do not verify moisture. DACL also emits 1,045 cavity components on this clip, so raw activity counts plainly cannot substitute for useful inspection.

## Technical recovery and first saved output

The initial export stops on its first source exposure because the runner reads observations from `metrics`, while the actual profile returns them in `domain_state`. All three profiles had completed three initialization calls each; one source inference ran, without a completed frame/video result. The [failure receipt](../output/evaluation/user-construction-videos-round4/baseline-v1-technical-failure.json) and executed source are retained. A separately named [v2 plan](../output/evaluation/user-construction-videos-round4/baseline-plan-v2-before-inference.json) fixes only report-field access/output naming and adds the owner's now-supplied issue descriptions. Inputs, model options, cutoffs and complete source selection remain unchanged; this is not an accuracy-driven retry.

Actual WA0000 sourceframe0 shows the author model tracing much of the long wall crack with a narrow predicted mask. It omits other segments, and later sampled frames60/90 emit no author crack prediction. DACL also follows the main crack with a visibly thicker trace and a baseboard region. The wall appearance detector instead labels a broad box as bulge, which does not verify the reported crack. The [saved-output renderer](../scripts/render_saved_user_video_frame_round4.py) reconstructs actual cached masks with zero NN and verifies original-frame/mask hashes. These observations are qualitative and do not establish full-clip recall.

![Actual original first frame and three baseline outputs](../output/evaluation/user-construction-videos-round4/VID-20261020-WA0000/qa/before-source-frame-00000.png)

## Other owner-video observations

At the humidity clip's deterministic middle frame165 (5.543s), neither author nor wall model emits a candidate. DACL emits two small floor candidates (`cavity`, `rust`) rather than explaining the wall concern. At upper-wall frame0, author and wall models emit nothing; DACL labels five object/desk regions `rust`, not the reported wall cracks. At fastening frame120, the author mask follows a wall/member boundary as `crack`; this does not detect incorrect fastening. These are qualitative checks of actual cached output, with no favorable-frame replacement or threshold changes.

- [Humidity middle-frame output](../output/evaluation/user-construction-videos-round4/VID-20261020-WA0002/qa/before-source-frame-00165.png)
- [Upper-wall first-frame output](../output/evaluation/user-construction-videos-round4/VID_20261020_075653953/qa/before-source-frame-00000.png)
- [Fastening frame120 output](../output/evaluation/user-construction-videos-round4/VID-20261020-WA0001/qa/before-source-frame-00120.png)

Two additional **output-selected diagnostic frames** check what the moisture labels actually mean; they are not independent or favorably selected evaluation samples. The first wall-model `leakage` emission is [frame99](../output/evaluation/user-construction-videos-round4/VID-20261020-WA0002/qa/before-source-frame-00099.png): its box is on a floor-tile joint, not the wall. The first broad-model `wetspot` emission is [frame280](../output/evaluation/user-construction-videos-round4/VID-20261020-WA0002/qa/before-source-frame-00280.png): the large candidate regions are again on the floor while the visible lower wall strip is unlabelled. Therefore the small leakage/wetspot output counts must not be presented as successful detection of the reported wall humidity fault. No cutoff/model selection was changed after these checks.

## Dashboard integration and verification

All four comparisons are available in **Construction → Include failure diagnostics and context footage → Inspection case → Owner… / three-model BEFORE**. The [registration receipt](../output/evaluation/user-construction-videos-round4/local-registration-v1.json) preserves all prior configuration hashes and backups. Twelve matching single-model presets use `source_kind:file` and the absolute original paths; the source button loads the original camera file, never the annotated comparison as model input. The replay setting15FPS is not a measured capacity result. Live inference stays stopped during training and benchmarking.

The existing unsupported installation category now explicitly names **fastener/anchorage installation defects** and links the owner's example. It remains **NO DETECTOR**, with no model reference. Required fixing arrangement, engagement, embedment and capacity need separate project/product evidence.

Real browser verification exposed a report-schema issue: the API rejected null metrics and absent matching-IoU fields in wholly unannotated output. It now permits those unknown values only for explicitly unevaluated, annotation-free frames whose predictions are all unscored/unmatched. Scored-match geometry validation remains in force. The42 focused evidence/video API checks passed, including a new positive unknown-metric fixture and six rejection cases. No model output or benchmark was altered by this UI fix.

The actual UI successfully plays the1400×1240 comparison, seeks sourceframe0, displays all five metrics as **Not evaluated**, and loads the raw WA0000 path as **Local video file** with the correct author model. The inference station and camera remain stopped.

The [localhost delivery check](../output/evaluation/user-construction-videos-round4/localhost-delivery-v1.json) confirms all four HTTP evidence responses equal their local reports, all four video HEAD/range responses work, all twelve original-file/model presets are available, and the fastening category remains unsupported. The full app software suite passed **462 tests, with3 skips and1 dependency deprecation warning, in49.03s**. The new Torch-only loss fixture is intentionally excluded from the Torch-free serving environment and runs separately in the training environment; this is not model-accuracy evidence.

![Actual dashboard: source crack masks and unknown metrics](../output/evaluation/user-construction-videos-round4/ui-local-owner-video-unscored-v2.png)

## Trained CPU AFTER: all four original videos completed

The fixed public calibration selected no eligible checkpoint. Therefore all four AFTER recordings use the same exact900 **FAILED** diagnostic, never a video-selected winner. All878 original exposures and exact rational timestamps are retained. The cached original-author BEFORE uses GPU; AFTER changes both learned weights and backend to CPU. Private recordings never enter fitting/calibration/selection/upload. All quality metrics remain unknown.

| Owner recording / reported fault | Original frames | Author BEFORE frames with a candidate | Trained AFTER frames with a candidate | Full real before/after video |
|---|---:|---:|---:|---|
|VID-20261020-WA0000 / wall crack|269|105|215|[Watch](../output/evaluation/user-construction-videos-round4/after-public-cal-choice-cpu-recovery-v1/VID-20261020-WA0000/raw-cached-before-actual-after.mp4)|
|VID-20261020-WA0001 / incorrect fastening|207|13|34|[Watch](../output/evaluation/user-construction-videos-round4/after-public-cal-choice-cpu-recovery-v1/VID-20261020-WA0001/raw-cached-before-actual-after.mp4)|
|VID-20261020-WA0002 / wall humidity report|330|8|9|[Watch](../output/evaluation/user-construction-videos-round4/after-public-cal-choice-cpu-recovery-v1/VID-20261020-WA0002/raw-cached-before-actual-after.mp4)|
|VID_20261020_075653953 / upper-wall cracks|72|0|1|[Watch](../output/evaluation/user-construction-videos-round4/after-public-cal-choice-cpu-recovery-v1/VID_20261020_075653953/raw-cached-before-actual-after.mp4)|

**These are output-presence counts, not recall or verified defects.** More output can mean more false foreground. An independent no-NN audit passes10,383 checks across every878 original frames and385 actual native masks, including RAW/gained hashes, exact timelines, per-frame receipts, packed mask hashes/foreground counts and complete MP4 streams. All385 masks are nonempty; that does not establish useful localization. [Independent audit](../output/evaluation/user-construction-videos-round4/after-public-cal-choice-cpu-recovery-v1/independent-native-evidence-audit.json).

### Qualitative failures at fixed and output-selected frames

- **Wall crack, fixed first frame0:** BEFORE traces much of the long crack; AFTER is broader over its left portion and misses much of the right portion. Both omit branches. [Actual frame](../output/evaluation/user-construction-videos-round4/after-public-cal-choice-cpu-recovery-v1/VID-20261020-WA0000/qa/after-source-frame-00000.png).
- **Fastening, fixed frame120:** BEFORE calls the member/wall boundary a crack; AFTER emits nothing. Neither verifies the owner-reported screw/anchorage installation issue. [Actual frame](../output/evaluation/user-construction-videos-round4/after-public-cal-choice-cpu-recovery-v1/VID-20261020-WA0001/qa/after-source-frame-00120.png).
- **Humidity, fixed frame165:** neither crack/spall model emits a candidate. The new model has no humidity category. [Actual frame](../output/evaluation/user-construction-videos-round4/after-public-cal-choice-cpu-recovery-v1/VID-20261020-WA0002/qa/after-source-frame-00165.png). Its first output-positive frame85 is an explicitly output-selected diagnostic: the mask follows a vertical wall edge, not the visible lower-wall stain. This is not evidence of detecting moisture. [Frame85](../output/evaluation/user-construction-videos-round4/after-public-cal-choice-cpu-recovery-v1/VID-20261020-WA0002/qa/after-source-frame-00085.png).
- **Upper wall, fixed frame0:** both models emit nothing. The new model emits a candidate on only the final frame71; that output-selected diagnostic covers broad decorative cornice boundaries, rather than establishing a faithful thin-crack trace. [First frame](../output/evaluation/user-construction-videos-round4/after-public-cal-choice-cpu-recovery-v1/VID_20261020_075653953/qa/after-source-frame-00000.png) / [only positive frame71](../output/evaluation/user-construction-videos-round4/after-public-cal-choice-cpu-recovery-v1/VID_20261020_075653953/qa/after-source-frame-00071.png).

No threshold, crop or checkpoint was changed after inspecting these outputs. The [same-CPU public122 results](construction-round4-measured-results.md) and fixed55 native-pixel benchmark separately quantify the trained model’s poor performance. The owner’s four videos therefore provide useful failure evidence, rather than supporting a claim that all reported errors are detected.

## Final local AFTER dashboard delivery

All nine new public/private AFTER recordings are registered as **FAILED research diagnostics**, with nine matching original-input presets. Existing BEFORE cases and model defaults remain. The trained CPU candidate rejects missing/default/GPU/wrong-thread execution and verifies the actual CPU provider/session; six mocked routing tests passed. Public FFV1 lossless inputs and private originals are never replaced by annotated exports as model input. The requested2FPS diagnostic replay is not measured real-time capacity.

The [HTTP delivery receipt](../output/evaluation/construction-round4-cpu-recovery-v1/local-integration/localhost-nine-recording-delivery.json) verifies all nine report/video HEAD/range routes and original-file presets. Actual browser verification played the1400×1240 owner comparison, sought exact original frame0, preserved all five unknown metrics as **Not evaluated**, and loaded the original WA0000 path with the FAILED900 CPU model. The inspection station and camera remain stopped.

The serving environment also performed exactly one sourceframe0 full-profile call plus three stateless warmups. Persisted labels, boxes, rounded score and native mask SHA exactly match the earlier export-environment output, despite different Numpy versions. An initial Python tuple/list comparison error and the no-NN JSON normalization correction are both preserved; no result or model call was replaced. This single integration check is not full cross-environment accuracy validation.

Full final software suite: **496 passed,9 skipped,1 dependency deprecation warning in40.86s**. The Torch-only TRAIN loss fixtures separately passed in the training environment. [Final verification](../output/evaluation/construction-round4-cpu-recovery-v1/local-integration/final-software-and-integration-verification.json).

![Verified local AFTER video and unknown metrics](../output/evaluation/construction-round4-cpu-recovery-v1/local-integration/ui-after-cpu-original-frame0.png)
