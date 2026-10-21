# Independent combined real-low-light dataset audit

The approved combined dataset's metadata and actual label text pass the independent source/class/geometry/copy audit. **This is not a model accuracy result.** No image or mask bytes were scanned/decoded, no NN ran, no heldout labels were read and no active training source was modified. Source pixel identities/gain are compared through pinned acquisition/preparation manifests; the trainer separately verified actual sample bytes before optimizer work.

The executable [audit](C:/Users/IAM/Documents/hackathon/opencv/project1/scripts/audit_construction_combined_lowlight_metadata.py) writes an additive [JSON report](C:/Users/IAM/Documents/hackathon/opencv/project1/output/evaluation/construction-combined-lowlight-independent-metadata-audit.json) with zero errors and exact source receipt hashes. It validates frozen training/data/native-input/exposure protocols, original source acceptances and acquisition manifests, preparation-plan/order receipts and the unchanged shared exposure implementation. Original acceptance/acquisition histories remain untouched.

| Actual label/metadata count | TRAIN | CAL |
|---|---:|---:|
| Inputs |1,171|122|
| Positive inputs |1,151|112|
| Verified negative photos |20|10|
| Class0: surface material loss/spalling boxes |585|38|
| Class1: crack boxes |1,016|125|
| Recorded gain applied |296|32|

TRAIN consists of990 genuine camera exposures,161 positive photographic views from121 unique source photo IDs, and20 negative photographs. The positive photo count includes120 DACL views from80 original photographs plus41 LCSD stills. Neither LCSD nor DACL is represented as a recorded video. Camera frames within a capture remain correlated; physical project/site and pretrained-data independence remain unproven.

All880 historical TRAIN inputs and122 CAL inputs preserve their raw source-file/native BGR/prepared input/label SHA identities, gain and dimensions in the new native manifest. Their complete source exposure/GT metadata also matches after path-only remapping. Actual new label-file bytes match the pinned label SHA, so the copy check is not merely a label-count comparison. The heldout protocol is byte-identical and original evaluation paths remain preserved. Zero exact raw-source file hashes overlap TRAIN and CAL. No independently unused heldout pixels were opened for this check.

## Accepted additions

**Dec31:** exactly250 views and405 publisher crack boxes from IDs16/17/18/20/24/27/35/37/41/42/46/48/51. Every record is TRAIN-only, matches an accepted original camera PNG identity, original frame/PTS, source camera/annotation SHA and gain receipt, and receives actual low-light gain. The six unresolved similarity endpoints30/34/40/45/49/50 and quality-rejected52 are absent. The [separate acceptance](C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/evcivil-dec31-future/acceptance-v1.json) remains the authority for source inclusion; original publisher boxes do not establish structural severity or assure useful visibility in every exposure.

**LCSD:** exactly41 original low-light still photographs and92 component-derived crack boxes match the accepted source set. All41 original mask paths remain present, with source-mask hashes matching both acquisition and acceptance metadata; their recorded codes are0 background and2 crack, no ignore pixels. Static preparation code explicitly uses `mask==2`, recomputes every8-connected component with minimum one native pixel, and checks exact source boxes without inversion, morphology, merging or top-K filtering. This independent audit verifies those retained mask-code/component records and actual normalized labels **without independently re-decoding masks**.

LCSD original source image/native/gained identities and gain values match the source receipt, and component-box metadata is preserved exactly. All41 are explicitly marked original author `VAL/TEST` repurposed solely into this experiment's TRAIN. There is no remaining LCSD validation/test accuracy claim. The root's actual visual receipt is correctly bounded to three fixed source sheets plus one flagged near pair; it explicitly says all41 were **not** manually reviewed. Real stills were not made into a video or an artificial night-camera sequence.

## Label geometry and scope

Every actual normalized label is finite, within native image bounds and uses the retained map0=spalling/material loss,1=crack. Class-aware one-to-one geometry matching against the source exposure/component boxes passes, with maximum native-coordinate rounding error **0.000002532118 pixels**. The smallest component label width/height is **1×1 native pixels**; these source components were intentionally preserved, not silently removed to improve apparent detection metrics. That correct geometry does not promise a detector can learn or localize such tiny targets.

Publisher camera boxes, complete DACL polygon regions and LCSD mask-derived component boxes have different annotation scope. They must not be interchanged with pixel segmentation metrics or interpreted as complete negatives for all unannotated defects. The broader material-loss category retains its explicit render/masonry/paving scope and is not automatically confirmed concrete-cover spalling. Source rights remain ev-CIVIL CC BY4.0, author-declared LCSD Apache2.0, DACL CC BY-NC4.0 and existing model/Ultralytics lineage restrictions. Nothing in this audit establishes Saudi-code compliance, structural acceptance or production readiness.
