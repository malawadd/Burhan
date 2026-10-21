# Construction Round4: actual CPU before/after validation

All models use the identical122 public calibration exposures, fixed0.25 confidence/0.45 NMS/native low-light gain, actual masks, CPUExecutionProvider and one inference thread. These are exposed, correlated development observations, not independent sites. Matches use class-specific publisher box IoUâ‰¥.50. Native pixel quality is evaluated separately on55 real polygon-labelled photographs.

| Model | Crack TP/FN/FP | Crack precision / recall | Spall TP/FN/FP | Spall precision / recall | Thin crack matches | Negative frames flagged |
|---|---|---|---|---|---:|---:|
|author-baseline|12/113/81|12.90% / 9.60%|0/38/0|NA / 0.00%|7/62|5/10|
|step0300|10/115/133|6.99% / 8.00%|0/38/9|0.00% / 0.00%|2/62|4/10|
|step0600|10/115/146|6.41% / 8.00%|1/37/35|2.78% / 2.63%|2/62|5/10|
|step0900|14/111/118|10.61% / 11.20%|7/31/51|12.07% / 18.42%|2/62|6/10|

**All three trained checkpoints fail every quality guard.** Numerical/input/direct-versus-serving parity passes. All actual predicted masks in these four122-frame runs are nonempty, but nonempty masks do not establish correct localization. The unchanged global rule chooses no eligible winner; exact900 is used only as a FAILED diagnostic for all subsequent videos/photos. No threshold adjustment, favorable-checkpoint fallback or production promotion follows.

Compared with the same-CPU author baseline, final900 raises crack box matches12â†’14 and spall0â†’7, while crack false alerts81â†’118, spall0â†’51, negative flags5â†’6 and thin matches7â†’2. This is not an overall improvement.

## GPU inconsistency and transparent backend recovery

The first step300 GPU calibration failed on source21 before completion. A separately declared six-call numerical diagnosis used exactly one identical actual input tensor, unchanged RAW/gain bytes, two GPUf32/one-stream instances and six alternated direct/profile calls. Five outputs agreed; the first profile-instance source call differed by2.08673 input pixels in coordinates,0.002843 in class scores and0.018733 in mask coefficients. Prototypes were identical and two native mask pixels differed. Within the same profile instance, subsequent identical-input calls agreed with the other five outputs. The underlying driver/kernel cause remains unproven. This is raw-output nonrepeatability before decoding, not evidence of an input-route difference.

The CPU recovery keeps all four original model assets, Torch fixtures, data, thresholds, matching and quality rules unchanged. The author baseline and each checkpoint process all122 frames with two actual CPU forwards/frame to verify identical input tensors and complete-profile native-mask parity. Every first numeric and parity gate passes. The original GPU failure, outputs and six-call diagnostic remain saved. CPU results do not qualify GPU deployment or construction15FPS.

## Real video and native-pixel diagnostics

All five complete public recordings and four complete private recordings use the same exact900 FAILED model. Cached public/private BEFORE footage contains actual historical GPU author masks; AFTER uses CPU and new learned weights, so backend and training both changed. The55-photo pixel diagnostic runs BOTH original-author and trained models on CPU. Private videos have no complete frame GT and therefore no recall/precision/false-alarm metrics. No owner video is used for fitting, calibration, threshold/model selection or upload.

- [Exact CPU recovery proposal](../output/evaluation/construction-round4-cpu-recovery-v1/recovery-proposal-before-cpu-network.json)
- [All checkpoint outcomes and global no-winner freeze](../output/evaluation/construction-round4-cpu-recovery-v1/global-three-checkpoint-freeze.json)
- [Saved GPU parity diagnosis](../output/evaluation/construction-round4-mixed-v3/parity-technical-diagnostic-v1/technical-report.json)
- [Owner recordings and qualitative findings](new-construction-video-review-round4.md)

The reported incorrect screwing is a **fastener/anchorage installation issue**. No installed model verifies screw engagement, tightening, embedment or required fastening arrangement. Stained/wet-looking RGB regions do not measure wall humidity. Those limitations remain visible in the application; random crack/rust boxes are not credited as detecting these faults.


## Complete five-video results

Every genuine camera exposure is retained (683 total). These reused development sources are fixed before the new execution; spall76 was previously selected post-hoc and earns no independent credit. The cached author BEFORE is GPU and trained AFTER is CPU: this video comparison changes both backend and weights. Counts are TP / missed publisher boxes / unmatched predictions on annotated frames; unlabelled frames are unscored. No pixel GT exists for these videos.

| Whole source | Original exposures | Cached author GPU TP / FN / unmatched | Exact900 FAILED CPU TP / FN / unmatched | Actual AFTER masks / empty |
|---|---:|---|---|---:|
|crack_89|24|0 / 8 / 7|0 / 8 / 5|14 / 0|
|crack_129|104|38 / 33 / 3|12 / 59 / 43|69 / 0|
|spalling_75|84|0 / 77 / 29|0 / 77 / 17|17 / 0|
|ros_crack_21|222|0 / 442 / 177|53 / 389 / 607|663 / 0|
|spalling_76|249|0 / 207 / 1|47 / 160 / 231|329 / 0|

All1,092 actual AFTER instance masks are nonempty, and no box match has an empty mask. This does not validate their shape. There were683 actual AFTER source forwards, three initialization forwards and zero new BEFORE forwards. Whole-video source timestamps, RAW/gain/native mask hashes, publisher boxes and per-frame matches/misses/unmatched predictions remain in the JSON/NPZ evidence. First executions completed without technical retries. Accuracy/media elapsed times were contended with other CPU jobs and do not establish FPS.

Saved-output visual review distinguishes genuine traces from misleading box matches:

- Crack129 fixed first annotated frame1 at0.05s: AFTER follows the visible horizontal crack, but the foreground is considerably broader than the apparent stroke. Complete-source coverage regresses from38/71 to12/71 and unmatched predictions rise3 to43.
- Thin crack89 remains0/8; spall75 remains0/77. The fixed spall75 frame0 visibly contains a small chip with no output from either model.
- Night21 fixed frame0: AFTER broadly labels surrounding texture, misses both source boxes and adds spall candidates. Its53 later box matches coexist with607 unmatched predictions; this is not a faithful thin-crack tracer.
- Spall76 fixed first annotated frame45: broad foreground covers the window/ordinary wall and misses the source box. The first matched spall box, frame79 at3.950002s (IoU0.5084), has40,685 positive pixels out of89,960 native pixels. Its foreground spills across the window/nearby wall and misses substantial actual loss. The47 whole-video box hits must not be described as correct material-loss masks. An additional wrong-class crack mask also covers broad regions.

The fixed first-annotated previews remain unchanged. The additional frame79 JPEG is an explicitly selected first-box-match quality inspection after inference, not model selection or independent accuracy evidence. It comes from the saved encoded video, with no new NN call.

## Same55 real-photo semantic pixel results

All55 previously exposed official DACL validation photographs were evaluated once per CPU model (110 source forwards). Native ground truth uses actual publisher polygons, Crack+ACrack unioned as crack and Spalling as spall; classes may overlap. No source box is filled to invent mask GT. Micro metrics exclude background true negatives. Coarse source boundaries, unknown physical-site identity and inherited author-training overlap limit interpretation. No model was selected or tuned using these results.

| Native class | Original author CPU precision / recall / IoU | Exact900 FAILED CPU precision / recall / IoU |
|---|---|---|
|Crack|49.09% / 2.93% / 2.85%|33.39% / 5.35% / 4.84%|
|Spalling/material loss|85.31% / 0.2356% / 0.2356%|2.78% / 0.2300% / 0.2129%|

Both models produced15 actual instance predictions, with zero empty masks. Crack coverage rises slightly while precision falls, and absolute recall remains very low. Spall precision collapses without a recall gain. Native predicted-mask presence on GT-absent photos was6/34 to4/34 for crack and0/19 to1/19 for spall; these are not detector-box alarm rates. Every native source/predicted class-union mask is retained.

**The new model has not fixed thin cracks or reliable spall localization.** No checkpoint is promoted. Humidity measurement, hidden leaks, fastener adequacy, physical crack width and SBC acceptance remain unsupported by this binary RGB model.

- [All five actual public video reports](../output/evaluation/construction-round4-cpu-recovery-v1/videos/five-complete-video-summary.json)
- [Complete same55 native pixel report](../output/evaluation/construction-round4-cpu-recovery-v1/fixed55-pixel-diagnostic/paired55-semantic-pixel-report.json)
- [First matched spall box visual quality review](../output/construction-round4-cpu-recovery-v1-spalling_76-first-box-matched-spall-review.jpg)
