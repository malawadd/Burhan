# Construction round4: actual masks and camera-domain boxes

Status:64 new public TRAIN photographs acquired;1,296 mixed TRAIN samples prepared and source-pinned. The exact900-update CPU1 fit completed in1644.07seconds and actually sampled60 of64 new photographs. A preserved GPU parity failure prompted a versioned deterministic CPU recovery. All four complete122-frame CPU evaluations passed numerical, input and strict full-profile/native-mask parity, but every trained checkpoint failed the quality guards. Final900 localized7/38 spall boxes with51 unmatched spall candidates, versus0/38 for the author; thin hits fell from7/62 to2/62. No model is qualified or promoted. The fixed900 FAILED diagnostic completed all five public videos and55 real pixel-labelled photographs; the full results and visible failures are recorded below. The four new local recordings remain private diagnostic-only inputs, never uploaded or used for fitting or model selection.

The previous author nano-seg model produces an actual visible trace on the daytime crack129 recording (38/71 publisher boxes localized), but gets0/38 spalling boxes on the unchanged122-frame calibration set. Its genuine thin-crack challenge remains0/8. These failures motivate joint domain adaptation with real mask supervision, rather than another threshold change or a detection-only head.

## One candidate

Retain the entire author YOLOv8n-seg source state, including both semantic channels and32 mask prototypes. Author class order is **crack0, spalling1**. Reconstruction uses only the installed official nano-seg schema and strict417 tensor restoration. No third-party YAML reaches the network parser.

The candidate retains the existing static640 input, native low-light gain exactly once, RGB/255 letterbox, confidence strictly>.25 for each class, NMS.45 and existing native prototype-mask decoder. The existing adapter/source remains immutable. There is no confidence search or per-video policy.

The approved fixed fit uses fresh AdamW.0002, first4 layers and their normalization state frozen, CPU1/batch1, exactly900 optimizer updates and snapshots300/600/900. A fixed six-step source cycle is actual-mask crack, actual-mask spalling, actual-mask spalling, camera-box crack, camera-box spalling, verified two-target-negative. It started only after the source-pinned real TRAIN gradient/memory/equivalence probe passed. There is no wall-time cutoff or outcome-based schedule change.

## More real data

The [official CUBIT-InSeg repository](https://github.com/BenyunZhao/CUBIT-InSeg) links its [publisher Drive dataset](https://drive.google.com/drive/folders/1HqS0rMWgwsy70TbaZYFqwzgFxLL_hp_b). Its README declares the images, annotations and metadata CC BY4.0. These dataset terms do not separately clear the inherited author checkpoint, whose model license and original training overlap remain unresolved.

Only the publisher TRAIN folder is being acquired. Its images.zip is15,034,071,180 bytes and the matching labels archive contains5,596 paired source photographs and YOLO instance-polygon annotations. Exact HTTP206 byte ranges allow selected ZIP members to be retrieved without downloading the14GB archive. ZIP CRC32, source compressed length, uncompressed length and SHA256 are retained. No Google account, paid service or remote inference is used.

The first128 label candidates were sorted by SHA256 of a fixed salt and archive member name before reading their annotations. The first32 positives per annotation class were selected without looking at image pixels or model outputs. Exact64-image selection was frozen before those images were downloaded (SHA256`9ab8bed2fcf0bc0197a6d3774d31b58e4ad1ad2035b1c487aa5d4c1d074ab5b7`). Actual total acquisition was176,625,588 bytes, under the250,000,000-byte bound including remote ZIP metadata and labels. `PAUSE` in the dataset root cooperatively stops before the next range request; existing completed files are preserved.

All64 original-file hashes are unique; all source headers have orientation1, all polygons have finite normalized coordinates, and the selected photos contain642 actual publisher instance polygons. File-prefix groups are CR1(32), SP0(22) and SP1(10), which does not establish independent scenes. The [source integrity report](../datasets/construction/round4-pixel-domain/cubit-inseg/qa/integrity-and-fixed-sheet-report.json) records six deterministic first/middle/last RAW/full/native-closeup/source-polygon/640-coverage sheets. The parent and a second agent inspected all six: the three crack photos show real thin branching wall cracks, the first two spall photos show clear material loss, and SP1_246 has a coarse source outline including a dark weathered edge beside loss. This is a bounded six-photo alignment review, not certification of every polygon or pixel-perfect/physical-width ground truth. The acceptance receipt is SHA256`d13fc153a2cb159c2d736dfb8582a0f7a8f41b247bbf0042a591d3524a8e887b`.

All newly acquired photographs are TRAIN-only. The existing complete122 calibration identities, source groups, boxes, negatives and timestamp convention remain unchanged. Existing camera heldout groups and exposed videos stay out of fitting. The new private videos stay out of both fitting and calibration. CUBIT physical-site identifiers and inherited checkpoint overlap are unknown, so a future result cannot establish independent-project generalization.

Existing actual source polygons in the80 official-TRAIN DACL photographs can add mask supervision without downloading those photographs again. Their [publisher semantic-annotation definition](https://arxiv.org/html/2309.00460v1) is multi-label and partly coarse; same-class semantic regions are not claimed to be native independent instances. Complete source polygons remain intact and are rasterized per recorded region. DACL's CC BY-NC4.0 terms apply even though the accessible mirror card says BY4.0. The55 previously examined DACL validation diagnostic photographs and all calibration/heldout negatives are excluded from fitting.

The completed source pool is128 CUBIT views from64 new photographs,158 DACL views from80 existing photographs,990 real camera exposures with publisher boxes and20 verified target-negative photographs. DACL train_2503 andtrain_6389 were already640Ã—480, so their full/native-closeup views would be identical; the declared view generator retained each once. This explains158 rather than160, without excluding a source photograph or a failed target crop. The complete TRAIN manifest is SHA256`39c86d5d82ca9fdfc38f7d0b06cce8505edd755a1df5282f6a300ebb4ca190f9`. The fixed900-step balanced sampler uses this pool rather than claiming a complete pass through every view; actual step identities and unique photo/view coverage will be reported.

For genuinely thin strokes, a separately recorded TRAIN-only native crop accompanies each full photograph. Its native window is capped640Ã—480, anchored at the real boundary vertex nearest the most extensive target's bbox center. It uses true source pixels, includes all clipped visible source polygons and shares the original photo identity. It is not a new photograph or a synthetic video. No calibration or diagnostic video receives a GT-centered crop. Source masks retain640 input resolution (`mask_ratio1`); the actual160 prototypes are upsampled by the training loss. Source mask pixels are area-resampled as fractional coverage rather than expanding narrow annotations into invented thick lines. Coverage>0 and coverageâ‰¥.5 are both recorded: numeric nonzero alone does not establish a recoverable positive thin stroke. The bounded TRAIN probe must establish acceptable memory use for the higher-resolution true-mask loss.

## Mixed supervision contract

The separately versioned `scripts/construction_round4_mixed_loss_v3.py` criterion has passed independent source review and the actual TRAIN gradient/equivalence checks recorded below. It imports the installed segmentation/detection losses without modifying their sources; the original scalar versions remain preserved.

| Source | Detection target | Pixel target | Class absence |
|---|---|---|---|
| CUBIT-InSeg TRAIN | Boxes derived from actual source polygons | Actual source instance polygons | Photographed annotated family only; other class unknown |
| DACL official TRAIN | Actual clipped semantic-region boxes | Actual source multi-label polygons | Both targets from complete published labels; coarse annotation limits retained |
| ev-CIVIL TRAIN camera exposures | Publisher boxes, exact protected timestamp association | **None** | Assigned positive GT-class anchors only; all unassigned anchors unknown |
| Verified target-negative DACL TRAIN | No crack/spalling targets | None | Both targets absent from complete published labels |

For camera-box batches, segmentation loss is disabled entirely. Neither rectangles nor empty target masks are manufactured. The prototype and coefficient branches receive a graph-connected zero loss, while the detector still receives genuine box/DFL supervision. Camera classification BCE applies **only to assigned positive GT-class anchors**. Unassigned anchors of the same class and all unannotated other-class anchors remain unknown, rather than being labelled healthy. Complete DACL annotations and verified target negatives supply background learning. For CUBIT actual-mask photos, background BCE is restricted to the photographed annotated family; this assumes that family's source instance annotations cover its visible targets, not that the other family is absent. LCSD crack-only images are excluded from this first joint candidate to avoid treating missing spalling annotations as healthy background.

Zero box-only mask-loss gradient does not mean mask weights stay identical during a mixed AdamW run: momentum and decoupled weight decay from prior actual-mask steps can still update them. The probe checks gradient semantics, not a stronger weight-immutability claim.

The box-only forward can also update trainable mask-branch normalization statistics; only the frozen first4 layers and their normalization state must remain identical. This is recorded domain adaptation, not pixel-mask supervision of the camera footage.

The default uniform mask BCE can let background dominate a1px stroke inside a large curved-crack box. This candidate instead declares one fixed actual-mask loss: half foreground/background-mass-balanced BCE plus half soft Dice, restricted to the actual source instance's box. Positive and negative source-pixel mass each receive half the BCE weight when both exist. Actual fractional coverage is retained; no dilation, rectangle fill or minimum-thickness target is introduced. The implementation calculates each matched true-mask ROI rather than all full640 foreground-anchor planes to bound memory. This is a training-only GT crop of the loss tensor; model input always remains the complete selected TRAIN view. The source class/box assignment and gains remain the installed detector's implementation. No loss-weight search is proposed.

Before a full fit, bounded real TRAIN checks must demonstrate finite forward/backward state, detector gradients on actual boxes, zero mask-head gradients on box-only data, nonzero mask-head gradients on genuine masks, ignored-anchor BCE gradient exactly zero, retained source-head transfer and frozen-layer/BN identity. The probe takes the first source-order mask-crack/mask-spalling/box-crack/box-spalling records and the largest actual-polygon-count prepared TRAIN mask view if distinct (at most5 backwards), each reset to the original source, CPU1 and peak-process RSSâ‰¤3GiB. These are numerical implementation checks, not accuracy measurements.

Five small no-network tensor fixtures passed before the real model probe: unknown-class BCE gradients are exactly0; unassigned same-class camera anchors are ignored; a1px stroke retains meaningful positive gradient without changing its target; fractional coverage is not dropped; and the memory-bounded actual-GT anchor-view implementation matches the parent assignment/gather objective and its prototype/coefficient gradients. They are unit fixtures, not generated inspection images or simulated video. Two agents independently reviewed the source math, class mapping, geometry and retained-state contract without running a network. The actual TRAIN probe remains necessary.

## Actual numerical checks and preserved recovery

The first implementation passed the first real crack case but rejected the second source spall mask: OpenCV area resizing had produced2,912 target values of exactly1.0000001192092896 (one float32 ULP above1). That failed attempt and its source/protocol remain intact. The separate v2 criterion rejects genuine nonfinite/out-of-range targets and permits onlyÂ±1e-6 numerical noise before locally clamping probability coverage to[0,1]. Source NPZ files, strokes and labels are unchanged. Dedicated numerical regressions confirm the one-ULP correction and rejection of1.001.

All five v2 real TRAIN cases passed, independently restored from the original source and discarded without optimizer updates:

| Fixed case | Actual GT assigned | Forward/backward seconds | Mask output gradient L1 |
|---|---:|---:|---:|
| New CR1_1132 full photograph |16/17 |24.17 |20.0431 |
| New SP0_1547 full photograph |2/2 |5.33 |7.7225 |
| Camera crack118 frame23, boxes only |1/1 |1.47 |0 |
| Camera crack122 frame155, mixed boxes only |2/2 |1.37 |0 |
| Largest polygon-count source CR1_2981,46 GT |44/46 |40.68 |19.8132 |

All gradients/state were finite; actual box gradients were nonzero, ignored-anchor classification gradients were exactly0, mask/prototype/cv4 gradients were exactly0 on box-only cases, and frozen first4 parameters/normalization state were unchanged. Peak process RSS was949MiB. These are contended CPU1 numerical/training costs, not inspection FPS. Surviving true masks do not guarantee every target receives a positive detector assignment, as the explicit16/17 and44/46 counts show. The receipt is SHA256`1b5b68b30909fabf19f2fca17ca0a0d4f11c6742dff5f9a24a3fe44b59fcc132`.

The mask-case cost motivated one exact-objective implementation optimization, not a loss or model search. V3 batches anchors sharing the same true GT/ROI and removes repeated full-prototype zero/slice-gradient work. Eight small tensor fixtures pass, including scalar/grouped objective and complete gradient comparisons within declared float32 tolerances. The actual five-case check also passed: it compared unchanged v2 loss and all four gradient arrays on detached actual network outputs, with one real TRAIN forward per case and trusted constructor initialization separate. Reference arithmetic time is kept separate from the grouped primary timing. V3 proposal SHA256`6e06b298e19c13e90e55e4f7ed9e76e6fe045d289c569ace26782ce78def8c7b`.

| Same fixed case | V2 seconds | Grouped V3 seconds |
|---|---:|---:|
| CR1_1132,149 assigned anchors |24.17 |4.285 |
| SP0_1547,20 assigned anchors |5.33 |3.045 |
| Camera crack118,10 assigned anchors |1.47 |1.327 |
| Camera crack122,20 assigned anchors |1.37 |1.254 |
| CR1_2981,272 assigned anchors |40.68 |7.428 |

All assignments, zero ignored-anchor gradients, true-mask/box-only gradient semantics and frozen first4/BN state remained correct. Maximum scalar/grouped loss difference was9.54e-7; maximum mask-coefficient gradient difference was1.33e-8. Peak process RSS including detached reference arithmetic was808MiB. No optimizer updates were saved. These are contended CPU1 training checks, not inference speed or accuracy measurements. The completed V3 probe receipt is SHA256`1a3fae13c54772e0405a64370d593a502c12c7e9b60c0e5b90bb50f7f301b915`.

## Evaluation before any integration

The exact900-update V3 CPU1 fit started at2026-10-20T17:40:10.995098+00:00 after all five real probes passed. The probe-only approval bytes were archived before stage authorization changed. Final approved-plan SHA256`185bbea5c1656feddf10de5918c27c11e0f496ca686409260b64fd59d9d7e814` binds the unchanged source/data/probe. All417 original tensors were restored exactly, with a fresh optimizer; no probe-updated state was reused. Actual training identities and unique photograph/view coverage are recorded, so the64 newly prepared photographs are not automatically reported as64 photographs actually sampled. Training overlaps another CPU1 factory fit and ordinary host activity. GPU export/calibration execution remains separately scheduled; no speed claim follows from these training times.

It completed all900 updates at2026-10-20T18:07:35.067985+00:00, with1644.07seconds total observed wall time (27.40minutes). All state/gradients remained finite, and frozen first4 state remained identical. The actual sampler used502 distinct prepared views:60 new InSeg photographs,75 existing positive DACL photographs and20 target-negative photographs, with repeated/cropped views sharing source identity. The source-step counts are150 mask-crack,300 mask-spalling,150 box-crack,150 box-spalling and150 negatives. The final step trace SHA256 is`dd55ebbba4f6a03b243e2b80d1ca31387e0cc82e892e6746a72678d2bb1517f9`. These training counts and numerical checks establish execution, not defect accuracy or inference speed.

All three fixed checkpoints are evaluated on the same complete122 calibration frames at exactly.25/.45. Every actual native mask is saved, including empty masks. Full serving-profile boxes, scores, masks and gain are checked against direct inference. Numeric Torch/ORTCPU/actualGPU guards precede calibration. No threshold rescue, source replacement, heldout opening or private-video fitting is permitted.

The first300 evaluation is preserved as a failed first attempt. Its deterministic Torch/ORTCPU/GPU fixture passed with GPU coordinate error0.00388 input pixels and class-score error4.91e-8. On calibration index21 (crack130 source frame30), both direct and full-profile calls returned four crack candidates, but scores differed by up to0.003255, one box changed from[112,107,190,131] to[111,107,194,131], and three native-mask hashes differed. Saved-output-only decoding reproduced all four cached signatures, so that saved cache was not overwritten by the later call. The second call's raw tensors/mask bits were not persisted, so a pixel-XOR count or cause cannot be recovered from that failure. No tolerance was relaxed and no calibration was retried. Later GPU evaluations were stopped; the already fixed900-step fit completed unchanged.

The separately approved six-call technical diagnostic completed without GT scoring or model changes. It used two independently compiled copies of the same300 model, three stateless initialization calls per copy, then alternating direct/profile calls A/B/A/B/A/B. All six actual NN input tensors had exactly the same hash, with unchanged RAW/gain and queried GPU/f32/one-stream configuration. Calls0/2/4 on A and later calls3/5 on B were identical. B's first source call differed before CPU mask decoding: maximum raw-coordinate difference2.08673 input pixels, class-score difference0.002843 and coefficient difference0.018733; prototypes were identical. Only two native pixels differed in the fourth score-ranked mask. A single CPUORT reference agreed with the other five GPU calls to0.00809 input pixels and2.15e-6 class score. This demonstrates within-instance raw-output variation for this bounded case, without identifying a specific driver/kernel cause. The technical report SHA256 is`b2eaad9aec4f2b647f0b5e6894362e1839b971a7a73d4ed38a38fb18fa74f97a`.

The transparent recovery retains all three exact weights, classes, sources, masks and.25/.45 cutoffs. A separately versioned CPUORT/one-thread protocol evaluated one original-author CPU122 baseline, then all three trained snapshots, preserving the same strict input/full-profile parity and numerical limits. No fitting, threshold rescue or retrospective GPU-failure rewrite is permitted. The same CPU global choice, or exact900 FAILED fallback, will drive all nine video diagnostics and the55-photo audit. Both55-photo models run on CPU; cached video BEFORE outputs remain historical GPU, so video comparisons explicitly combine a backend change with a weight change. Construction CPU speed remains unmeasured and may be slower; it earns no15FPS claim. The factory's15FPS qualification is separate.

CPU recovery protocol SHA256`2df501f9061578ce57f51187c88ff98ba1f550f1c4ab397a7a5ba302a109f927` was approved before CPU evaluation. The original-author sameCPU baseline completed all122 and reproduced the prior box counts exactly. All actual prepared inputs, strict full-profile/native-mask signatures and numerical guards passed. All three trained snapshots also passed the technical guards but failed every quality guard:

| Same CPU122 scope | Crack TP/FN/unmatched | Crack P/R | Spall TP/FN/unmatched | Spall P/R | Geometric-thin hits | Negative photos flagged |
|---|---|---|---|---|---:|---:|
| Original author |12/113/81 |12.90%/9.60% |0/38/0 |NA/0% |7/62 |5/10 |
| Fixed300 |10/115/133 |6.99%/8.00% |0/38/9 |0%/0% |2/62 |4/10 |
| Fixed600 |10/115/146 |6.41%/8.00% |1/37/35 |2.78%/2.63% |2/62 |5/10 |
| Fixed900 |14/111/118 |10.61%/11.20% |7/31/51 |12.07%/18.42% |2/62 |6/10 |

These are regressions or very weak results, not a fix. All predicted masks were nonempty: baseline93, snapshot300152, snapshot600192 and snapshot900190. Nonempty masks establish output presence only, not correct tracing. Global selection recomputed every guard and rank from the complete reports and returned no winner (freeze SHA256`4a010b0dae18c0dcd3d4ea539553d74daf8d8ac8b51a3936304890e476711910`). Exact900 remains the FAILED diagnostic for every subsequent video and pixel audit. No policy or weights were changed from these results. Camera metrics remain class-aware boxIoU.50 with correlated frames and no pixel GT. All unmatched/wrong-class candidates are included.

Checkpoint selection must be declared before any results: require class precisionâ‰¥80%, each class recallâ‰¥50%, geometric-thin recallâ‰¥50%, negative-image flagsâ‰¤10%, numeric and full-serving parity. These are minimum experiment guards, **not production acceptance or proof that the user's failures are fixed**. Retain all rejected checkpoints and results. If none pass, preserve the failure and do not promote a model.

Whole-video comparison retains the four already fixed development sources89/129/75/night21 plus explicitly post-hoc source76. It keeps all genuine source exposures/PTS and separate RAW, actual gain input, external publisher GT and actual predicted trace. Source76 earns no independent credit. New local user videos can provide an unscored before/after diagnostic once both model choices are frozen; without manual ground truth they do not establish recall or false-alarm rates.

No masks are treated as verified pixels merely because a predicted box overlaps a publisher box. The camera dataset has boxes, so strict boxIoU.50 and visible trace review are separate evidence. No claim is made about crack width, hidden moisture, structural cause, installation adequacy or Saudi code acceptance from these images.

After the global checkpoint choice is frozen, a separately predeclared diagnostic compares the original author and chosen candidate on all55 existing DACL official-validation photographs with actual semantic masks. If no candidate passes the calibration minima, the exactfinal900 remains a failed diagnostic. The55 results cannot select or tune a model. This exposed diagnostic can report native pixel precision/recall/IoU, with coarse-annotation, physical-site and inherited-training-overlap limits. Plan SHA256`184e5f534e37975fadd853cb4a41d96553c093069e0f619868defd90343f5e59`.


The five public-video execution protocol was frozen after global choice (SHA256`dd3dd439c502d3542b9c09754fdfdae93872404fb59bd6b0a8fcc69cce94f2ad`); the same55-photo CPU pixel protocol is SHA256`693c862c07d839ee88b411b9f2aa7436fe4a36192c745d03112b9a012f7b2424`. Both first executions completed with explicit parent release, overlapping private-video CPU inference and the factory CPU fit. These are contended accuracy/media jobs, not inference-speed benchmarks. Historical public-video BEFORE remains GPU; AFTER is CPU and trained weights, so their comparison does not isolate training alone. The sameCPU author122 baseline and bothCPU55-photo comparison provide the declared backend controls.


## Completed native pixel audit

All55 fixed previously exposed DACL validation photographs were evaluated on each CPU model, with no model selection from these outcomes. Report SHA256`6c3318968606e20d75fa341645670f80a20ddcc9b605bebcc021da878c26f763` binds110 actual photo inferences, native source/predicted-mask bits and complete records. The same source polygons are unioned by class; Crack and ACrack form crack, and overlaps between crack/spall remain allowed. Background true negatives do not inflate the metrics.

| Native semantic pixels, same55 | Original author CPU precision / recall / IoU | Exact900 FAILED CPU precision / recall / IoU |
|---|---|---|
| Crack |49.09% /2.93% /2.85% |33.39% /5.35% /4.84% |
| Spalling/material loss |85.31% /0.2356% /0.2356% |2.78% /0.2300% /0.2129% |

Crack foreground coverage rises slightly while precision falls; absolute recall remains very low. Spall foreground precision collapses and recall does not improve. These results do not establish a working crack or material-loss tracer, despite the camera box-match increase. Source annotations are coarse in places and inherited author-training overlap/physical-site identity remain unknown, so these are exposed diagnostics, not independent validation. Native mask presence on GT-absent photos was6/34 to4/34 for crack and0/19 to1/19 for spall; these are mask-presence flags, not detector-box false-alarm rates. Every source and predicted mask is saved without post-hoc filtering.

The55-photo records contain15 actual instance predictions for each model, with0 empty native masks in either model. The low semantic recall therefore comes from sparse or inaccurate foreground coverage, rather than silently discarded empty masks. Instance-pixel sums are not substituted for class-union metrics.


## Fixed video visual checks while complete exports finish

The first publisher-annotated preview is fixed by source order, not selected for a favorable outcome. Crack129 frame1 (0.05s) shows the new native mask following the real horizontal crack, but substantially thicker than the apparent stroke; this earns neither physical-width nor pixel-accuracy credit. The complete crack129 result is worse than the author:12/71 box matches and43 unmatched versus38/71 and3 unmatched. Spall75 frame0 shows the small visible wall chip with no output from either model, and complete AFTER coverage remains0/77. Night21 frame0 displays broad AFTER foreground over surrounding texture while both publisher boxes are missed;53 later box matches in the correlated sequence cannot be described as faithful thin-crack segmentation. The actual gain8 input is shown separately from the unmodified dark camera image, and all false foreground remains visible.


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
