# Construction round 3: thin cracks, spalling and additional real damage classes

20 October 2026. This document records a new supervised experiment. Round 2 artifacts and benchmark videos remain intact.

## Concrete failures being addressed

The existing joint-aware crack model localizes 136/162 branching-crack frame instances, but **0/8 thin-crack instances** in the exposed `crack_89` recording. It has no spalling class: all **72 spalling annotations** in `spalling_35` are unsupported. These are separate problems. A crack box must never be renamed “spalling” to appear to fix the missing category.

The thin-crack annotation was inspected on original pixels. Its narrow vertical branch is visible inside the publisher box; the old model generally predicts a different, thicker diagonal branch or an enormous whole-frame region. Whole-instance class-specific box IoU .50 correctly rejects those predictions. The new experiment retains that criterion and the unchanged exposed annotations.

## New real supervised data and split

[ev-CIVIL / DTU](https://doi.org/10.11583/DTU.25924228), CC BY 4.0, provides actual DAVIS grayscale video exposures and timestamped crack/spalling boxes. New selected recordings were acquired from the original field ZIP with bounded HTTP member ranges. The selected member's uncompressed length, ZIP CRC and local SHA-256 are verified. No generated image, still-image movie or synthetic-labelled record is used.

The frozen protocol is `datasets/construction/evcivil-round3/protocol.json`; every selected exposure and original frame/timestamp is in `exposures.json`.

| Split | Capture groups and globally distinct defect IDs | Selected real exposures | Box instances | Thin-crack box instances |
|---|---|---:|---|---:|
| Fit | `dset_3`: cracks 118,119,120,122,123,125,126; spalling 54,55,56. `dset_4`: cracks 140–147; spalling 72 | 456 | 374 crack, 107 spalling | 31 |
| Calibration | `dset_6`: cracks 129,130; spalling 75,76,77 | 80 | 45 crack, 38 spalling | 23 |
| Fresh heldout | `dset_5`: cracks 196,197,198; spalling 70,71 | 80 sampled for the dataset; full original videos used in final comparison | 55 crack, 32 spalling in sampled exposures | 1 in sampled exposures |

“Thin” in this descriptive audit means a publisher crack box whose short side is at most 30 original pixels and long side at least twice its short side. It is not calibrated physical crack width. The term does not change training labels or the evaluation acceptance criterion.

Uniform sampling chooses at most 24 publisher-labelled exposures per fit recording and 16 per calibration/heldout recording, before model training. Source groups and global class/numeric defect IDs are separated. Known `crack_120`/`crack_121` duplicate data are avoided by excluding 121. Spalling 70/71 and cracks 129/130 are excluded from fit even when another capture folder contains those IDs. Pixel-identical sampled exposures across splits are rejected. Entire `dset_1`, `dset_2` and `day_time_challenge` groups containing the three exposed videos remain outside fit and calibration. The publisher's `at_eth/synth_spalling` records are excluded.

Capture-folder/defect-ID separation is not a verified independent bridge/project split. Original geographic or physical identities are not fully documented, and neighbouring defects may share a structure. Adjacent frames remain correlated; exposure count is not physical-defect or project count.

## Meaningful negative backgrounds

The separate [DACL10k author dataset](https://github.com/phiyodr/dacl10k-toolkit) supplies complete published damage polygons. Forty selected original photographs contain no `Crack`, `ACrack` or `Spalling` annotation: 20 fit, 10 inner calibration from official train and 10 fresh negatives from official validation. Other damage, such as rust or efflorescence, can exist; these are **two-class negatives**, not globally healthy buildings.

Original RGB images and complete source polygons remain under `datasets/construction/dacl10k/`. All imported negatives undergo the **same grayscale conversion** used for the video inputs, followed by a fixed central crop to the camera aspect ratio and resize to 346×260. This also matches input geometry/padding rather than allowing source aspect ratio to encode the class. No crop is selected using a model score; the complete original source has no target-class annotation. This prevents a simple gray-positive/RGB-negative shortcut. Original and derived hashes, source partitions and licenses are recorded in `evcivil-round3/background-import.json`. No unlabelled ev-CIVIL exposure is treated as a healthy negative.

The DACL author terms are **CC BY-NC 4.0**, even where the mirror incorrectly lists CC BY 4.0. That non-commercial restriction must remain in the trained model provenance. Physical bridge identities are absent from the selected mirror, so distinct image hashes do not establish bridge independence. Negative-image false alarm rate is reported separately; still images do not establish false alerts per hour of healthy video.

Preparation used one CPU thread while a separate factory training job was active, with parent coordination. The 616 video exposures were extracted in approximately 8.9 seconds; no neural inference occurred during preparation. Training wall time is not a hardware-isolated inference benchmark.

## Actual new model, not a label change

The small candidate is a **two-class YOLOv8n detection model**, 416-pixel letterbox input, with classes `[spalling, crack]`. It transfers compatible feature and localization weights from the previously pinned crack/joint checkpoint. Its crack class remains the trained crack channel. The former joint classification channel is **reset** before training and receives actual spalling supervision; it is not exposed as spalling without learning. Mask-only source head parameters are discarded because the new corpus supplies boxes rather than pixel masks.

The first ten backbone blocks are frozen; neck and detection head are trained with new crack/spalling boxes. Both fit and calibration contain thin crack boxes and spalling. Photometric brightness augmentation applies to both positive and negative real inputs; it is training augmentation, not generated video evidence. An explicit initialized trainer preserves transferred/reset parameters instead of silently rebuilding a random YAML model.

The bounded plan uses a one-epoch timing preflight, then at most 32 nominal epochs with early stopping after eight non-improving calibration epochs and a 12-minute training-time limit. Actual completed epochs and durations are reported from the training outputs. Only new calibration data can choose a checkpoint or confidence operating point. Exposed and fresh heldout predictions are not opened until the model and operating point are frozen.

For each class, the predeclared threshold grid is .10,.15,.20,.25,.35,.45,.55,.65,.75,.85. Select maximum recall subject to calibration precision ≥.80 and false positives on at most 1/10 genuine negative images; if infeasible, retain the highest-F1 operating point with an explicit failed gate. The same frozen class thresholds apply to every subsequent recording. No per-video threshold tuning, box enlargement or IoU relaxation is allowed.

## Additional damage model

The parallel research work located the DACL author's actual trained 19-channel SegFormer MiT-B1 checkpoint. Its adapter preserves **independent sigmoid channels**, rather than incorrectly forcing mutually exclusive softmax/argmax labels. It can therefore support overlapping visible conditions when its actual class-specific predictions are validated. Author structural-component classes are separate from damage candidates. Export, licenses, actual hashes, pixel-ground-truth checks and category availability are recorded by the accompanying multiclass research report; availability is not a passing inspection.

The public interface is `mode: damage` with an explicit registered `damage_variant`. The shared loader verifies the actual frozen asset hash, architecture, input dimensions, class order and capability IDs. Old PPE/crack profiles remain available. Unregistered or missing damage models fail explicitly. No inference path downloads models.

## Evaluation and evidence

The frozen new detector is compared with the **same existing joint-aware baseline** on all five new heldout recordings, ten real negative images and all three exposed recordings. The old detector's spalling annotations remain unsupported; a crack output cannot match spalling. New scores require class agreement and one-to-one box IoU ≥.50. Unlabelled exposures are unscored. Pixel-mask IoU cannot be claimed from ev-CIVIL box labels.

Every original challenge exposure is retained in the result JSON and timestamp-faithful annotated video, including misses and unscored frames. External publisher annotations and model predictions use separate dashed/solid colors. Per-class matches, misses, unmatched predictions, precision, recall and unsupported counts accompany aggregate scores. Model-only speed is separate from video decode, overlays, encoding and full runtime throughput. Lossless FFV1/NUT inputs preserve raw camera pixels and original microsecond PTS, while annotated H264 remains a visual output.

```powershell
.venv\Scripts\python.exe -X utf8 scripts\prepare_evcivil_round3.py --fetch
.venv-roads-export\Scripts\python.exe -X utf8 scripts\prepare_evcivil_round3.py --prepare
.venv\Scripts\python.exe -X utf8 scripts\import_construction_negative_round3.py
.venv-roads-export\Scripts\python.exe -X utf8 scripts\train_construction_damage_round3.py --epochs 1 --name evcivil-yolov8n-r3-preflight
# Run the bounded fit only after reviewing actual preflight timing and receiving a compute handoff.
.venv-roads-export\Scripts\python.exe -X utf8 scripts\train_construction_damage_round3.py --name evcivil-yolov8n-r3 --epochs 32 --max-minutes 12
.venv-roads-export\Scripts\python.exe -X utf8 scripts\evaluate_construction_round3.py --phase calibrate --model models/construction/evcivil-yolov8n-r3.onnx
# Do not run --phase compare: it loads protected heldout recordings.
# Use the explicitly whitelisted development-video command below instead.
```

Neural execution and timing windows are serialized on the laptop. Preparation and research can run in parallel. Real results are appended below only after the actual jobs complete; planned classes are not counted as functioning detections.

### Additive fresh low-light holdout, frozen before fitting

The initial five day recordings contain limited thin-box coverage. Before any new construction model inference or fit, all three genuine recordings from the publisher's separate `night_outdoor_jan_2` capture group were added as **heldout only**: `2_jan_crack_32`, `2_jan_crack_43`, `2_jan_crack_44`. They remain absent from train/calibration YAML and have distinct global numeric IDs from the selected fit/calibration examples. `night-holdout-protocol.json` records this additive plan; member provenance records CRCs and SHA-256. No model score selected these recordings.

The final comparison reports fresh **day**, fresh **night** and old exposed challenges separately, including the same geometry-defined thin-box subset with unchanged IoU .50 matching. Before unsupported spalling stays separate from actual model misses. `all_annotated_class_localization_coverage` can additionally credit unsupported annotations as zero system coverage, while the legacy `recall` field scores supported classes; these two denominators must not be silently compared. Best/worst review frames include separate raw pixels, publisher GT and model prediction panels, while every frame remains in the unfiltered aggregate.

### Pre-fit timestamp-integrity amendment

Before **any** new construction fit, calibration or model inference, the new preparation/evaluation helper was corrected to the publisher's established **first annotation at or after the camera timestamp**, with gap bounded by median frame period/2 + 1,000 microseconds. It must not use a past-nearest annotation or an arbitrary 10 ms cutoff. `scripts/construction_annotations.py` supplies the shared function. Original annotation timestamps remain in every expected box. The earlier pre-fit protocol/exposure metadata are archived. Only this script's own generated PNG/label files within its verified dataset root were regenerated; source H5/NPY assets and original challenge annotations were preserved. This was a correctness amendment before learning, not a metric-driven adjustment.

### Actual night fit/calibration amendment, also before any new fit

Additional genuine Dec27 night exposures `27_crack_1`, `27_crack_2`, `27_crack_4`, `27_crack_5` are assigned to **fit**, and separate Nov18 recordings `ros_crack_21`, `ros_crack_22` to **calibration**. The heldout Jan2 group remains separate and untouched. Global numeric crack identities are checked across groups; spalling identities are a separate category. These additions target a real low-light domain shift rather than relying solely on brightness augmentation. They add at most 96 fit and 32 calibration exposures, for **552 fit / 112 calibration / 80 sampled day-heldout exposures**, plus the 20/10/10 grayscale, geometry-matched DACL negatives. Exact final counts are determined by eligible publisher timestamps and are recorded after preparation. Full day/night recordings remain the final video tests. The amendment is frozen before model training or any heldout prediction is viewed.

### Actual prepared counts and preflight schedule

The final timestamp-qualified preparation contains **466 fit / 112 calibration / 80 sampled day-heldout video exposures**, plus **20 / 10 / 10** real grayscale, geometry-matched two-class-negative images. The four Dec27 night recordings contribute 3, 5, 0 and 2 qualifying camera exposures; distant publisher boxes are not carried forward to fabricate more training labels. The earlier 552-fit figure was an upper bound, not an achieved count. All 17 construction/geometry/timestamp fixture tests passed before the fit.

The one-epoch timing preflight uses the installed trainer default nominal batch size 64. Before the full fit, the nominal batch is fixed to 16 (physical CPU batch remains 4) to provide more optimizer updates within the measured CPU time budget. This schedule change is made before any heldout prediction; it does not change class mapping, data selection, thresholds or matching.

The frozen final preparation has 390 crack / 107 spalling fit boxes (33 geometry-thin crack boxes), 125 / 38 calibration boxes (62 geometry-thin), and 55 / 32 sampled day-heldout boxes (1 geometry-thin). Protocol SHA-256: `82557b1bbd4f43e9b138f5c58beb6118e3c7e9f8d369ce1f95414f183264d24f`; final negative-import manifest SHA-256: `f03351531828d6989e3c5a697531996395f5214cea9154de3079e125c6acdf57`.

Calibration caches raw network outputs and invokes the exact serving decoder for every threshold, including OpenCV NMS's strict score boundary and final cap. After selecting the class thresholds, actual `infer()` is rerun on calibration and its predictions/counts must match before any heldout prediction. The combined negative-image false-alarm rate is reported separately from the two individual class rates.

The initial 12-minute fit is a checkpoint rather than a user-imposed time budget. If calibration is still improving or clearly underfit, further supervised fitting may continue the locally learned checkpoint **before** opening any heldout predictions. A continuation retains the newly learned spalling classifier, uses a recorded fresh optimizer, and records its parent checkpoint/hash; it never resets the classifier to joint semantics or changes heldout splits.

### Coordinated training overlap amendment

After the isolated construction preflight, the coordinator allowed a bounded, fixed-step factory **CPU fit on one thread** to overlap construction's four-thread CPU fit. This does not change either dataset or use heldout predictions, but training wall time and completed work under a wall cap can be affected. Construction convergence/calibration, rather than the initial wall cap, determines whether more fitting is needed. All reported final GPU accuracy/capacity and full-runtime speed measurements are run after the competing fit exits; no overlapped training duration is presented as isolated model throughput.

The training scanner removed one duplicate box row in fit exposure `dset_4-crack_140-frame00064.png`; source annotations and the exposure manifest remain intact. There were no corrupt fit/calibration images. The isolated one-epoch preflight transferred 355 compatible tensors, completed in 169.1 seconds including validation, and changed the newly reset spalling classifier (parameter L2 change .14637). Its low one-epoch validation score is an underfit timing diagnostic, not an advertised working model. Preflight ONNX SHA-256: `0492a9d4a9c62d40f14eb21a2f0309cd28e7365223781c42533ce614ee921c8c`.

The installed trainer restricts `channels_last` to CUDA and ignores it on CPU; its source documents no CPU benefit and a Windows slowdown. The verified CPU training path therefore keeps its normal memory format.

### Actual first-fit localization diagnosis

The serving decoder's raw-output calibration and actual-runtime parity check passed. The selected crack cutoff was .65 and spalling .20. This does not rescue the model:

| Diagnostic scope | Crack TP / FN / FP | Spalling TP / FN / FP | Meaning |
|---|---:|---:|---|
| All fit samples, same selected cutoffs | 179 / 211 / 7 | 79 / 28 / 64 | Learned the fit appearances, but imperfectly |
| Unchanged calibration | 2 / 123 / 0 | 1 / 37 / 12 | Severe generalization failure; reject |
| Original joint-aware model on same calibration, .25 | 8 / 117 / 10 | 38 annotations unsupported | Baseline also struggles on this new domain |

Lowering crack confidence to .10 still yields only 10/125 matched calibration boxes with 31 unmatched predictions. The failure is therefore not just the selected threshold. Spalling fails the calibration precision gate; combined negative-image false-alarm rate is 1/10. These are fit/calibration diagnostics, not fresh heldout results.

Independent inspection of separate raw/GT/model panels confirms that the calibration crack and small spalling patch are real and correctly aligned. The broad fit spalling band is localized while the smaller calibration chip is missed, showing a scale/domain gap. The actual panels and full records are retained under `output/evaluation/construction-round3/evcivil-yolov8n-r3/fit-diagnostics/`; the original baseline has its own diagnostic folder. No heldout or exposed-challenge predictions were opened for this candidate.

### Frozen continuation plan

The next candidate retains the learned two-class head, unfreezes all except the first **four** early blocks, keeps the **416** input, physical batch4 / nominal batch4, **three CPU threads**, AdamW learning rate .0005, no fresh warmup, **12 fixed maximum epochs**, patience8 and no wall-time cap. It may overlap a fixed-step one-thread factory fit; training wall time is labelled contended and is not an inference benchmark. Best-checkpoint selection uses the exact unchanged calibration set. This explicit configuration is chosen from the fit/calibration failure and independent review, before any fresh heldout prediction.

Its versioned training YAML adds 80 authentic DACL official-train photographs with complete retained polygons (125 spalling and 66 crack-family boxes), plus timestamp-qualified exposures from five additional genuine Dec27 recordings, IDs6/7/8/13/15. Both supplements were acquired separately and excluded all existing source images/known exposed, calibration and heldout canonical IDs. All old calibration/test samples remain byte-for-byte unchanged. Source and derived assets remain separate; DACL crops are training photographs and are never turned into video evidence. `scripts/prepare_construction_continuation_round3.py` creates the separate manifest and records actual eligible night counts before fitting.

The 40 crack-cohort source photos also have one deterministic native closeup each, preserving actual physical stroke detail that a full-view downsample can erase. These are **40 additional training views of the same sources**, giving120 derived training views from80 unique photographs. Native crops are320×240–960×720, actual grayscale pixels resized to640×480, with all in-view target polygons retained. There is no evaluation zoom or photo-to-video conversion. Original80-view manifest SHA-256 `2dc962b359bba28ba8eb1871184c2c2c5d8aa841cf4540d393fccd150276c360` and separate40-closeup manifest `3d376743bb6a75525c316e193c5298628fb720286271754dca774e42cca48d18` remain intact.

Actual extra night timestamp-qualified exposure counts are2,3,4,0,1 for IDs6,7,8,13,15. Their native camera rates are only about0.5FPS; they are genuine low-light supervision, not high-rate demonstration videos. The continuation has **476 actual video fit exposures +120 photographic training views +20 true two-class negatives =616 fit images**, with112 calibration exposures+10 negatives unchanged. The frozen continuation protocol SHA-256 is `99afdeb6e315ecf133c8c1cbb53d4ceadd99131fee71b49dfea69daa0470a39d`; exposure manifest SHA-256 `798bbfc94172a8661daa0fdb5799a0a5d4914014f0e9e173a669da567d70178c`.

### Unsupported-frame scoring correction, before heldout inference

The initial baseline diagnostic counted five unmatched crack predictions on spalling-only frames as false alerts. Those frames do not supply supported crack ground truth and must remain **unscored**, matching the round2 contract. The original console/summary and per-frame scope-v1 records are retained in the baseline diagnostic folder. The corrected calibration result is8TP/117FN/10FP for crack;38 spalling annotations stay unsupported. This correction does not alter any new-model fit/calibration predictions, dataset or threshold.

The final paired report additionally compares each class on the **same class-annotated frames**, with the same ten true negative images in an explicit combined scope. Unsupported spalling contributes zero system coverage in the all-class denominator, rather than a fabricated legacy model miss. Aggregate precision across different supported/evaluated class scopes is not a direct before/after comparison. Unlabelled and unsupported-only video exposures display **UNSCORED**, with unverified candidates; they never display a misleading zero-miss claim.

```powershell
.venv-roads-export\Scripts\python.exe -X utf8 scripts\prepare_construction_continuation_round3.py --closeup-manifest datasets/construction/dacl10k/round3-crack-closeup-contingency/asset-manifest.json --closeup-sha 3d376743bb6a75525c316e193c5298628fb720286271754dca774e42cca48d18
.venv-roads-export\Scripts\python.exe -X utf8 scripts\train_construction_damage_round3.py --data-root datasets/construction/evcivil-round3-continuation --continue-from output/training/construction-round3/evcivil-yolov8n-r3/weights/best.pt --name evcivil-yolov8n-r3-adapted --epochs 12 --max-minutes 0 --freeze 4 --batch 4 --nominal-batch 4 --threads 3 --learning-rate 0.0005
```

Before model scoring, the full eight fresh day/night recordings were also converted to lossless FFV1/NUT and verified against **all874 raw camera frames**, with exact original microsecond PTS and OpenCV BGR pixel identity. Combined with the original258 verified frames, the planned complete video comparison contains **1,132 actual exposures**. This preparatory check executes no model and makes no accuracy claim. Source sequences are113/155/116 frames for cracks196/197/198,153/110 for spalling70/71, and17/138/72 for the three Jan2 night recordings. These complete source timelines remain fixed before any heldout prediction is opened.

## First bounded fit: retained failed candidate

The first fit completed **5 epochs**, 756.0s including validation (757.9s through local export), with some CPU training contention as disclosed above. The best checkpoint was epoch4; validation mAP50 across epochs was .01548, .01174, .01178, .03421, .02236. This is not an acceptable detector. It remains archived rather than advertised as a working thin-crack/spalling fix. ONNX SHA-256: `19137c9ff183d65f389c27abdcc8536a4b1aa79578054c55ce8d7b64b0824410`. The new spalling final classifier changed substantially (L2 norm5.077), so the poor result is not an unchanged/relabelled joint channel.

Manual raw-camera/GT checks confirm class mapping, input geometry and aligned source boxes. The publisher sometimes labels only one of several visible crack branches; unmatched predictions are mismatches to the declared annotations and are not automatically proof that the physical surface is sound. This incomplete-instance supervision and the gray/low-light domain shift are practical limitations. Actual fit-versus-calibration predictions are checked next, before choosing continuation settings or opening heldout predictions.

## Additional thin-crack semantic contingency, before heldout selection

While the fixed12-epoch adaptation runs, a separate author-trained crack segmentation checkpoint is prepared for **unchanged calibration only**. The primary source is [LiteCrackSeg](https://github.com/1Kaleb/LiteCrackSeg/tree/082d33111f1a3b9ac1d78e984b06fd3530bf3533), at the pinned revision shown in that link. Its actual author-linked [TUT checkpoint](https://drive.google.com/file/d/12BDNmnJYWvVhkGNI0m2zGQdJHBoGTOHs/view) was downloaded (12,189,876bytes), SHA-256 `c12fb1ce991e479a32e5a26d7648ce9d20a440b9af4fd0411d321aa6a2eed497`. It is a tensor-only state dictionary; export uses restricted `weights_only=True`, strict complete tensor loading and the pinned local architecture/configuration. The whole-model/repository license is **undeclared**. A MobileViT subcomponent license does not resolve whole-model commercial use or redistribution.

Static inspection of the actual author training and test pipelines found **RGB float0..255 input**, with no division by255 or ImageNet normalization. The apparent ImageNet constants are unused. The export preserves this actual path, rather than guessing from the encoder family. Its main output is an actual crack probability mask, not a spalling channel. MobileViT-XXS plus the author's dynamic morphology and decoder is tested first at384×384, stretching the complete frame without any ground-truth crop. The optional constructor change loads the local architecture JSON and then all trained TUT tensors; it does not replace the trained encoder with random or ImageNet-only features. DropPath is identity at evaluation, and unused profiling imports are omitted.

The protocol is frozen in `output/evaluation/construction-round3/litecrackseg-calibration/preregistered-protocol.json` **before any specialist forward pass**. It uses the exact original112 calibration exposures and10 true two-class negative photos. Native-size probability maps are bilinearly restored, thresholded strictly at the preregistered grid .10/.15/.20/.25/.35/.45/.55/.65/.75/.85, and converted to actual8-connected components with at least4 native pixels. There is no closing, dilation, fragment grouping, inflated box, evaluation zoom, or invented video. Component boxes are compared with publisher boxes at IoU.50 as an explicitly limited localization proxy. Predicted semantic masks are saved separately; ev-CIVIL has no pixel masks, so pixel IoU cannot be claimed. Thin recall uses the previously declared native box geometry; physical crack width remains unknown.

Selection requires precision≥.8 and negative-image FPR≤.1, then maximum crack localization recall. A best-F1 point if no threshold passes remains explicitly **failed**, rather than a released fix. Spalling-only exposures remain unsupported/unscored by this specialist. Old exposed recordings and all fresh heldout predictions stay unopened until the global construction selection is frozen. Accuracy-only inference may overlap CPU fitting; those durations are not laptop capacity or real-time benchmarks.

Reproduction scripts are `scripts/export_litecrackseg_contingency.py` and `scripts/evaluate_litecrackseg_calibration.py`. Export creates only a separate experimental asset and does not replace the running application model. Other bounded searches did not establish an immediately verified small, clearly licensed trained replacement: UltraFastCrackSeg's published speed is not this laptop's speed, and no released trained checkpoint was verified; DeepCrack's author weights use a larger architecture. These alternatives are not presented as measured improvements.

### Actual specialist calibration result: fails release gates

The strict742-tensor export succeeded:2,939,010 parameters,384 ONNX SHA-256 `20936464f5fb6ef32cf5d84a87c84616bf1d5461364fbfb466dd7c4846d04055`. On the fixed first actual calibration exposure, exported ONNXRuntimeCPU agrees with author TorchCPU: maximum absolute probability error7.78×10⁻⁶, mean8.66×10⁻⁹. This confirms the exported model is behaving like the actual author checkpoint on that input; it does not establish inspection accuracy.

All122 unchanged calibration images were evaluated locally. No threshold passes the precision/negative-image guards. BestF1 at.15 yields **6TP /119FN /111 unmatched components**, precision5.13%, recall4.80%, and **10/10 negative images flagged**. Native geometric-thin localization is4/62 (6.45%) under the same component-box IoU.50 proxy. It stays an unregistered failed contingency, with no speed or deployment claim. No exposed/heldout prediction was opened.

The actual masks explain why these numbers need their stated scope: the fixed first crack129 frame has a red mask visibly following the horizontal hairline, but its tight component box does not overlap the taller padded publisher box enough for IoU.50. The near-black fixed night exposure is missed. The first negative photograph includes shattered glass (not a DACL concrete-crack annotation) and the model also wrongly fills a ground region. Both the annotation/class scope and actual false positives matter; there is no justification to remove those negatives after seeing results. A visible trace on one development frame does not overcome the failed release guards.

Probability maps, selected native masks, all component records and five fixed separate raw/GT-box/model-mask panels are under `output/evaluation/construction-round3/litecrackseg-calibration/`. `render_litecrackseg_calibration_cache.py` reconstructs those panels without inference. Pixel metrics remain unavailable; predicted red pixels are never presented as publisher masks. The two actual-mask/component decoder fixtures pass. Accuracy runs overlapped the fixed CPU fit and provide no valid capacity benchmark.

### Separately frozen camera-exposure diagnostic

The first actual night calibration image has original grayscale range0–35, median5 and99th percentile12. A separately preregistered input policy applies a global BGR gain only when the original native grayscale99th percentile is below64: gain=`min(8,128/max(p99,1))`, followed by clipping and uint8 conversion. There is no generated spatial content, GT-dependent region or evaluation zoom. Bright inputs remain byte-identical and reuse their exact prior probability maps; changed low-light inputs are recomputed with the same installed checkpoint. All122 calibration images remain in the score, and original artifacts stay intact.

At the same best-F1 cutoff.15 this exposure candidate gives **21TP /104FN /275 unmatched components**, precision7.09%, recall16.8%; geometric-thin matches5/62 (8.06%). Negative-image FPR stays **10/10**, so no threshold passes release guards. It is not a successful production fix. The fixed first night mask remains a useful inspectable diagnostic, with actual RAW / GAINED MODEL INPUT / PUBLISHER BOXES / MODEL MASK shown separately. Results are retained under `litecrackseg-calibration/low-light-gain/`. No heldout or exposed inference was opened. Three decoder/exposure fixtures pass; cached exported masks now contain exactly the retained min-area/top100 serving components rather than discarded threshold speckles.

The original calibration protocol identity is `e740b80e9993236528fd040db1ab49bb04b8ea4eb2d47ebf06548f5dc89007ba`. Original model outputs and actual author-preprocessing parity remain unchanged. Both semantic candidates fail the same declared guards; the visual thin-trace nuance does not justify relaxing IoU or eliminating false alarms.

### Installed DACL512 camera transfer audit: also fails

The actual installed author-trained19-channel model was additionally tested with only its crack and spalling channels selected, on the **same complete122 calibration images**. Its separate protocol was frozen before these camera forwards, SHA-256 `2c08762298784bbdc7835c1da0e657fb447246e0e48e410c099aba4fae500d1c`. The existing512 input, RGB/ImageNet normalization, independent sigmoids, nearest projected components and4-grid-pixel minimum remain unchanged. The original14-category photo audit and default photo presets are preserved.

Neither class passes the same release guards. Crack best-F1 cutoff.35 gives **3TP /122FN /7 unmatched components** (P30.0%, R2.4%). Spalling cutoff.85 gives **0TP /38FN /6 unmatched components**. Combined negative-image FPR is1/10; geometric-thin matches1/62. This trained photographic model does not demonstrate a camera-domain spalling fix. Its previously measured slow photo throughput is not reused as a new video benchmark, and these accuracy forwards overlap CPU fitting.

All two-channel probability maps, exact retained native masks, records and fixed raw/GT-box/model-mask panels are retained in `output/evaluation/construction-round3/dacl512-camera-calibration/`. Per-class confidence is now an optional validated serving configuration; ordinary uniform presets retain their existing path, and an explicit uniform confidence overrides a specification's per-class defaults. Actual `DaclConstructionModel.infer` postprocessing on all122 identical cached numerical maps matches calibration labels/boxes/scores at the selected point. This is decoder parity, not a second neural accuracy or speed measurement. Eight focused DACL/Lite geometry/exposure/threshold tests pass. No exposed or heldout input was scored for this comparator, and no new video model was released.

### Original joint-aware model plus shared gain: also fails

A separate complete122-source calibration comparator retained the actual original joint/crack checkpoint, unchanged decoder and shared input gain. At the same.25 threshold, RAW gives8/125 localized cracks with10 unmatched boxes; GAIN still gives8/125 with16 unmatched boxes. Both flag3/10 negatives. No grid point passes the declared precision/negative guard; the gain fallback.10 gives13/125,42 unmatched, geometric-thin5/62 and4/10 negative alarms. Actual selected `infer()` parity passes all122 inputs. No constructor/default or shared preprocessing change follows this failed comparison.

Spalling remains unsupported (38 instances), with37 spall-only exposures unscored for this crack-only model. Its common crack comparison therefore uses85 crack-annotated/negative exposures. Own two-class full operating-point calibration still scores all122 and penalizes wrong-class predictions on the other defect class; separate common-class scopes keep paired tables comparable without concealing those full-system errors. Exact comparator artifacts are under `output/evaluation/construction-round3/joint-aware-shared-gain-calibration/`. This does not establish pixel-mask accuracy or a thin-crack fix.

## Actual adaptation completed: retained failure

The fixed maximum12-epoch continuation stopped after9 epochs under patience8, with best checkpoint at epoch1. It completed2390.41s fitting/validation and2392.79s through export, with disclosed overlap of other fixed CPU training and accuracy-only GPU work. It was never suspended; speed qualification is deferred to an idle laptop. The actual retained source checkpoint is `output/training/construction-round3/evcivil-yolov8n-r3/weights/best.pt`, SHA-256 `ac3072242f4d7890312ba227f3d394fc034d491f8dfe95f665ad6795802a7182`. This observational identity is retained in final provenance without rewriting the already-started plan, whose previous-candidate identity referred to its ONNX export.

Adapted ONNX SHA-256 is `62e2da13a7cf072140876f2d16fb63f46560ea394c317de581cbd7e0f02dda44`, input416, class0 spalling / class1 crack. Unchanged calibration freezes crack.10 and spalling.85: **10/125 crack instances localized with29 unmatched predictions;0/38 spalling localized, no spall predictions**. Combined negative-image false-alarm rate is3/10. Actual serving raw-output/threshold/global-cap parity passes. Both accuracy guards fail. This model is not a production fix and is not substituted for the original public video detector.

The frozen diagnostic-video selection uses the complete first training spalling54 sequence and the complete first calibration spalling75/crack129 sequences. Seen-training and calibration roles are explicit; all original camera exposures, misses and unmatched predictions remain. It contains no exposed or heldout forward and provides zero independent accuracy credit. The current model retains its actual.85 spalling cutoff even though the video then shows misses. A separate historical failed-model export uses its already frozen.20 spalling cutoff on those same recordings, including the same calibration failure controls; it is not a newly successful candidate or a threshold chosen to make a demonstration look better.

## Distinct next data version and bounded head fit

The future acquisition reviewed12 additional genuine spalling capture groups and121 flagged similarity pairs. Nine reviewed concrete-surface sequences (216 real selected exposures) are accepted for a **distinct** future fit: dset1 spalling11/6/16/8/9/13 and dset2 spalling43/46/40. Ambiguous spalling38 is excluded; spalling24/27 are separate render/masonry/paving material scope and excluded from the concrete-only fit. No exact duplicate or matching visible damage layout was confirmed in reviewed pairs, but physical-site independence and unreviewed frame similarity remain unknown.

`scripts/prepare_construction_spall_round4.py` prepares only these accepted/hash-verified images and labels in `datasets/construction/evcivil-round4/`; it starts no training. The result is832 training images:692 actual video exposures,120 photographic views from80 unique photos and20 true negatives. Original calibration/test paths and negative splits remain unchanged. Data-protocol SHA-256 `0b484de6a900716f1cac805c173c19d19dcdb9487a7b342d80338533c1cb6d9d`; combined exposure manifest `444898d49674abe21c94bd467304f9eaf8dfc68d5c9145b163a4f6dd6d880a06`. Acceptance review SHA-256 `4aaea999944cfc20495244b02f709ff210c26b51896b6be787dbebb9481c217a` links the immutable acquisition. Unlike the previous plan, dset1/2 can now contribute fitting data, so the old exposed recordings become development only and cannot provide independent generalization credit. Entire new capture groups remain training only; dset5/6 and heldout night exclusions remain intact.

### Higher-resolution export alone does not fix the failed model

A protected640 export used an isolated copy of the actual adapted best checkpoint; it did not overwrite the416 `best.onnx` or retrain weights. Frozen resolution protocol SHA-256 `d06cf73dc19f8e0eedac2e7209a97da40e876e31ecea9af83ae468969ab5fa4d`;640 ONNX SHA-256 `e1c9a3eb962cf36706e2059893904f48123f5793bfaeb945f9ff5a9e0072c399`. Actual unchanged122-image calibration and serving parity run locally.

| Same learned weights, raw camera policy | Crack TP / FN / unmatched | Spalling TP / FN / unmatched | Negative-image flag rate |
|---|---:|---:|---:|
|416, separately calibrated.10/.85|10 /115 /29|0 /38 /0|3/10|
|640, separately calibrated.10/.85|10 /115 /45|0 /38 /0|3/10|

Resolution alone supplies no accuracy fix; the next fit needs the actual additional source diversity. There is no speed claim from this accuracy run or a claimed640-trained model—the underlying weights were trained at416.

### Shared future camera-input policy and bounded training plan

`construction_preprocessing.camera_exposure` now implements the exact previously declared bounded global gain for both native preparation and future serving before letterbox. Its pinned source SHA-256 is `451b3f899fee6217066b425c664e3b3b9689cf81fd67dc1450eb5d9c90ad8707`. Existing installed models default to policy `off`; their inputs remain unchanged. `ConstructionModel.prepare_input` is also used by calibration raw-output caching, preventing a future gain model from being calibrated on raw-dark inputs while serving gained inputs.

`scripts/prepare_construction_gain_round4.py` creates separate lossless native input PNGs for all832 fitting and the same122 calibration source identities, without touching any raw source or publisher label. Five fitting and32 calibration inputs actually receive gain. Every decoded prepared PNG is verified pixel-identical to the shared function's native BGR output. The input manifest retains raw image paths/file hashes, raw and gained BGR hashes, applied gain, unchanged geometry and label hashes. Fixed first low-light train/cal raw-versus-input previews are saved before any new fit. The original heldout source path stays untouched, and no heldout model forward occurs.

The next model uses640 and this single declared camera policy. Its first stage is **four fixed head-focused epochs with freeze10**, preserving the first failed model's actually learned two-class head, rather than committing immediately to another long full-backbone fit. After the factory's final isolated measurements, a bounded TRAIN-only batch4/8/16 capacity probe measures at most four actual optimizer steps per trial, with identical checkpoint/seed resets and no saved training output. It selects safe training images/second within a6GiB process RSS limit; no calibration accuracy or heldout input influences batch choice. The final batch and complete fit configuration are frozen before training. Any later freeze4 continuation requires a distinct version and the same untouched evaluation source identities. There is no hidden wall-time stop or production claim from a short initial stage.

The capacity probe is now complete. Batch4/8/16 measured2.55/2.19/0.85 **training images/second**, with process peak working sets0.73/1.10/1.81GB. These CPU4 measurements can overlap the factory's fixed one-thread fit and are explicitly contended; they are not inference FPS and do not predict full augmented training time. The initial integration attempt stopped before any optimizer update because a loaded checkpoint's reduced inference arguments omitted loss weights; its console is preserved. Restoring the official training defaults allowed the unchanged frozen protocol to complete. No probe weights were saved and no calibration or heldout image selected the batch.

The resulting head-fit plan is frozen, SHA-256 `53c390e9f4fd23f24c2329d10d1296c4ddfaa8f3259ba0eedef71b150fb3a532`. Actual command uses batch4 / nominal batch4,640, freeze10, AdamW learning rate.0005, **three CPU threads alongside the factory's one thread**, four fixed epochs, no fresh warmup and no wall cap. The CPU3 fit versus CPU4 probe distinction is explicit. It begins from the first learned checkpoint SHA`ac3072242f4d7890312ba227f3d394fc034d491f8dfe95f665ad6795802a7182`; all learned class weights are retained. `train_construction_damage_round3.py` verifies that source identity, the frozen command configuration, shared exposure-policy source and every prepared image/label hash before fitting. The protected four-epoch fit has completed; its console is `output/training/construction-round4-head-console.log`. Its complete operating-point calibration fails the minimum accuracy guards; exact results are recorded below.

Before calibration of any gain model, the evaluator checks the prepared-data protocol identity and shared preprocessing source against that model's training provenance. It then loads the **original raw calibration images**, applies the same shared policy once and verifies selected threshold predictions against actual serving inference. Per-frame records retain the original source-frame index, actual gain and policy; the summary states how many images were enhanced. Diagnostic exports show the raw camera separately from the actual native enhanced input, before letterbox. This prevents double enhancement or presenting adjusted inputs as untouched camera evidence.

The independent static and TRAIN-label audit in `docs/construction-round4-independent-static-review.md` confirms the retained class mapping, once-only gain contract and native label geometry. The832 fitting images contain1051 boxes (532 spalling,519 crack), with20 negatives. All812 positive training records agree with normalized labels to at most2.54e-6 native pixels; exact raw-file train/cal overlap is zero. This is a data/code contract check, not neural accuracy or physical-site independence. Some boxes are only1.333 native pixels high, which describes annotation geometry rather than measured physical crack width.

An input-distribution limitation remains: the20 genuine night-labelled fitting views are bright enough that **none receives gain**. The five gain-applied TRAIN inputs instead comprise one photographic closeup and four spalling13 views; the32 gain-applied CAL inputs are actual dark night-camera views. The fixed first gained TRAIN preview and first gained CAL preview therefore do not establish equivalent low-light training coverage. This measured mismatch may affect transfer but does not prove the cause of any miss. The shared function, active fitting data and original calibration remain unchanged.

An additive continuation guard was recorded before any R4 operating-point search, without changing the active fit: each class and the unchanged geometric-thin crack subset must reach at least50% box-localization recall, each class must retain at least80% precision, combined negative-image alarms must be at most10%, and serving parity must pass. Failure triggers a distinctly frozen four-epoch freeze4 continuation after coordinated factory timing. **This is a minimum threshold for deciding whether more training is required, not evidence that50% recall solves the inspection problem.** All122 calibration sources remain; heldout inputs never decide continuation. The clarification is `datasets/construction/evcivil-round4-gain/stage2-continuation-guard.json` and explicitly records that ordinary epoch1 checkpoint-selection validation had already completed.

During epoch2 the consecutive batch104/105 console timers jump from3:43 to37:53, a2050-second walltime interval. A separate factory tool also reported a long interval at the same time. The same worker remained alive and resumed progressing; no training suspension, restart, work reduction or configuration change was requested. Subsequent read-only Windows System evidence confirms a coincident low-power interval: Power-Troubleshooter event1 records sleep04:55:16.7061573Z through wake05:29:31.3114540Z, a2054.6052967-second timestamp interval. `output/training/construction-round4-walltime-observation.json` links the hashed event record at `output/optimization/factory-round3/host-low-power-interval.json`. No power setting was changed. Total fitting walltime will include this interval without subtraction; it cannot be interpreted as isolated inference throughput.

### Broader-material follow-up prepared; no fit started

The serving task already uses generic **publisher-labelled surface material loss**, because a grayscale chip does not establish concrete-cover failure. The current head's nine-source supplement deliberately excluded render/masonry and paving. A distinct conditional proposal now admits the previously reviewed spalling24/27 material groups:24 real views each,53 original publisher boxes, with source-image/label hashes verified. Ambiguous spalling38 remains quarantined. This is explicit broader training-material coverage, with the original annotation class retained; it does not relabel predictions or change calibration ground truth.

The approved preparation now contains880 fitting images (740 real camera exposures,120 views from80 unique photographs,20 negatives),585 spalling boxes and519 crack boxes. All122 calibration source identities/GT and heldout boundaries remain. The same-data stage2 helper and current832-image head history are preserved. Exact proposed records and scope remain immutable in `output/evaluation/construction-round4-broad-stage2-proposal.json`, SHA-256 `7771c7d92452d70003149219cbf4cca848f95fa39df4144be6fced9ea0d035de`.

`scripts/prepare_construction_broad_stage2.py` copied the original954 train/cal native images and labels byte-for-byte into the distinct broader root, then prepared the48 additional actual source images through the same shared gain once. Each new decoded PNG equals the actual native model input; source-file hashes, original frame indices/timestamps, camera/annotation identities, exact gains and material roles remain. Original122 calibration sources and original832 head inputs are untouched. Data-protocol SHA-256 is `1da0533a85aa7f322a90e8ea87ae1f6b27a28b148cc04ca1523793d2cde1e935`, native-input manifest `a16f19d3217fe38dfeb596db212147846ba8019f0849bdf1f0b0dd7078e054d4`, exposure manifest `77b0ae1d141d425782482b88b2ebac9e99419d50b8413ed26b8480762e504f76`. No new training protocol or fit has started. Actual continuation remains conditional on the complete head calibration and coordinated timing release.

Before either conditional continuation, a separate four-trial TRAIN-only efficiency probe is prepared: NCHW/channels-last crossed with3/4 CPU threads, each reset to the same completed learned head and seed, batch4, freeze4 and at most four actual forward/backward/optimizer steps. Only the final three steps measure training images/second; no weights are saved and no calibration accuracy selects the format. `output/training/construction-round4-layout-capacity/plan.json` pins this design. Execution and any new fit wait for the parent's factoryD final timing release. The eventual distinct four-epoch continuation configuration will be frozen only after these measurements and source/input identities are known.

### Six explicit whole-recording development diagnostics

The first failed and adapted failed models both run on exactly the same fixed recordings, at their already frozen calibration cuts. Each preserves all real exposures and original PTS, independently checked source/lossless-replay identity, external publisher GT, actual model boxes, misses and unmatched predictions. All six JSON count/timeline results additionally passed a separate class-aware real-IoU audit (`docs/construction-recording-count-audit.md`).

| Failed candidate / complete source | Role | Crack TP / FN / unmatched | Material-loss TP / FN / unmatched |
|---|---|---:|---:|
|Historical / spalling54|Seen during fitting|0 /0 /5|65 /185 /88|
|Historical / spalling75|Calibration control|0 /0 /0|0 /77 /3|
|Historical / crack129|Calibration control|4 /67 /1|0 /0 /14|
|Adapted / spalling54|Seen during fitting|0 /0 /127|0 /250 /0|
|Adapted / spalling75|Calibration control|0 /0 /0|0 /77 /0|
|Adapted / crack129|Calibration control|17 /54 /16|0 /0 /0|

The historical seen-source clip visibly demonstrates actual surface-material-loss localization in some frames and material misses elsewhere. Substrate/coating identity is not certain from grayscale appearance, so this is **publisher-labelled surface loss**, not confirmed structural concrete-cover failure. Calibration failures remain alongside that seen-source success. None supplies independent accuracy credit, replaces a primary category reference, or becomes a production fix. Reviewable model/preset/recording blocks with exact artifact hashes are in `output/evaluation/construction-round3/diagnostic-integration-proposal.json`; the root integrates six diagnostic entries at most. Every-frame replay can be slower than original time and is not a15FPS qualification.

## R4 head completed: more spall matches, still fails every accuracy guard

All four fixed epochs completed. Fitting/validation took4193.90 seconds,4196.17 through export, retaining the documented2054.605-second low-power interval and CPU contention. Best checkpoint remains epoch1; later generic validation mAP does not establish a useful model. The actual frozen ONNX is `models/construction/evcivil-yolov8n-r4-head-640-gain.onnx`, SHA-256 `33e7ab167a93c7ee9c9e80e544951ec2e61fcdaf977377ea56e321dd4fea41ad`; retained best PT is `5e48033ee0122d39a3a6301aa926dad6bd18b141e89468185a66d1871a5eef6c`. Its actual executed source archive matches SHA `cb44fde015f46c68a3f2d4c9e20bc4a5d16f381a9aef418c8769a9bfbe5f7522`. No class reset occurred; the learned spalling classifier changes by L2.22263 and monitored frozen backbone blocks remain unchanged as expected.

The same complete122 original raw calibration inputs receive the shared gain exactly once. Actual inference at the jointly selected point matches the cached exact decoder on every input. Crack.45 and spalling.10 are the frozen fallback operating point, not successful production thresholds.

| R4 head calibration scope | TP | FN | Unmatched | Precision | Recall |
|---|---:|---:|---:|---:|---:|
|Crack, full122 two-class scope|5|120|6|45.45%|4.00%|
|Material loss, full122 two-class scope|4|34|28|12.50%|10.53%|
|Material loss, common47 spall-annotated/negative exposures|4|34|5|44.44%|10.53%|
|Geometric-thin crack subset|0|62|Not a separate FP scope|—|0.00%|

The full scope retains23 additional spall-class false alerts on crack exposures; the narrower paired scope never erases those errors. Combined negative-image alarms are3/10. Every accuracy/false-alarm minimum guard fails; only serving-decoder parity passes. Spalling improves from the first model's1/38 calibration matches to4/38, but remains unreliable and thin cracks remain unsolved. No heldout source is opened, no default model is promoted and no inference-speed qualification is inferred from these contended accuracy forwards. A distinctly frozen continuation is warranted after coordinated factory timing and any reviewed genuine low-light data version.

These125 crack and38 material-loss boxes are frame-level observations across the fixed calibration recordings, with repeated views of physical defects. They are not163 distinct physical defects or independent sites. The same applies to full-video frame counts in the development comparisons.

Five inspectable cached panels are retained under `output/evaluation/construction-round3/evcivil-yolov8n-r4-head-640-gain/review/`: fixed first crack, fixed first material loss and fixed first gained night input, alongside explicitly **posthoc calibration diagnostics** for the first material-loss match and most unmatched boxes. Each shows RAW / EXTERNAL GT / ACTUAL NATIVE INPUT / MODEL separately. Visual review confirms missed real horizontal cracks and the small wall chip, with incorrect boxes elsewhere. The spalling76 match visibly covers missing render exposing masonry, consistent with generic material-loss scope rather than a confirmed concrete-cover diagnosis. These panels carry zero independent accuracy credit; they do not substitute for real video demonstrations or conceal the misses.

## Distinct real low-light continuation: four fixed epochs completed

The failed head's unchanged122-source calibration triggers further work, rather than release. A separately accepted1171-view TRAIN dataset combines the preserved broad880 data with250 genuine dark camera exposures from13 accepted Dec31 sources and41 real low-light LCSD still photographs. It contains990 camera exposures,161 photographic views from121 unique photographs, and20 background photographs; labels contain585 material-loss and1016 crack boxes. The161 views include overlapping views of the same photos and are not161 independent photographs. Actual gain applies to296 TRAIN views versus32 CAL inputs. Quarantined Dec31 IDs30/34/40/45/49/50/52 and ambiguous spalling38 remain excluded. LCSD author test photographs are explicitly repurposed solely as TRAIN transfer, with no LCSD benchmark claim or invented video.

Native preparation verifies all source/image/label identities and once-only shared gain; all122 calibration labels/source identities and heldout boundaries remain unchanged. The combined data protocol is `c298085a4fd7ab4b1051f432c807c73f67f0d3902948f47b5f51c5d7430882a0`, native manifest `e60c9aba61715e0585b9ed6e13e022548c2779c9a07c0ce3a0c9822ff68a9f55`, and exposure manifest `a12fc6db27fae02429634eea4bb654702af55989335855cb0145da22a7eb9bef`. The preparation-order observation retains that the durable combined plan was written after unchanged base copies but before291 added sources; all source acceptances were already pinned. No source choice or bytes changed during that provenance clarification. Original832/880 histories stay intact. Source-specific bounded visual review and conservative duplicate screens do not establish physical-project independence.

The predeclared four-trial efficiency probe is complete. Each trial resets the identical actual learned head/seed, uses four oldTRAIN examples, batch4,640, freeze4 and four optimizer steps, with the first step warmup and final three timed. No weights are saved or calibration/heldout performance used to choose a format.

| TRAIN-only probe | Training images/s | Peak process working set |
|---|---:|---:|
|Contiguous, CPU3|1.9422|0.928GB|
|Contiguous, CPU4|2.2134|0.940GB|
|Channels-last, CPU3|2.2496|0.948GB|
|Channels-last, CPU4, selected|2.5813|0.956GB|

All trials meet finite-loss and6GiB guards. Channels-last CPU4 is about16.6% faster than contiguous CPU4 in this tiny sample. These are TRAIN optimizer throughput observations, not inference FPS or production capacity; the working-set metric conservatively includes preceding trials. Probe protocol is `21436560a3d53e61e03ca0e59eb0fc7ee56fb1dd7d16f09f7fb5b5ea13dd6063` and summary is `output/training/construction-round4-layout-capacity/summary.json`.

The new fit protocol was frozen **before optimizer work**, SHA-256 `f77168288653f986ddf23a8c601141767d84a172ee8e0d31f9d5059c95da625d`. It pins source acceptances, manifests, gain/helper/trainer source hashes, the head's failed guard and actual122-input decoder parity, exact head ONNX33e7… and retained learned PT5e48…. Configuration is four fixed epochs, freeze4,640, batch4/nominal batch4, CPU4/channels-last, fresh AdamW learning rate.0005, warmup0 and no wall cap. Learned class heads remain0 material loss/1 crack. Checkpoint selection uses the same122 calibration identities/GT; actual serving calibration/parity follows all four epochs before any heldout work. This stage has no hidden extension or automatic promotion.

```powershell
.venv-roads-export/Scripts/python.exe -u -X utf8 scripts/train_construction_damage_round3.py --data-root datasets/construction/evcivil-round4-real-lowlight-gain --continue-from output/training/construction-round3/evcivil-yolov8n-r4-head-640-gain/weights/best.pt --name evcivil-yolov8n-r4-real-lowlight-640-gain --epochs 4 --image-size 640 --max-minutes 0 --freeze 4 --batch 4 --nominal-batch 4 --threads 4 --memory-format channels_last --learning-rate 0.0005 --exposure-policy low_light_gain
```

All four fixed epochs completed and the process exited0. Its complete console is `output/training/construction-round4-real-lowlight-fit.console.log`. The trainer re-verifies every prepared image/label hash before fitting. Its authored source hash is `7f5d07eeeec51a920226558a65ee6cb5656f5fb78201559ad66de163d2ac9c70`; previous executed head source stays archived separately. The layout changes tensor storage, not pixel values/channel order/geometry. About30 minutes of bare optimizer work was a tiny-probe estimate only: actual fit/validation takes2977.46 seconds and through export2979.73 seconds, about49.7minutes, under explicitly authorized shared-host contention. No time interval is subtracted and these are not inference-speed results. No heldout source is opened and neither thin cracks nor material loss is yet fixed.

The best checkpoint remains epoch1: ordinary checkpoint-selection mAP50 is.10184/.08968/.07112/.08126 across epochs1–4. TRAIN losses decline while ordinary CAL selection worsens; extra epochs are not silently added. The selected learned head and adapted backbone are real: monitored blocks4/6/9 change L2.52676/1.00278/1.41042 and classifier L2.28007, with no class reset. The resulting ONNX SHA-256 is `2fbff4feb9f173da3aabab3146832f78d0c200a616d71e66b482f4edb1d4199e`. Ordinary Ultralytics validation is not substituted for the distinct original-source operating-point audit below.

### Combined model final122-source calibration: still fails the minimum

Complete calibration and actual selected-point serving parity finish successfully on all122 unchanged original raw sources. Both classes remain supported; allwrong-class/unmatched alerts are retained. The frozen point is crack.10/material loss.25. Crack has no grid point satisfying the precision/negative constraints, so.10 is the explicitly failed F1 fallback. Material loss.25 maximizes recall within its class precision/negative guard; it does not satisfy the separate minimum recall requirement.

| Full122 two-class comparison | Head BEFORE TP/FN/FP | Combined AFTER TP/FN/FP | Before precision/recall | After precision/recall |
|---|---:|---:|---:|---:|
|Crack|5 /120 /6|7 /118 /73|45.45% /4.00%|8.75% /5.60%|
|Surface material loss|4 /34 /28|5 /33 /1|12.50% /10.53%|83.33% /13.16%|
|Geometric-thin crack annotations|0 /62 /—|0 /62 /—|— /0.00%|— /0.00%|

The small material-loss gain comes with materially fewer wrong-class alerts, but33/38 annotated observations are still missed. Crack precision worsens sharply and the thin subset remains completely missed. Combined true-negative photograph alarms remain3/10. Material loss's common47-source scope also has5TP/33FN/1unmatched; the full122 result never hides crack-class false alerts. Every minimum fails except material-loss precision and selected-serving parity. **Thin cracks and spalling are not fixed.** No heldout model forward, automatic promotion, new YOLO extension or production claim follows.

The immutable operating point is `models/construction/evcivil-yolov8n-r4-real-lowlight-640-gain.operating-point.json`. All122 final runtime records, all candidates and cached raw outputs remain under `output/evaluation/construction-round3/evcivil-yolov8n-r4-real-lowlight-640-gain/`; complete console is `real-lowlight-final-calibration.console.log` in the evaluation root. Accuracy timings are not capacity evidence. All ownCPU/GPU jobs exit before the parent's separately isolated factory performance block; the four fixed genuine-video development exports wait until that block releases compute.

Additive final receipt `models/construction/evcivil-yolov8n-r4-real-lowlight-640-gain.source-license-addendum.json`, SHA-256 `88c9d8fbb0d39e6837f78911c4d531b44693d1c7a1affcf9c73d951ed2503f5f`, pins the unmodified original metadata, calibrated point, exact executed-source archive, source acceptances and failed review. It explicitly adds LCSD's author-linked Apache-2.0 declaration and forfeited author test/val benchmark without changing weights, camera policy, cuts or inherited ev-CIVIL/DACL/AGPL restrictions. The source declaration does not independently clear universal rights or initial source-weight data licensing.

At06:27:34UTC, while epoch1 batch104/293 was already fitting, an additive **pre-operating-point** review receipt repeats the original minimum50% recall for each class and geometric-thin subset,80% precision for each class, at most10% combined negative alarms and exact serving parity. It pins the original head guard and original calibration-source protocol. This is explicitly not retroactive pre-fit registration: ordinary epoch validation still chooses the checkpoint. Receipt `datasets/construction/evcivil-round4-real-lowlight-gain/evaluation-minimum-guard.json`, SHA-256 `52d24e01b187c47c57e95702fda829b259add2575ede35d96214f849416d7002`, does not alter the protected fit protocol or sample data. Full122 results retain wrong-class errors. Even all minimum checks passing requires root approval before heldout inference and supplies neither a fix nor production qualification. Failure causes no automatic hidden training extension.

### Fixed whole-video development comparison, before any combined postfit scoring

The four exact source recordings are old exposed `crack_89`, day CAL `dset_6/crack_129`, material-loss CAL `dset_6/spalling_75`, and first night CAL `night_nov18/ros_crack_21`. BEFORE is the latest R4 head at its already frozen cuts; AFTER is the combined candidate at its complete122-source calibrated point. No training-success clip, favorable posthoc replacement or heldout recording enters this comparison. All are explicitly development/calibration and receive zero independent accuracy credit.

Protocol `output/evaluation/construction-round3/four-development-video-comparison-protocol.json`, SHA-256 `0006e730fc4951044d395a7921796b1c226884d574ae10ed4d8a09604d19ee7b`, pins source-file identities, original source roles, before model/cuts and after fit/evaluation receipt. `compare_construction_development_videos.py` checks this explicit whitelist against protected calibration/heldout metadata **before loading pixels**. It never invokes the generic comparison loader, which includes heldout sources. Three metadata-only tests exercise fixed selection, heldout collision and rejection of an unlisted TRAIN source; all pass. Its prepared four-pane view separates untouched RAW, actual native gained input, external publisher GT and model-only predictions, preserving every original exposure/PTS and every scored miss/unmatched/wrong-class prediction. Unlabelled frames remain explicitly unscored. Both classes are supported in both models; unsupported annotations must be zero.

All eight exports are now complete: the same434 genuine camera exposures per model,868 encoded annotated frames in total. Both candidates remain explicitly **FAILED CALIBRATION**. Exposed/calibration clips, contended exports and encoded playback are not production qualification or15FPS evidence; old immutable seen-training exports remain separately preserved.

| Complete source | Frames | Before TP / FN / unmatched | After TP / FN / unmatched |
|---|---:|---:|---:|
|Exposed thin crack89|24|0 /8 /4|0 /8 /8|
|Day calibration crack129|104|0 /71 /49|1 /70 /1|
|Calibration material-loss75|84|0 /77 /1|0 /77 /0|
|Night calibration ros_crack21|222|0 /442 /45|10 /432 /412|

Counts include every scored observation and both classes. Before crack129 includes46 unmatched material-loss predictions and the night clip45; after night's412 unmatched crack predictions remain visible and counted. Unlabelled exposures are explicitly unscored. Unmatched means no match under the publisher's available boxes, not independent proof that every candidate is physically undamaged. These are correlated frame observations, not598 distinct physical defects. The night model gains a few box matches but produces many unmatched candidates; the thin89 and small material-loss75 remain entirely missed. This is failed research evidence, not a repaired inspection system.

`output/evaluation/construction-round3/four-development-video-paired-summary.json` lists all eight actual MP4/report paths, hashes and per-class totals. Each video separates untouched RAW, actual native gained input, external publisher GT and model-only overlays in a2×2 arrangement. Nearest2× enlargement is display only. Fixed first-annotated previews are chosen by source annotation order, never favorable model output. Visual QA confirms the actual thin crack, small wall chip and night cracks are present; the corresponding model misses/unmatched candidates are plainly marked.

All four replay inputs are re-decoded with PyAV and OpenCV against their exact whitelisted original arrays before inference: every gray/BGR pixel and original microsecond PTS matches. Existing replays are read-only; only the previously absent complete night-CAL replay is created. No original source or older report/video is overwritten. Annotated MP4 retains one frame per actual camera exposure and zero PTS discrepancy.

The independent count/actual-IoU/timeline auditor passes all19 reports (the eight new plus11 existing), documented in `docs/construction-development-recording-count-audit.md`. A separate JSON-only audit also verifies identical before/after raw/gained hashes, GT and full timelines. The export environment uses NumPy2.4.6 while serving/CAL uses2.2.6, with identical OpenCV5.0.0/OpenVINO2026.1 builds; actual prediction order, labels, boxes and scores agree to1e-6 on all48 shared sampled CAL exposures per model. This is measured parity rather than an assumption based on library names. Evidence is `output/evaluation/construction-round3/four-development-video-environment-parity.json`. The three whitelist/leak tests pass again in.38seconds.

Reproduce after the prerequisite frozen model/points and coordinated GPU release:

```powershell
.venv-roads-export/Scripts/python.exe -u -X utf8 scripts/compare_construction_development_videos.py --export
.venv/Scripts/python.exe -X utf8 scripts/audit_construction_development_parity.py
.venv/Scripts/python.exe -X utf8 scripts/prepare_construction_development_integration.py
```

`four-development-video-integration-proposal.json` contains actual hash-checked model/preset/recording blocks for the root-owned app registry, with diagnostic-only flags and no auto-default. It does not itself mutate registrations. Input15FPS in these replay presets is a requested schedule, never a measured capacity claim.

The first invocation used the app environment, which lacks PyAV, and stopped before any NN/media work. The existing isolated export environment supplies PyAV/HDF5 without an installation. A later presentation-only missing-license-key exception stopped before the first AFTER media write; completed BEFORE outputs are preserved and hash-verified on recovery, never overwritten or re-inferred. Separate failure/recovery consoles remain. No case, data, weights, input policy, operating point or scoring rule changes during these repairs. Export overlaps separately authorized pixel-model CPU1 work; none of its elapsed time qualifies speed. GPU is explicitly handed back for factory C calibration after all eight outputs finish.

A generic `lite_semseg` dispatch path is prepared for a future separately qualified crack specialist, with no actual weight registration or default change. It preserves the existing pinned Lite adapter and raw input, forwards declared gain/single-crack cuts, records gain metadata without double-enhancing NN input, and rejects spalling thresholds. Eighteen affected adapter/profile/legacy tests pass in.49seconds. The separate pixel-supervision experiment remains owned by its training/evaluation agent; this compatibility path is not evidence that its accuracy passes or that it detects spalling.
