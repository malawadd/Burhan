# Factory accuracy improvement experiment

This study targets higher visible surface-defect recall while retaining at least 15 processed observations per second on this laptop. It does not certify an industrial line, arbitrary products, or invisible/electrical defects. Results will include false positives, missed defects, abstentions and speed.

**No experiment improved genuine localized-defect recall on the reused 80-image comparison: the best remains 8/11 defective views (72.7%). The factory recall goal is not achieved, and no review tier is selected as an accuracy upgrade.** The apparent 81.8% review score was flagged-image classification: its extra box was elsewhere in the defective image and did not touch the real defect. The fused neural model improved precision and speed, while retaining the same three missed defects. The original baseline remains the appropriate reference, not a production-qualified model.

Speed was preserved: the experimental review path processed 900/900 observations at configured 15 FPS for 60 seconds with no drops, late decisions or errors, and a separate 16 FPS input run measured 15.9 processed observations/s including drain. Accuracy and runtime capacity are separate results. Four-fold item-disjoint validation found the constrained review policies less robust than the baseline, including 80 abstentions in one fold; full results are below.

## Preserved baseline and evaluation boundaries

The original locally fitted artifact is `models/factory/kolektor_surface_rf.joblib`, SHA-256 `549d8528c1e2b8ca483b772e9ada4f71284782dca2cd631be7127171cadd0473`. Its metadata and prior reports remain unchanged. The original native-backend operating point is 32 trees, width 256, pixel vote threshold 0.995 and minimum component area 8.

The 50 physical items remain separated into the original 30 training, 10 calibration and 10 comparison items. The 80 comparison images have already been examined repeatedly during development. Any new score on those images is therefore a **reused development comparison, not a fresh blind test**. Item separation prevents direct training-image leakage but does not erase human familiarity with the comparison set.

The original 80-image comparison was 8 TP / 3 FN / 1 FP / 68 TN: recall 72.7%, precision 88.9%, with zero quality abstentions. The small denominator is material: one additional detected defective view changes recall by 9.1 percentage points. No performance here establishes production acceptance.

**Evaluation correction:** the initial metric counted a defective image as detected whenever any candidate appeared anywhere in it. That is flagged-image classification, not proof that the defect was located. Saved display boxes were therefore checked against the real masks. The extra review flag in `kos02/Part6.jpg` is a false location; it cannot be counted as a recovered defect. The corrected objective and final comparison additionally require retained prediction pixels to overlap the actual original-resolution annotation. Predicted components are nearest projected from the analysis grid, retaining the same minimum size and largest-20 candidate cap as runtime. Foreground pixel recall/precision and micro pixel IoU exclude background true negatives; they are separate from per-view recall.

Unless explicitly marked localized, the historical model-development recall/precision figures below are flagged-image classification metrics.

| Reused 80-view experiment | Flagged defective images / 11 | Defective images with any displayed box touching the mask / 11 | Normal false alarms / 69 |
|---|---:|---:|---:|
| Original baseline | 8 | 8 | 1 |
| V3 learned forest | 6 | 6 | 0 |
| V4 learned forest | 7 | 7 | 2 |
| 3/70 experimental review | 9 | 8 | 3 |
| 7/70 experimental review | 9 | 8 | 7 |
| V8 learned fused neural model | 8 | 8 | 0 |

The box-touch criterion is permissive and is not segmentation IoU. Even this lenient check shows no additional genuinely located defective view. Offline reports are `review-box-overlap.json`, `v3-box-overlap.json`, `v4-box-overlap.json`, `v8-box-overlap.json` and `high-sensitivity-box-overlap.json`; they reuse saved findings and perform no fitting, threshold selection or inference.

## Rejected threshold-only experiment

Before scoring calibration images, the first experiment fixed component area at four grid pixels and declared vote thresholds 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.98, 0.99 and 0.995. Selection required calibration precision at least 90%, then maximized recall and sensitivity. No comparison labels were loaded in this experiment.

No operating point met that precision floor. At 0.99, all ten calibration defects were found but six normal views were rejected, giving 62.5% precision. Even 0.995 with four-pixel components produced two false positives and 83.3% precision. Lower thresholds were worse, including the unacceptable all-positive classifier. This approach was rejected rather than claiming better recall by flagging everything.

The raw calibration-only report is `output/optimization/factory-accuracy/calibration-sensitivity.json`. Collection took 27.75 seconds with the local native backend; this was an exploratory scoring run rather than a deployment speed benchmark.

## New training protocol

`scripts/improve_factory_surface.py` writes its complete protocol before fitting or calibration scoring. It never opens comparison images or masks. A new versioned artifact is used; the original model is not overwritten.

The planned learner is a 16-tree Random Forest, maximum depth 16, minimum leaf size two, with balanced per-tree class weights. It retains the existing 11 features and width-256 input. Each training image supplies 1,000 random normal pixels and 1,000 difficult normal pixels selected from a 20,000-pixel training-only pool using the original forest. Up to 2,000 annotated positive pixels are sampled and repeated twice. Fractional mask resizing preserves any annotated defect coverage in a feature cell, reducing loss of thin defects during target generation.

Operating point selection uses calibration images only, requiring at least 90% precision and then maximizing recall, precision, lower vote threshold and smaller component area. The predeclared component grid is 2, 4, 8, 16, 32, 64 and 128 pixels, using the same vote grid as above. Calibration item jackknife fits the operating point on nine calibration items and evaluates the tenth, exposing its sensitivity to individual items. This is threshold-selection validation with a fixed forest, not independent training cross-validation, and the calibration set was already used for the earlier model.

The fit/calibration run completed in 84.04 seconds using 546,076 sampled rows, including 68,076 positive rows after repetition. It selected vote **0.7 / component area 128**, producing 10 TP / 0 FN / 0 FP / 70 TN on calibration. The item-jackknife produced the same aggregate counts. A second calibration-perfect point was 0.995 / area 16, but the predeclared lower-vote tie-break selected 0.7 / 128. This larger component requirement is a meaningful tradeoff: weak connected regions can be retained at a lower vote, while small isolated regions are discarded. Neither calibration result predicts unseen-defect sensitivity with confidence.

The candidate has been frozen before comparison evaluation: `models/factory/kolektor_surface_rf_v3_hardneg.joblib`, SHA-256 `6dd8e418073270c92d4f5c16b12cf59313d088e7bca08fc45abf4dab1d06b4c2`; metadata SHA-256 `73573df0b2a743c011f6cd2bd8b60ef62448200afad38b94bcd5a77ca2f9578a`. The raw fit/calibration report is `output/optimization/factory-accuracy/training-calibration.json`; the prewritten protocol is beside it.

## Rejected v3 result

The predeclared v3 operating point failed both accuracy and speed. On the reused 80 views it produced **6 TP / 5 FN / 0 FP / 69 TN** (54.5% recall, 100% precision), with no abstentions. Paired baseline remained 8 / 3 / 1 / 68. Mean full-profile time rose from 30.62 to 169.87 ms, and P95 from 48.07 to 288.63 ms. Serial capacity was only 5.89 observations/s. It is retained as a rejected experiment and is not configured as a replacement.

The calibration-perfect result clearly failed to generalize; its low threshold also required many more native tree traversals. Those results are preserved in `output/optimization/factory-accuracy/before-after-first.json`. Subsequent models are development iterations with this failure known. No later comparison on these images is a blind test.

## Predeclared v4 experiment

Before v4 fitting, the protocol changes to an unweighted 16-tree forest with depth 14, 4,000 random normal samples and 250 hard normal samples per training image, retaining dense positive sampling and source-coverage masks. This reduces the class-weight-induced high background leaf scores that weakened early rejection. The input grid is width 320, preserving more real defect detail than 256; width 256 is the predeclared speed fallback, chosen only using calibration throughput and accuracy.

The component grid is restricted to 2, 4 and 8 pixels to retain small isolated defects. Vote thresholds are restricted to 0.9, 0.925, 0.95, 0.975, 0.98, 0.99, 0.995 and 0.999. The calibration rule requires precision at least 90%, then maximizes recall, precision, smaller area and higher vote threshold. A calibration speed gate precedes any reused-comparison evaluation. This explicitly addresses the v3 calibration selection's large-region and CPU-cost weaknesses; no v4 result is claimed before measurement.

V4 selected 0.995 / 8 pixels with calibration 10 TP / 0 FN / 0 FP / 70 TN. Calibration item jackknife was weaker: 9 TP / 1 FN / 1 FP / 69 TN. Fit/calibration took 96.91 seconds. It passed the calibration speed screen at 17.32 serial observations/s (mean 57.74 ms, P95 86.59 ms), with no abstentions.

The frozen reused-comparison result then failed accuracy: **7 TP / 4 FN / 2 FP / 67 TN**, recall 63.6%, precision 77.8%, zero abstentions. Paired mean/P95 were 53.18 / 80.34 ms (18.81 serial observations/s). This candidate is rejected; the speed result is not sufficient for selection. Reports are `training-calibration-v4.json`, `calibration-speed-v4.json` and `before-after-v4-first.json` in `output/optimization/factory-accuracy/`.

## Explicit review operating point on unchanged weights

The precision floor in the first experiments was a development choice, not a user requirement. A separate recall-first, human-review protocol was therefore declared before selecting another operating point. It uses the original forest, vote grid 0.95 / 0.975 / 0.98 / 0.99 / 0.995, component grid 1 / 2 / 4 / 8, and a maximum calibration false-positive rate of 3/70 normal views. Among eligible points it maximizes recall, then prefers the lower vote threshold and smaller area. An all-positive classifier cannot meet that budget. This is an operating-point change, not a newly trained model.

It selected **0.99 / 8**, yielding 10 TP / 0 FN / 1 FP / 69 TN both on the calibration set and the calibration item jackknife. Calibration full-profile capacity was 30.50 observations/s. On the frozen reused comparison its **flagged-image classification** was 9 TP / 2 FN / 3 FP / 66 TN, image recall 81.8% and precision 75.0%. This did not recover another localized defect: the additional box in `kos02/Part6.jpg` was away from the annotation. Actual localized defective-view recall stayed 8/11 (72.7%). False positives increased from 1/69 to 3/69 normal views. F1 fell from 0.800 to 0.783, and the supposed recall gain disappears under localization checking. It is an experimental frame-review workload change, not a defect-recall or overall-accuracy improvement. Both models abstained on zero views.

Paired full-profile mean/P95 were 33.13 / 50.06 ms before and 33.33 / 50.18 ms after. Serial capacity remained approximately 30 observations/s; this is not a 15 FPS runtime acceptance test. `models/factory/kolektor_surface_rf_v2_recall_review.joblib` contains unchanged original weights; its separate metadata identifies the review operating point. Original model and metadata stay untouched. Reports: `high-recall-calibration.json`, `calibration-speed-review.json`, `before-after-review-first.json`.

## Learned two-stage region model

The next predeclared experiment uses the first two original training-only trees at a fixed 0.25 proposal threshold, followed by a newly trained region classifier. The proposal step is deliberately sensitive and cheap. Nineteen descriptors measure real connected-component dimensions, fill ratio, image position, texture contrast, gradient and vote statistics. A 32-tree depth-10 forest learns which proposed regions overlap annotated training defects. Original training/calibration/comparison item boundaries remain intact. No generated images or video enter fitting or evaluation.

Region thresholds are selected on calibration with the same 3/70 normal-view false-positive budget, maximizing recall and then sensitivity. Public comparison labels do not enter that selection. The versioned bundle explicitly contains both proposal and region forests; runtime validation checks feature counts, class inventories and the artifact hash. Candidate scores are region-forest leaf fractions, not calibrated probabilities or pixel probabilities.

The two-stage v5 model is **rejected at calibration**, before evaluating any reused comparison image. Its proposals cover all 31 defective training views, and its region learner used 118,605 components including 286 positive ones. However, the best declared-budget calibration point was 0.925: **5 TP / 5 FN / 1 FP / 69 TN**, only 50% recall. Training/calibration took 27.69 seconds. Training proposal coverage did not establish generalization; filtering removed real defects. Report: `training-regions-v5.json`. The experiment remains explicit and reproducible rather than becoming an unsafe default.

## Frozen neural texture representation

The next experiment changes the representation rather than repeating forest sampling variants. It uses official torchvision MobileNetV3 Small ImageNet V1 weights, downloaded from the URL exposed by the installed torchvision enum, checked by the official hash prefix and loaded with `weights_only=True` in the separate export environment. Only the first four feature blocks are exported. The runtime uses a local ONNX file with SHA-256 verification and explicitly selects the installed Intel GPU through OpenVINO; there is no runtime Torch dependency or network call.

Input is 512×1280 uint8 BGR. RGB conversion and ImageNet normalization are inside the exported graph. Sixteen stride-two channels preserve approximately the original input detail; 24 stride-eight texture channels are upsampled and concatenated with the existing 11 grayscale features. The head receives 51 channels on a 256×640 grid. The predeclared supervised head is 16 trees / depth 14 / minimum leaf two, trained using only training masks, 2,000 sampled normal pixels per view and up to 2,000 positive pixels repeated four times. Brightness factors 0.9 and 1.1 apply only to defective **training** views, with matching masks; they are augmentation, not validation examples or fabricated demonstration footage.

The encoder is profiled at its actual fixed input shape before fitting the head, to check whether 15 FPS remains plausible. Calibration uses vote grid 0.95–0.995, minimum regions 1 / 2 / 4 / 8 and the declared maximum 3/70 normal-view false-positive rate, maximizing recall then sensitivity. Final accuracy and complete-runtime capacity require measurement; ImageNet pretraining alone does not establish industrial suitability.

The first FP32 feature/fusion probe took 56.47 ms warm median and 61.57 ms P95 before the head and quality checks, leaving insufficient headroom. Stage profiling found approximately 3.2 ms preprocessing, 10.0 ms GPU encoding, 9.6 ms handcrafted features and 22.7 ms CPU upsampling/concatenation. Instead of fitting that forest, the actual implementation moved the learned pointwise head and fusion into the exported GPU graph. Only one 256×640 score map is returned to the CPU.

V7 used a 40→16→1 pointwise neural head on frozen pretrained channels, trained for 15 epochs on 945,756 sampled real training pixels including repeated/augmented positives. Fitting took 8.17 seconds. Its median/P95 GPU prediction path was 22.86 / 35.00 ms, but its calibration false alarms were unacceptable: all ten defects and 46/70 normal views flagged at 0.995 / area 8. No calibration point in the predeclared grid met the false-positive budget. V7 was rejected without running the reused comparison.

V8 mined the top 1,000 wrongly high-scoring **normal pixels from each original training view** using frozen V7, repeated those hard normal samples four times, and fit a 40→32→16→1 head for 22 epochs. The feature standard-deviation floor is 0.05 to avoid amplifying almost constant grayscale color channels. GPU output stores logits; a CPU float32 sigmoid avoids FP16 probability quantization. Before V8 calibration, the score grid was expanded using logits 0, 1, 2, 3, 4, 5, 6, 7, 8, 10 and 12. This addition is recorded in the pre-fit protocol and was made before V8 calibration scores were viewed; it was not tuned using comparison misses.

V8 selected **0.98 / area 8**: calibration 10 TP / 0 FN / 3 FP / 67 TN, recall 100%, precision 76.9%. Item-jackknife was 10 / 0 / 4 / 66, precision 71.4%. This meets the declared review false-alarm budget, **not** the earlier 90% precision gate. Its full-profile calibration timing was mean 30.56 ms / P95 38.35 ms (32.73 serial observations/s), zero quality abstentions.

On the single frozen reused comparison, V8 produced **8 TP / 3 FN / 0 FP / 69 TN**: recall **72.7%**, precision **100%**, F1 0.842, and no abstentions. Baseline remained 8 / 3 / 1 / 68. Mean/P95 complete-profile timing was 29.54 / 36.76 ms before and 24.89 / 29.72 ms after; V8's serial capacity was 40.18 observations/s. It improves measured precision and speed, but does **not** achieve the requested recall improvement. The same three defects remain missed. Runtime sustained FPS and moving-video accuracy are separate questions.

V8 ONNX SHA-256: `10265fb5df70b37e4e429269b31b4d67b12be0c6352ae2da11d20d1b5d1b768b` (55,371 bytes). Official pretrained source SHA-256: `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`. Raw reports: `neural-head-fit-v8.json`, `neural-calibration-v8.json`, `calibration-speed-v8.json`, `before-after-v8-first.json`.

## Review authority and additional preregistration

Review artifacts explicitly set `manual_review_only=true`, `automatic_acceptance=false` and `automatic_rejection=false`. Confirmed candidate observations emit status `review`; negative observations emit `no_candidate`, not `clear`. Both request human judgment; the negative title/guidance state that a missing candidate does not approve the unit. Event labels identify review candidates rather than verified defects. Runtime telemetry can record a completed observation without authorizing a physical accept/reject actuator.

The higher-sensitivity tier was preregistered in `high-sensitivity-preregistration.json`: unchanged original forest, vote grid 0.95 / 0.975 / 0.98 / 0.99 / 0.995, minimum region 1 / 2 / 4 / 8 and at most 7/70 calibration normal views flagged. It maximizes recall then prefers a lower threshold and smaller component area within that budget. This spends a larger manual-review workload; it is not an automatic rejection or production qualification policy.

Calibration selected **0.975 / area 8**: 10 TP / 0 FN / 5 FP / 65 TN, recall 100%, precision 66.7%; the calibration item-jackknife had identical counts. The complete-profile calibration gate measured mean 39.18 ms / P95 60.84 ms (25.52 serial observations/s), zero abstentions. On the single frozen reused comparison, it produced **flagged-image classification** 9 TP / 2 FN / 7 FP / 62 TN, image recall 81.8%, precision 56.25%, zero abstentions. Only eight flagged views have any annotated defect overlap; localized recall remains 72.7%. Mean/P95 were 35.15 / 54.83 ms baseline and 35.55 / 52.54 ms after. The more sensitive calibration policy did not recover an additional comparison defect; it increased manual-review false positives from 3/69 to 7/69 normal views. It is retained as a measured tradeoff, not selected as the preferred review setting. Reports: `high-sensitivity-calibration.json`, `calibration-speed-high-sensitivity.json`, `before-after-high-sensitivity-first.json`.

The lower-alarm experimental review artifact is `models/factory/kolektor_surface_rf_v2_recall_review.joblib`, score 0.99 / area 8, manual review only; it is not an accuracy upgrade or the default production model. Its weight hash is identical to the preserved original forest, so **weight hash alone does not identify its policy**. Profile metadata and persisted evidence expose `model_metadata_sha256`, effective `decision_threshold`, `minimum_component_area` and any `region_decision_threshold`. The metadata hash identifies the exact loaded policy bytes. Review status/title/action and physical-unit disposition state are explicit. Triggered `completed_units` means an observation was delivered, not that a product passed or a reject actuator fired.

The completed four-fold study excludes the original ten exposed comparison items. The remaining 40 items rotate through ten outer-evaluation items, ten inner-calibration items and twenty training items per fold. Each forest is refit only on its inner-training items; both 3/70 and 7/70 policies select their operating point only on inner calibration. Every outer item is excluded from fitting and threshold selection. This is development cross-validation of fixed policies, not a new external blind test. It reports accuracy only; training-era execution timings are not latency benchmarks. A fold unable to meet its calibration budget emits explicit abstentions, counted separately.

## Cross-validation coverage and actual localization

The study evaluates **319 unique views from 40 items**, including **41 defective views and 278 normal views**. In one fold neither constrained review policy could meet its inner-calibration false-alarm budget. Both therefore abstained on that entire outer fold: **11 defective and 69 normal views**, leaving only **239/319 views assessed (74.9% coverage)**. The original policy assessed all 319. Headlining recall only among assessed images would conceal those missed opportunities.

| Fixed policy | Localized defects / all 41 | All-view localized recall | Normal false alarms | Abstained defective / normal | Foreground pixel recall | Foreground pixel precision | Micro pixel IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| Refit original F1 policy | 34 | 82.9% | 4 / 278 | 0 / 0 | 15.92% | 92.24% | 0.1570 |
| Refit 3/70 review policy | 28 | 68.3% | 6 / 209 assessed | 11 / 69 | 14.62% | 92.32% | 0.1445 |
| Refit 7/70 review policy | 30 | 73.2% | 19 / 209 assessed | 11 / 69 | 19.31% | 84.94% | 0.1867 |

Here localization requires actual retained predicted foreground pixels to hit the original mask. The more permissive display-box check produced the same aggregate view recall in these CV folds. Pixel recall and IoU include annotated foreground from abstained defective views in their denominator. Review 3/70 conditional recall among assessed defective views is 28/30 = 93.3%; review 7/70 conditional recall is 30/30 = 100%. **Neither conditional figure describes all-view performance**: their all-view recalls are 68.3% and 73.2% because eleven defective views were not assessed. Neither policy is validated as an improvement.

Reports: `item-crossvalidation-review/report.json` and `item-crossvalidation-review/mask-localization-report.json`. Fold models, calibration choices, every original and repeated mask-evaluation prediction, and model/metadata hashes are retained. The mask postprocess performs a second accuracy-only inference with the frozen fold models and asserts agreement with the original fold classification statuses. It does not select parameters or report inference speed. `metadata-provenance-corrections.json` records corrections to inherited source timestamps/calibration descriptions; learned weights and effective thresholds were unchanged.

## One corrected mask-aware calibration objective

After discovering the image-level evaluation flaw, one corrected objective was registered before scoring calibration: original weights, vote grid 0.95 / 0.975 / 0.98 / 0.99 / 0.995, component areas 1 / 2 / 4 / 8, calibration normal-view false-alarm rate at most 7/70. It maximizes localized defective-view recall, then annotated foreground-pixel recall, then pixel precision, then smaller component area. No comparison scores or missed-image scores entered selection.

It independently selected the same **0.975 / area 8** operating point. Calibration localized 10/10 defective views, with five normal false alarms, 36.37% foreground-pixel recall, 90.39% pixel precision and micro pixel IoU 0.3502. The candidate was frozen before its one reused 80-view comparison.

| Corrected original-resolution mask metric | Original baseline | Mask-calibrated review |
|---|---:|---:|
| Localized defective views / 11 | 8 | 8 |
| Localized defective-view recall | 72.7% | 72.7% |
| Normal false alarms / 69 | 1 | 7 |
| Flagged defective frames with no true foreground hit | 0 | 1 |
| Foreground pixel recall | 11.37% | 19.78% |
| Foreground pixel precision | 96.88% | 88.45% |
| Micro pixel IoU | 0.1133 | 0.1928 |
| Abstentions | 0 | 0 |

The corrected policy covers more pixels within already detected defects, but still misses the same three defective views and increases normal false alarms. It is not a factory recall upgrade, and no further tuning is presented. Reports: `localized-protocol-before-selection.json`, `localized-calibration.json`, `localized-frozen-comparison.json`. This experiment corrects the evaluation objective; it does not create a fresh independent test or justify automatic machine disposition.

## Matched sustained runtime test

With the actual shared runtime, preview enabled and the same 80 genuine public images cycling under a software clock, the matched 60-second tests at configured **15 input FPS / 200 ms deadline** processed **900/900 observations** before and after. Both had **zero inspection drops, zero late decisions and zero errors**. Host-receipt-to-decision P95/max were **56.1 / 95.3 ms** before and **56.4 / 91.6 ms** after. The source lifecycle includes approximately 0.2 seconds of EOF drain, so the configured active capture rate should not be confused with dividing 900 by the complete session duration.

These load runs execute real inference for every processed frame, but repeated still images are not moving factory footage, additional independent accuracy cases, camera exposure/transport measurements or every-item industrial coverage. Reports: `output/optimization/factory-accuracy/runtime-before-15fps.json` and `runtime-after-review-15fps.json`.

The separate **16 FPS / 200 ms**, 30-second acceptance run processed **480/480** observations with zero drops, late decisions or errors, and measured **15.9 processed observations/s including EOF drain**. Host P95/max were **57.8 / 88.0 ms**. This demonstrates at least 15 processed observations/s in that actual short laptop run, while the tested lab configuration uses 15 FPS for ordinary capture. This review setting is not adopted as an accuracy improvement. Report: `output/optimization/factory-accuracy/runtime-after-review-16fps.json`. A Windows laptop result is an observation, not a hard real-time guarantee or a qualified production miss rate.

## Reproduction and artifact inventory

Run timing jobs one at a time with the application station stopped. These commands operate on real local public images; software-clock still replay is explicitly a load test, not a fabricated video or independent accuracy case.

```powershell
# Recollect calibration component statistics without opening comparison items.
.venv/Scripts/python.exe -X utf8 scripts/collect_factory_calibration_scores.py
.venv/Scripts/python.exe -X utf8 scripts/calibrate_factory_recall.py --budget 3 --cache output/optimization/factory-accuracy/calibration-scores-recollected.json
.venv/Scripts/python.exe -X utf8 scripts/calibrate_factory_recall.py --budget 7 --cache output/optimization/factory-accuracy/calibration-scores-recollected.json --model models/factory/kolektor_surface_rf_v2_high_sensitivity_review.joblib --output output/optimization/factory-accuracy/high-sensitivity-rerun.json
.venv/Scripts/python.exe -X utf8 scripts/benchmark_factory_accuracy.py --after models/factory/kolektor_surface_rf_v2_recall_review.joblib --repeat 3 --output output/optimization/factory-accuracy/review-rerun.json
.venv/Scripts/python.exe -X utf8 scripts/crossvalidate_factory_surface.py --candidate review-both
.venv/Scripts/python.exe -X utf8 scripts/evaluate_factory_cv_masks.py --cv output/optimization/factory-accuracy/item-crossvalidation

# One corrected objective; this experiment did not improve localized view recall.
.venv/Scripts/python.exe -X utf8 scripts/calibrate_factory_localized.py
.venv/Scripts/python.exe -X utf8 scripts/evaluate_factory_mask_comparison.py

# Paired sustained software-clock load tests; do not overlap these runs.
.venv/Scripts/python.exe -X utf8 scripts/benchmark_runtime.py --model models/factory/kolektor_surface_rf.joblib --seconds 60 --fps 15 --deadline 200 --varied --output output/optimization/factory-accuracy/runtime-baseline-rerun.json
.venv/Scripts/python.exe -X utf8 scripts/benchmark_runtime.py --model models/factory/kolektor_surface_rf_v2_recall_review.joblib --seconds 60 --fps 15 --deadline 200 --varied --output output/optimization/factory-accuracy/runtime-review-rerun.json
```

The rejected learned candidates remain reproducible, using versioned artifacts and separate reports:

| Experiment | Entry point | Outcome |
|---|---|---|
| V3 hard negatives / balanced forest | `scripts/improve_factory_surface.py` | Rejected for lower recall and insufficient speed |
| V4 dense normals / width 320 | `scripts/improve_factory_surface_v4.py` | Speed screen passed; lower comparison recall |
| V5 learned component filter | `scripts/train_factory_regions.py` | Rejected at calibration; no comparison evaluation |
| Frozen MobileNet export | `.venv-roads-export/Scripts/python.exe scripts/export_factory_encoder.py` | Trusted official weights, local ONNX; initial CPU feature fusion too expensive |
| V7 fused pointwise head | `scripts/prepare_factory_neural_samples.py`, then export-environment `scripts/fit_factory_neural_mlp.py`, then `scripts/calibrate_factory_neural.py` | Useful speed; unacceptable calibration false alarms; no comparison evaluation |
| V8 train-only hard negatives | `scripts/mine_factory_neural_hardneg.py`, then export-environment `scripts/fit_factory_neural_mlp_v8.py`, then `scripts/calibrate_factory_neural_v8.py` | Improved precision and speed, unchanged comparison recall |

PyTorch/torchvision/ONNX export runs in the existing separate export environment; ordinary runtime requires no Torch installation. The actual installed GPU, weights, artifact hashes, train/calibration boundaries and raw per-image results are recorded in the reports. Dataset/model rights remain those described in [factory surface model provenance](models/factory-surface.md): the Kolektor-derived artifacts are non-commercial research prototypes under the dataset's CC BY-NC-SA terms. These experiments do not enlarge the visual task beyond Kolektor-like commutator surface close-ups.

Focused contract tests cover region-bundle validation, manual-review statuses/actions, same-weight/different-policy metadata hashes, neural score-map coordinate mapping and the explicit GPU backend requirement. Their generated unit fixtures are contract checks only; they are not included in dataset accuracy or demonstration claims.
