# Factory false-alarm reduction and expanded construction damage inspection

Measured locally on 20 October 2026. This round starts from the spatial factory model and joint-aware construction crack detector described in [round two](inspection-improvement-round2.md). It does not restart the comparison at the older 72.7% factory baseline.

## Requirements and measurement rules

- Factory: improve the current 10/11 genuinely localized defect views, reduce the current 5/69 normal false alarms, and sustain at least 15 processed observations/s. Keep original-resolution mask precision, recall, IoU and component matching visible alongside image-level results.
- Construction: improve the thin-crack video with zero matches, train or acquire a genuine spalling detector, and add useful visible-damage categories backed by actual data/models. Keep external annotations distinct from model predictions.
- All inference and training execute on this laptop. Public asset downloads and source research use the network; inspection frames are not uploaded for inference.
- Research, code and downloads run in parallel agents. Initial fits and inference windows take turns. A later scheduling amendment allows the fixed-step shallow factory fit on one CPU thread alongside the construction CPU fit on four threads; those training durations explicitly include contention. Qualification speed tests remain isolated from other training/inference jobs. Unrelated OS/browser processes are not isolated or disabled.
- The original factory comparison and three construction challenge clips have already been exposed. They are development comparisons; training/calibration use separate declared groups. Repeated exposures do not increase independent accuracy sample size.
- Preserve rejected candidates, missed defects, false alarms, dropped observations and latency tails. Do not lower a matching standard to manufacture success.

## Fresh before measurements

The parent independently reran the frozen v10h original-mask audit before model edits. It reproduced round two exactly:

| Factory measure | Before: v10h |
|---|---:|
| Genuine localized views, any real-mask overlap | 10/11 (90.91%) |
| Views with foreground IoU ≥0.10 | 10/11 |
| Views covering ≥50% of actual defect pixels | 7/11 |
| Normal views flagged | 5/69 (7.25%) |
| Foreground pixel recall | 54.44% |
| Foreground pixel precision | 68.73% |
| Micro foreground IoU | 0.4363 |
| One-to-one region recall, IoU ≥0.50 | 5/11 |
| Abstentions | 0 |

Raw report: `output/optimization/factory-round3/before-localization-audit.json`. Every defective and false-alarm view has saved original-resolution predicted/publisher masks and four-panel error images in `before-masks/`.

The first runtime replay overlapped an already-running dashboard RF station. Its 960/960 observations, 11 late decisions and P95 57.9 ms remain in `before-runtime-dashboard-overlap-16fps-60s.json`; it is **not the controlled before timing**. Dashboard state/settings were saved in `dashboard-before-benchmark.json`, then the station was paused before repeating.

The repeated before run requested 16 FPS for 60 seconds with preview at 6 FPS, metadata persistence enabled, event-image persistence disabled and a 100 ms decision deadline:

| Shared-runtime measure | Before: v10h |
|---|---:|
| Captured / processed | 960 / 960 |
| Actual throughput from rounded uptime | 960 / 60.2 ≈15.95/s |
| Dropped / source-skipped / inspection errors | 0 / 0 / 0 |
| Decisions later than 100 ms | 9 |
| Median / P95 / P99 | 38.7 / 56.2 / 96.6 ms |
| Maximum | 138.4 ms |
| Station initialization before capture | 9,461 ms |

Raw report: `before-runtime-16fps-60s.json`, including opt-in per-call stage timing traces. Inputs cycle 80 predecoded real public stills. This measures the shared runtime rather than a physical moving line, camera exposure or reject actuation. Startup is excluded from steady-state FPS and reported separately. Average throughput does not mean every deadline was met.

## Parallel experiment ownership

- Factory: learned context and shallow ablation, improved thin-mask training targets, hard negatives and calibration-only false-alarm constraints. [Protocol and results](factory-accuracy-round3.md).
- Construction crack/spalling: additional genuine field-video groups and a locally trained class-aware model. [Protocol and results](construction-accuracy-round3.md).
- Construction category expansion: official multiclass datasets, model assets, licenses and measured category evidence. [Research and results](construction-category-expansion-round3.md).
- Parent: fresh baseline audit, shared compute scheduling, application/API/UI integration, final independent audit, sustained replay and regression verification.

**Host workload scope:** a later read-only2-second process sample found an unrelated, non-project Python service using about one logical CPU, alongside the configured CPU3 construction trainer (about4.12 logical CPUs during that short sample). No unrelated process was stopped or modified. Accordingly, “isolated” throughout this record means no overlapping project training/other project inference; it does not establish whole-host exclusivity. Earlier runs lack an equivalent per-process sample, so their unrelated background load cannot be reconstructed. Final paired measurements retain contemporaneous workload evidence. The sample is `output/optimization/factory-round3/host-workload-sample.json`; configured thread counts are not measured core utilization.

During the fourth factory and round-four construction fits, a **low-power interval** interrupted wall-clock progress. The read-only Windows System Power-Troubleshooter event records sleep04:55:16.7061573UTC and wake05:29:31.3114540UTC, a2054.6053s timestamp interval. Construction epoch2 batch104→105 spanned about2050s and the factory checkpoint600 tool call returned after2051.29s; both fixed fits resumed without restart or schedule changes. Raw elapsed times retain the gap. Evidence:`output/optimization/factory-round3/host-low-power-interval.json`. No power setting or process control was changed. Contended training/calibration duration, including this interruption, is not an inference FPS measurement; final speed must be measured afresh after project fits finish.

## Application changes

The new **Visible damage categories** panel reports per-category condition, measured evidence, next inspection and local model reference. Its API checks model bytes against the registered SHA256 and verifies that the registered model actually declares the capability. A research entry or a missing/mismatched model cannot become an installed detector merely because its category appears in the catalog.

Source code: `edge_runtime/construction_capabilities.py`; route `/api/construction/capabilities`. Data availability, implemented inference and field validation are distinct states. The broader Saudi lifecycle checklist remains an evidence plan, with no automatic code approval.

Experiments remain active; each frozen attempt and its failures are retained below. A calibration improvement alone does not qualify a deployment change.

## First frozen factory candidate: target failed

All six predeclared context/shallow checkpoints finished calibration before the global freeze. The fixed rule selected context step400 with10/10 localized calibration defects,0/70 normal alarms and pixel IoU.63598. Its isolated complete-profile calibration capacity was25.34 predictions/s (baseline25.55/s), so the >15/s capacity screen passed before the original comparison was scored. This is profile capacity, not shared-runtime throughput.

The independent original-mask comparison **does not meet the requested accuracy improvement**:

| Original80-view comparison | Preserved v10h | Frozen context400 |
|---|---:|---:|
| Any-pixel localized defective views |10/11|10/11|
| Defective views with pixel IoU≥.10 |10/11|9/11|
| Defective views covering≥50%GT pixels |7/11|8/11|
| Flagged normal views |5/69|1/69|
| Foreground pixel recall |54.44%|60.12%|
| Foreground pixel precision |68.73%|78.19%|
| Micro foreground IoU |.43629|.51483|
| One-to-one region recall at IoU≥.50 |5/11|6/11|

Lower false alarms and better aggregate masks coexist with a weaker localized hit. The old90.9% recall is therefore **not improved**. The candidate is retained as a failed target conjunction, not promoted as a successful fix. Raw reports and every defective/false-alarm mask remain under `output/optimization/factory-round3/` (`before-localization-audit.json`, `after-localization-audit.json`, `before-masks/`, `after-masks/`).

Paired isolated runtime replays immediately after capacity/audit used the same16FPS clock,60s duration,80 predecoded real stills,6FPS preview, metadata persistence and100ms deadline:

| Runtime measure | v10h paired before | Context400 |
|---|---:|---:|
| Captured / processed |960/960|960/959|
| Dropped / inspection errors |0/0|1/0|
| >100ms decisions |0|5|
| P50 / P95 / P99 |36.3/48.1/63.3ms|37.5/51.3/80.3ms|
| Maximum |87.1ms|209.7ms|
| Approximate processed rate, including drain |15.95/s|15.93/s|

The context candidate maintains average throughput above15/s but has a dropped observation and latency tails. Reports: `paired-before-runtime-16fps-60s.json`, `context400-runtime-16fps-60s.json`. The earlier clean before run's nine late decisions are also retained; repeated baselines reveal host variability, not permission to discard a slow result. No physical moving-line or actuation guarantee follows from these software-clock still replays.

The context trace's worst observation spent63.11ms waiting before processing,146.52ms in the profile and209.64ms from host capture to profile return. Its preprocessing interval was100.61ms versus a5.49ms median across the run; neural inference/readback was26.60ms versus23.50ms median. Another nearby observation spent43.58ms in postprocessing. These are measured wall intervals, including host scheduling; they do not isolate a GPU-kernel bottleneck or prove a particular OS cause. The dropped observation remains counted.

The optional official SDD2 test remains **unopened** while a second, explicitly versioned train/calibration-only experiment is prepared. The first winner, its selection rule, reports and masks remain unchanged. Construction's failed five-epoch fit also remains archived; it learned training examples but failed the distinct calibration source groups, so additional diverse real data and backbone adaptation are required.

## Second frozen factory candidate: recall target still failed

The separately versioned SDD2-rehearsal/FN-weighted context experiment selected step400 by its predeclared calibration rule. Calibration localized10/10 defects with1/70 normal alarms, weakest-positive mask IoU.29438, pixel recall83.86%, precision70.00% and micro IoU.61690. All three checkpoints were calibrated before the winner was frozen; the original comparison did not choose it.

| Original80-view comparison | Preserved v10h | Frozen context + SDD2/FN step400 |
|---|---:|---:|
| Any-pixel localized defective views |10/11|10/11|
| Defective views with pixel IoU≥.10 |10/11|10/11|
| Defective views covering≥50%GT pixels |7/11|9/11|
| Flagged normal views |5/69|1/69|
| Foreground pixel recall |54.44%|71.71%|
| Foreground pixel precision |68.73%|65.77%|
| Micro foreground IoU |.43629|.52218|
| One-to-one region recall at IoU≥.50 |5/11|5/11|

The lower false-alarm count and better coverage **still leave one defect missed**. This candidate is not a successful recall fix and is not promoted. Report/masks: `output/optimization/factory-round3b/after-localization-audit.json` and `after-masks/`.

An execution-order amendment was frozen before this comparison: with construction's CPU3 fit still running, a provisional contended calibration-capacity screen could precede the accuracy audit, while final isolated speed qualification remained mandatory. The paired screen measured baseline18.68 versus candidate17.41 complete-profile calls/s, mean53.55 versus57.43ms, P9566.61 versus69.10ms. These are **not isolated or full-runtime throughput claims**. Report: `factory-round3b/contended-calibration-paired-profile-capacity.json`; active-process snapshots: `provisional-workload-before.json` / `provisional-workload-after.json`. Neither read-only pause-watchdog probe established host independence; neither attempted suspension. Final isolated measurements wait for the training process to finish.

A full-resolution eight-channel refinement preflight subsequently measured only8.51 complete-profile calls/s (117.50ms mean,125.01ms P95) under the same construction-fit contention. That design failed its capacity screen before training, calibration or comparison scoring. A distinct phase-packed refinement keeps expensive feature work on the half-resolution grid while retaining2×2 raw-image pixel phases for full-resolution output; its own protocol/preflight precede any fit. Failed variants remain archived.

The phase-packed full-output version also failed its pre-fit screen:9.86 calls/s,101.41ms mean and117.99ms P95. Besides the CPU3 construction fit, a separate one-thread real-video extraction overlapped this run; it is not an isolated capacity determination. Neither full-output variant was trained or scored on calibration/comparison images. The next distinct serving variant learns native-resolution phase logits but emits their2×2 maximum on the original640×256 grid; any localization results must explicitly use that coarser serving geometry. It has its own pre-fit declaration rather than being presented as an exact-output runtime optimization.

That coarser-output variant's nonconstant TRAIN-only preflight measured14.9708 complete-profile calls/s,66.7965ms mean,74.97ms P95 and77.52ms maximum, with the construction fit and unrelated background service still active. It **does not pass a15/s qualification**. It was admitted to the fixed600-step CPU1 training experiment because removing project training contention may provide sufficient headroom; no final speed or accuracy result follows from that decision.

The first fixed checkpoint200 calibrated10/10 meaningful defects with0/70 normal alarms, weakest-positive IoU.30239, minimumGTcoverage35.22%,9/10 at least50%coverage, pixel recall86.98%, precision66.90% and micro IoU.60813. Its calibration policy is seed.999/area1/growth.8 on the original640×256 serving grid. This intermediate result cannot choose the final checkpoint: all200/400/600 results are required, training remains the unchanged600-step schedule, and original-comparison scoring waits for the global freeze and declared capacity screen. Calibration-map collection overlaps later fixed training steps and is not a speed benchmark.

The independent mask audit now also records **actual emitted runtime bbox coordinates** and one-to-one box-IoU matches against component bounding boxes derived from the original publisher masks. These supplementary box proxies do not replace pixel precision/recall/IoU, do not award one box multiple target matches, and reject invalid/out-of-bounds app boxes. Eleven geometry/localization fixtures passed; older reports without the added fields remain valid and unmodified.

An [independent static review of the active phase-max experiment](factory-attempt3c-independent-static-review.md) checked the full-resolution loss/coarser serving distinction, actual-grid bbox mapping, train/calibration/test identities, calibration shortlist and winner freeze without running NN or opening official-test contents. It found no active split/geometry/GT-to-inference defect. The actual warmstart `.pt` matches its protocol; future launches now also assert that protected SHA before loading. The policy search is explicitly a heuristic seed shortlist with a final precision/false-alarm guard, not an exhaustive proof of the best possible operating point.

## Third frozen factory candidate: lower alarms, same missed defect

The fixed600-step phase-max fit completed in557.24seconds including export, with mixed contention and no calibration feedback changing the training schedule. All three checkpoints localized10/10 calibration defects; the fixed ranking selected200 (weakest IoU.30239,0/70 alarms), ahead of400(.26225,1alarm) and600(.24799,0alarms). Protected freeze:`winner-attempt3-phase-max-freeze-before-comparison.json`; model SHA256:`ca9b90e33f1cacc8cd83ccc8e406b32faa8f45947fd2d672dc48e678b20d7318`; metadata SHA256:`6d727b95adef756399dd72a9d0a6d814c4ce2a4b72c7cf4ee15abd4a3b702bdb`.

With project fits/inference stopped, the final paired3×80 calibration capacity screen measured preserved v10h25.2651 versus frozen phase-max19.6621 complete-profile calls/s. Mean39.580/50.859ms, P9547.786/58.641ms; the new model's P9965.796ms and maximum73.597ms already preclude a guaranteed66.7ms deadline. Read-only before/after snapshots showed the unrelated service still using roughly one logical CPU; it was left running. Report:`output/optimization/factory-round3/attempt3-final-calibration-paired-profile-capacity.json`.

The subsequent independent80-view audit, including actual emitted bbox coordinates, still **fails the recall target**:

| Original80-view comparison | Preserved v10h | Frozen phase-max200 |
|---|---:|---:|
| Any-pixel localized defective views |10/11|10/11|
| Defective views with pixel IoU≥.10 |10/11|10/11|
| Defective views covering≥50%GT pixels |7/11|9/11|
| Flagged normal views |5/69|1/69|
| Foreground pixel recall |54.44%|75.40%|
| Foreground pixel precision |68.73%|61.54%|
| Micro foreground IoU |.43629|.51249|
| One-to-one mask-region matches at IoU≥.50 |5/11|4/11|
| Actual runtime box matches at derived-box IoU≥.50 |5/11|9/11|

Box geometry improved while fine mask precision fell and one actual defect remained missed. The box proxy is not substituted for pixel accuracy. All defective/false-alarm RAW/GT/MODEL/difference panels, separate exact masks, runtime boxes and source hashes are under `output/optimization/factory-round3e/`. Neither the higher calibration recall nor larger output-box match count makes the original comparison100% recall. This candidate remains a failed target conjunction.

The same frozen candidate then completed the paired16FPS/60s shared-runtime benchmark, with all project training/other inference still stopped:

| Runtime measure | Preserved v10h | Frozen phase-max200 |
|---|---:|---:|
| Captured / processed |960/960|960/960|
| Drops / inspection errors |0/0|0/0|
| Decisions over100ms |1|0|
| P50 / P95 / P99 |35.5/48.0/52.4ms|51.6/63.4/70.1ms|
| Maximum |100.2ms|84.7ms|
| Duration including drain |60.2s|60.2s|
| Processed observations / total duration |15.95/s|15.95/s|

Reports:`factory-round3e/before-runtime-16fps-60s.json`, `after-runtime-16fps-60s.json`, including per-observation traces. This run meets average≥15/s without a late decision for the candidate. It does **not** repair the failed recall condition or prove a hard deadline on future runs; earlier paired runs' tails remain reported. Identical80 real predecoded stills,6FPS preview, event persistence and100ms deadline were used. There was no camera exposure, physical moving line or actuation in this software-clock replay.

The next separately declared experiment retains this serving graph and changes TRAIN-only crop/augmentation coverage. A source-mask audit found32horizontal/1diagonal/0vertical components; it does not support a claim that horizontal defects are absent. More directly,19/33 native component bounding boxes are wider than the old256px training crop(medianwidth287); on the actual512-wide fitting grid this is20/33(medianwidth294). None exceeds512. Larger square source contexts and isotropic scales can therefore expose whole defects previously truncated, while balanced quarter-turns address the observed orientation imbalance. This is training augmentation of real source pixels/masks, not simulated inspection footage. The original comparison remains exposed development data.

The separately frozen fourth experiment uses900 fixed optimizer steps with checkpoints300/600/900, four320-square patches per batch, source-square sizes256/384/512 and balanced quarter-turns. It starts from the third candidate's calibration-selected200 checkpoint and keeps its exact serving graph. All three calibration outcomes are required before selection. Its augmented sampling, batch and training budget change together; this is not a single-factor causal ablation. Calibration may overlap later fixed CPU fitting, but cannot alter that schedule. Final paired speed and runtime still require project training and other project inference to finish. Details and immutable protocol identities are in [the factory experiment log](factory-accuracy-round3.md).

A separate [FIT-source geometry audit](factory-fit-admission-geometry-proposal.md) inspected only source headers and hashes: the preserved H/W gate[2.3,2.7] rejects1793/1865 inner-FIT SDD2 images(96.14%), although these sources were eligible for pretraining/rehearsal. All239 SDD1 TRAIN sources pass. A proposed distinct admission envelope uses the combined fitting minimum×.98 and maximum×1.02, yielding[2.4206,3.4757608696]. This does not modify any prior candidate, result or test protocol. Any future variant needs its own metadata/freeze; admitting an image does not establish correct localization or physical-domain validity. No official-test image/mask or calibration pixels were opened by this audit.

The [additive admission tool](factory-fit-admission-variant-tool.md) is implemented and checked with13 fake-byte identity/rollback fixtures; it has not yet created a real variant. It requires the exact future D freeze and FIT proposal hashes, copies NN bytes unchanged and permits only the aspect gate plus explicit provenance to differ in metadata. Runtime descriptions and serialized decisions now retain a copied `supported_height_width_ratio` and `admission_variant` alongside NN/metadata hashes. An integrated profile/event fixture verifies that identical NN identity does not hide a changed policy and that subsequent list/dict mutation cannot alter the recorded policy. Together with the existing construction provenance contract,2 tests passed in.68s. No model weights, current gates or measured pixel outputs changed through this metadata addition.

An additive official-SDD2-test declaration v2 pins this exact freeze and requires11/11 original meaningful hits, fewer normal alarms and successful sustained≥15/s runtime before opening. Because the original-recall condition failed, **the official test stays sealed**. The historical v1 is preserved unchanged; the audit now refuses an unpinned/mismatched freeze before any model execution or archive-content read. Fifteen guard fixtures passed. A different future selected candidate needs its own pre-opening declaration, never a rewrite of a prior test opening.

## Fourth frozen factory candidate: larger-context experiment regressed

The native-input phase-max graph was retained while fitting900 fixed steps on the separately declared rotated/multiscale320-square crops. The fit completed in3227.078s, including the verified low-power interval; no elapsed-time subtraction or inference-speed claim is made. Calibration300/600/900 meaningful hits were10/10,10/10 and8/10; normal flags1/70,1/70 and0/70. The fixed ranking chose600 (weakest-positive IoU.35641), not the final checkpoint. Freeze:`winner-attempt4-orientation-freeze-before-comparison.json`, SHA256`69b93c8f313c5e9b9342b45aa70fc8f6785c3b6d2ceedcfc576e51f26fc946c0`. Selected NN SHA256`93e836b11b2b2bfdb2696b557afa1e4a0e35cb31399101ebe3e396eaf5dfb916`, metadata SHA256`2786262a40742f51a8f7ebfb206231fc03bb6b7512bb0121a699d99c99d9d59e`, seed.999/area1/growth.7.

The parent independently reran both baseline and selected-candidate audits under the ongoing construction CPU fit. These are accuracy measurements, not speed qualification:

| Original80-view comparison | Preserved v10h | Frozen native-input/multiscale600 |
|---|---:|---:|
| Any-pixel / IoU≥.10 localized defective views |10/11|9/11|
| Defective views covering≥50%GT pixels |7/11|7/11|
| Flagged normal views |5/69|2/69|
| Foreground pixel recall |54.44%|69.38%|
| Foreground pixel precision |68.73%|57.72%|
| Micro foreground IoU |.43629|.46001|
| One-to-one mask-region matches at IoU≥.50 |5/11|3/11|
| Actual emitted-box matches at derived-box IoU≥.50 |5/11|6/11|
| Abstentions |0|0|

The failed model misses the existing thin-wavy case and an additional visible horizontal crack. No losing checkpoint is scored on the original comparison to replace the frozen winner. Reduced normal alarms do not repair regressed recall or mask precision. All defective/false-alarm exact masks and RAW/GT/MODEL/error panels remain in `output/optimization/factory-round3f/`; reports are `before-localization-audit.json` and `after-localization-audit.json`. It is not promoted. The next proposed experiment tests native-resolution full-width training contexts without the downsampling used by the multiscale experiment; the fourth experiment cannot establish which simultaneous training changes caused the regression.

Before the fourth candidate's original result was available, the separately authorized admission tool created a gate-only copy of its already frozen600 winner. NN bytes and seed/area/growth are identical; only the TRAIN-derived gate plus provenance differ. New sidecar SHA256`839fea3143142dbef389108adbc82acd210e959059de1c98bdc3af1bf812ccd4`; additive freeze:`winner-attempt4-fit-admission-v1-freeze.json`, SHA256`050fcceca218451312858c56261b95f3ab4abcaec8f71a79ab13f39e20affc60`. It updates no defaults, original artifacts or test declaration. This artifact does not fix the failed learned localization. **The official SDD2 test remains unopened.**

## Fifth factory experiment: native-resolution full-width fitting

The next fixed experiment keeps the already measured C phase-max serving graph and its calibration-selected200 warmstart. Its training crops are640H×512W at the actual fitting-grid resolution, with no crop resizing or90-degree turn. All33 TRAIN component bounding boxes fit this geometry (maximum height120, width452), whereas20 could not fit the old256-wide crop. Batch1 now contains the same327,680 pixels as C's batch2×640×256; preserving pixels does not make batch statistics or gradients identical. Matched180-degree/horizontal flips, bounded contrast/brightness and mild blur remain TRAIN-only.

The sampling cycle uses two SDD1 positive crops, one TRAIN-hard crop, one unbiased crop, then one SDD2 inner-FIT rehearsal alternating positive/normal. Widening a hard/unbiased crop does not turn included annotated damage into a negative: a40-batch pairing fixture verified that every role keeps its actual full-width GT. An [independent static review](factory-attempt5-native-width-independent-review.md) checked source hashes, splits, selected-component vertical containment, exact slices and the unchanged serving/calibration contracts without new NN evaluation.

Protocol SHA256`648e1252c2cba0f3617e67f9561e2524e34ca1f7c34862383859af1cef766ba6` fixes900 optimizer steps/checkpoints300/600/900, CPU1, seed20261025 and the same false-alarm/pixel-precision calibration guard and weakest-positive ranking. All three outcomes precede a new freeze and original comparison. Accuracy-only checkpoint caching can overlap the unchanged fit and construction CPU training. Final paired capacity and sustained runtime require project fitting/other inference to finish. This is an exposed-development hypothesis informed by prior failures; the comparison images never enter fitting loss or inference inputs as annotations. No successful result is inferred from the preserved serving graph or TRAIN containability.

## Fifth-candidate outcome and latest sustained replay

The fixed native-width experiment completed all900 optimizer steps in707.18 seconds including exports. Calibration selected step900 after all300/600/900 candidates were scored; the authoritative descriptive-provenance-corrected freeze is `output/optimization/factory-round3/winner-attempt5-native-width-freeze-v2-before-comparison.json` (SHA256 `f8ffd7ec9774ad143ec414afa6f18fe927a17385f271c0a05463dfcbb427e4d7`). Its NN bytes are `a49106fce0573554b1a0794f541d14501f3db60b0fdb4147deccf011a20c8c01`. No losing checkpoint was evaluated on the original comparison.

| Original exposed-development comparison | Preserved v10h | Native-width E900 |
|---|---:|---:|
| Defects with meaningful mask overlap |10/11|10/11|
| Normal images triggering an alarm |5/69|2/69|
| Defects with at least half their pixels covered |7/11|9/11|
| Pixel recall / precision |54.44% /68.73%|76.74% /57.15%|
| Mean defect-mask IoU |.4363|.4871|
| One-to-one mask-region IoU≥.5 matches |5/11|3/11|
| Actual runtime-box IoU≥.5 matches |5/11|9/11|

The baseline accuracy reference is the already executed, unchanged-code/weights/policy audit in `factory-round3f/before-localization-audit.json`; it was not rerun under a new filename. The E audit is `factory-round3g/after-localization-audit.json`. E reduces image alarms and improves coverage/boxes while losing mask precision and strict mask matches. It does **not** improve the90.9% meaningful-defect recall and is not promoted.

An additional posthoc offline grouping makes the sample's production-unit limitation explicit: all10 physical comparison items contain damage (one has two defective views). The69 normal views are unaffected views of defective products; there are **zero fully normal physical items** in this comparison. Therefore the normal-view alarm counts do not estimate false rejection of good products. Baseline/D/E each meaningfully localizes at least one defect on9/10 items; all defective views are localized on9/10,8/10 and9/10 items respectively. The unchanged eleven-view mask benchmark remains primary. `scripts/summarize_factory_items.py` hashes its source audits in `factory-round3g/offline-item-grouping-v1.json`; it runs no inference, supplies no runtime grouping/coverage guarantee and correctly records normal-product alarm rate as unmeasured/null.

With project fits, other NN jobs and data-heavy work finished/held, three passes through80 predecoded calibration images measured27.445 complete profile calls/s for v10h and21.069 for E. Mean/P95/max were36.44/43.56/61.81ms and47.46/55.51/74.46ms respectively. `factory-round3g/calibration-paired-profile-capacity.json` excludes capture queues, UI, storage and actuation; it is not physical production-line coverage.

The subsequent fixed16-input-FPS,60-second shared-runtime replays were executed sequentially and retained without retries for nicer tails:

| Model | Captured / processed | Drops | Decisions >100ms | P50 /P95 /P99 /maximum ms |
|---|---:|---:|---:|---|
| v10h |960/960|0|0|30.9 /38.6 /59.8 /93.7|
| D600 multiscale |960/958|2|25|46.0 /63.6 /132.5 /168.5|
| E900 native-width |960/960|0|14|41.7 /51.1 /114.6 /151.5|

All three reported zero inspection errors. Runs lasted about60.2 seconds. The baseline/E reports are `factory-round3g/before-runtime-16fps-60s.json` and `after-runtime-16fps-60s.json`; D is `factory-round3f/after-runtime-16fps-60s.json`. E's complete observation count does not erase its late decisions. D also drops observations. Source images are actual public stills on a software replay clock, not new physical video/product coverage measurements. Host snapshots before/after preserve ordinary host activity and an unrelated external Python service using roughly one logical CPU; that external service was never controlled. The snapshot helper samples its own CPU outside the timed runs. “Project-quiescent” does not mean whole-host-exclusive.

Read-only decomposition of the retained traces finds4/960 baseline calls,28/958 D calls and12/960 E calls exceeding the62.5ms input period. Every one of D's23 traced late results and E's14 traced late results has a profile call below100ms; queue wait contributes to the deadline failure, reaching95.88ms for D and93.39ms for E. D's other two official late counts are the stale observations dropped before inference, which have no profile-call trace. These facts do not identify a specific scheduler/process cause or permit dropping late observations from the official report. `scripts/summarize_factory_runtime_trace.py` preserves source-report hashes in `factory-round3g/runtime-trace-analysis-v1.json`; it runs no NN and never retimes results.

## Sixth and seventh factory experiments: normal-density fusion

The next bounded hypothesis uses v10h's shared early features to learn normal surface statistics from TRAIN-normal images only, then fuses signed anomaly evidence with semantic logits in one GPU graph. No result or speed is assumed. A nonconstant full-profile preflight precedes bank fitting, all fixed calibration candidates precede freezing, and independent localization plus sustained runtime remain necessary.

The F preflight subsequently passed64/64 actual neural profile calls on16 TRAIN views:23.480 calls/s, mean42.589/P9548.248/P9953.517/max56.671ms. This uses an unfitted, nonconstant dense-whitening head and is not final model performance. Actual GPU source/control raw logits match exactly on the fixed TRAIN input, original-policy masks agree, and all zero/TRAIN outputs are finite. The dummy density arithmetic differs from independent NumPy by at most.006278, within the fixed.1 guard. Standalone versus fused diagnostic fine features differ by up to.088867; fitted-head arithmetic and actual one-output parity therefore remain required before calibration.

Normal-bank fitting then used all208 actual normal TRAIN views/34,078,720 feature pixels and425,984 fixed quantile samples, with no CAL/comparison input. Its53.20-second elapsed time overlaps the authorized construction CPU4 fit and is not an inference benchmark. TRAIN centered/whitened clipping fractions are both zero, covariance is finite, and no eigenvalue needs flooring. Bank SHA256 is `4eb6b26dcb2a243708883fc4b856bd44da59078a9cd4ff4f692a2f0d691382d6`. Four independent numerical fixtures verify whitening against the inverse regularized covariance, finite singular-covariance handling, bounded zero/extreme distances and invalid-moment rejection; all pass in.75s. These fixtures do not establish device parity or defect accuracy. All nine actual exported GPU graphs, their fitted numerical checks and global freeze are still necessary.

The fitted F numerical gate **fails before any calibration**: actual GPU versus independent NumPy evidence differs by maximum.79603, exceeding the unchanged.1 limit. Source/control raw logits and masks still agree exactly, one-output candidate7 versus diagnostic fused logits agree exactly, and outputs are finite. The standalone versus full diagnostic feature mismatch remains.088867. No F CAL/original prediction is run and no guard is relaxed. A separately bounded diagnostic on the same fixed TRAIN input and zero input will distinguish head arithmetic from compiler-context feature differences before any distinctly versioned correction/refit. All F models/bank/error reports remain preserved.

The bounded diagnostic finds that context-only correction is insufficient: using the fitted graph's own actual fine features still differs by maximum.49529, whereas rounding reference constants to their actual device FP16 values reduces this to.008186. Pre-fit and fitted full-graph features match exactly. A separately protected F2 experiment therefore fits from the retained full compiler context and explicitly stores FP16-representable means, whitening, q95 and independent qspan. Covariance is correctly centered around the deployed quantized mean; whitening magnitudes round toward zero to preserve the8.0 row-L1 bound. Data, nine fusion coefficients, scoring grids and.1 error limit stay fixed. Its protocol SHA256 is `1b1a4761e697f5deec693d24b52d0be4ac431a32b2697dc3e432a71cca77965e`.

F2 normal fitting completes in81.83 seconds, explicitly overlapping construction CPU4 work. Its bank SHA256 is `211e3176df214c71e5c2450876a55657d614420bf8c28b1d293e19804a8c5038`; normal source identities are unchanged. The actual fitted GPU guard passes at maximum evidence error.007570, exact source/control logits, exact full-context features and exact single-output/diagnostic fused logits, with finite TRAIN/zero outputs. No F2 calibration input is opened before that pass. The earlier23.48calls/s receipt is explicitly reused as an unchanged-serving-architecture gate; it is not a new F2 speed execution. Final fitted-model capacity/runtime still waits for project fitting to finish. The [density experiment record](factory-normal-density-round3.md) and [independent static review](factory-normal-density-independent-review.md) preserve source/constant/error provenance and all subsequent fixed-candidate outcomes.

All nine F2 actual GPU calibrations subsequently finish. The fixed strict control comparison **rejects F2**: no eligible fusion outranks its own control. Control matches7/10 meaningful calibration views with1/70 normal-view alarms and7/10 half-coverage; fusion IDs1/2/6 also match7/10 but only6/10 half-coverage, IDs3/5 match fewer, and4/7/8 are ineligible. Every eligible point has worst-positive IoU zero. This recalibrated control obeys the new strict false-alarm budget and differs from the deployed v10h policy; it does not replace the preserved baseline. Rejected freeze SHA256 is `1454427f6eac17891ce59cc527423a1226a056f48f46ded02306d3ca3762666b`. No F2 original audit or promotion follows. A32-file executed-source snapshot and SHA manifest preserve the exact source/protocol/export versions before future shared-code edits. Any later fitted-head timing is explicitly a CAL-only failed-model diagnostic, not alternate candidate selection or an accuracy claim.

## Eighth factory experiment: bounded native-width curvature fitting

The next separate fit starts from immutable C200, strictly retaining its learned context/refiner weights and exact serving graph. Native640H×512W full-width crops use batch1/CPU1, seed20261027 and600 fixed optimizer steps with checkpoints200/400/600. It retains the C/E loss, optimizer and frozen-BN policy plus20% SDD2 inner-FIT rehearsal. Protected protocol SHA256 is `9d64a1b07154f4579f8a6d8bc47493ae9abeadbc35e8d22422d6d108e9ef8420`.

With probability.5 across positive, hard, unbiased and normal TRAIN roles, a smooth whole-patch vertical remap uses amplitudes8/16 input pixels, periods128/256/512 and uniform random phase. The source image uses linear interpolation and its actual corresponding mask uses nearest interpolation. Source coordinates stay inside observed camera pixels; no invented border is labelled normal. Positive SDD1 component bounds constrain the valid crop; infeasible geometry uses an explicitly counted identity fallback. Hard/unbiased patches keep any actual damage they include. These are declared TRAIN augmentations, not new real defect photographs or simulated evaluation videos. All evaluation inputs remain untouched.

Before any NN fit,127 coordinate/boundary/fallback checks,200 role-batch checks and396 paired containment checks across all33 original TRAIN components pass. The orientation audit32horizontal/1diagonal does **not** measure curvature or prove that missing curved examples caused the miss. The bend hypothesis follows exposed development failures without putting comparison pixels, IDs, masks or ROIs into training, mining, threshold selection or inference. The fit overlaps construction CPU4 work; its elapsed time cannot establish inference speed. All three checkpoint calibrations must finish before one fixed global freeze and independent original audit. Prior C capacity supplies an architecture receipt only; new-weight complete-profile capacity and sustained runtime remain required after all project fits finish.

## Additional official factory test protocol

The original eleven defective comparison views have already influenced earlier development. A separate, one-shot audit is therefore predeclared on **all 1,004 official SDD2 test images (110 defective / 894 normal)**, whose image/mask contents remain unopened at declaration. SDD2's official train partition was used for the preserved model's pretraining; its test partition was excluded from every fit, calibration and checkpoint choice. Physical-product identities are unavailable, so this is an official image split, not proof of independent factories or physical items.

The global six-candidate calibration selection and isolated capacity gate precede the original eighty-view audit; this added official-test audit follows them. Both v10h and the same frozen winner use the installed profile's unchanged preprocessing, quality/aspect gates and decision policy. Defective abstentions count as misses. No official-test finding may choose another checkpoint, threshold, crop or fallback. The paired report includes original-mask localization, region matching, normal alarms and abstention counts separately. Actual camera/video performance remains a separate question.

Protocol: `output/optimization/factory-round3/sdd2-test-protocol-before-opening.json`; executable: `scripts/audit_factory_sdd2_frozen.py`. A separately named, also predeclared raw-NN diagnostic bypasses only quality/aspect rejection, keeping the same pixels, model resize and frozen mask policy. This distinguishes domain rejection from model transfer; it is never credited as deployed inspection performance or used to select a fallback. Eligible-normal false-alarm rate accompanies the all-normal rate, so abstention cannot appear to be a solved false-alarm problem. The source ZIP is preserved and decoded directly by the audit; an opening journal records model/protocol hashes, including interruptions. The older acquisition manifest is not rewritten to imply that test contents remain sealed afterwards.

## Broader construction category measurements

The added author-trained DACL model supplies fourteen visible-condition channels, with actual masks evaluated against complete publisher polygons on 55 fixed real photographs. This is a **photo diagnostic**, not verified multi-category video inspection. The [category study](construction-category-expansion-round3.md) contains every class's pixel precision/recall/IoU, connected-region matches and class-absent image flags; the [fourteen-case gallery](construction-dacl-gallery-round3.md) keeps raw photos, external masks and actual inferred masks separate, including failed first examples.

The 512px model measured 2.97 FPS with boxes and 2.43 FPS with masks; the smaller 320px model measured 9.18 / 6.30 FPS, with worse localization. These are dedicated preloaded-photo adapter measurements, excluding decode, capture, shared queues and UI. They **do not meet a 15 FPS video requirement**. The main crack/spalling video experiment uses a separate compact detector, and the factory ≥15 FPS requirement remains separate.

Concrete surface loss, exposed reinforcement, rust, wet spots, efflorescence, cavities, rockpockets, weathering, residual formwork, graffiti and alligator cracking are now executable model labels with actual per-class measurements. Most broader classes remain marked **weak localization**. Inspector-marked hollow-area boundaries mean chalk/crayon markings following acoustic testing; they do not establish that a camera detects internal voids. Exposed steel and retained formwork also require construction-stage context before calling them mistakes.

Fourteen **PHOTO** presets use the first labelled example per class in a frozen ordering. They never turn stills into videos or feed publisher masks into the model. Each persistent setup note records that exact example's mask IoU and pixel precision/recall, including IoU zero. Category buttons load explicit source/model settings, and Start remains separate. Missing/modified model bytes disable the corresponding presets. An unavailable requested model cannot silently become a different installed model; an empty model choice is rejected before inference.

The server was restarted with `INSPECTFLOW_DATA_DIR=runtime-data-round3`; older `runtime-data-round2` files remain preserved. During shutdown of the old long-running server, buffered output included SQLite lock errors. This is recorded as an observed old-session issue, not claimed fixed by a new model. Fresh round-three server/API verification uses a separate directory.

## Additional real construction data and failed transfer experiments

The original-author LiteCrackSeg checkpoint was actually exported and checked against author Torch inference, then evaluated on every one of the122 predeclared camera calibration images. Its best-F1 point produced6TP/119FN/111FP at the component-box IoU.50 proxy, thin-case4/62 and10/10 negative-photo flags. A separately declared exposure-correction candidate improved counts to21TP/104FN/275FP and thin-case5/62, still flagging10/10 negatives. Neither passed the precision/false-alarm gates; neither was made a default or evaluated on fresh held-out captures. Actual input gain and raw camera exposure are shown separately. A genuine thin line is traced in one saved example, but tight semantic component boxes can fail the larger publisher-box comparison; there are no video pixel masks to claim a measured pixel fix.

The exact installed DACL512 model was also evaluated as a two-channel camera comparator on the same complete set. Its failed-gate points were crack3TP/122FN/7FP and spalling0TP/38FN/6FP, with thin-case1/62 and combined negative-photo flags1/10. This poor camera transfer is distinct from its55-photo, fourteen-class diagnostic. Per-class cutoffs are implemented in the actual adapter and all122 cached-output decodes were checked against its inference path. The original uniform threshold behavior, fourteen photo presets and photo audit were not rewritten to improve this outcome. Reports, maps and fixed RAW/GT-box/MODEL panels are retained under `output/evaluation/construction-round3/`.

A separately preregistered [joint-aware crack plus exposure-gain comparison](construction-joint-aware-shared-gain-calibration.md) then tested the original640/f32 model without changing its weights, class mapping, joint suppression or decoder. An explicit external wrapper applies the shared native-frame gain once, followed by the unchanged model with internal exposure correction off. All122 raw inputs and actual selected-cutoff inference parity were checked. Crack scoring uses75 crack-annotated exposures plus10 negative photos;37 spalling-only frames and38 unsupported spalling annotations are not credited as crack observations. At the same.25 cutoff, raw8TP/117FN/10 unmatched becomes gain8TP/117FN/16 unmatched, with3/10 negative photos flagged. No grid point passes precision80%/negative-alarm10%; the declared fallback.10 gives13TP/112FN/42 unmatched, thin5/62 and4/10 negative flags. This is another failed comparator, with no new speed, deployed policy or spalling-capability claim. Fixed RAW/native-input/publisher/model panels retain both ordinary localization and dark-night misses.

An additional twelve publisher spalling sequences were downloaded into a separate future-experiment folder:24 source camera/label files,261.5MB compressed, all CRC/SHA verified,2,380 timestamp-qualified exposures and288 preselected views. The initial manifest is immutable and **does not itself clear the data for training**. A966-known-image screen found no byte-identical images but flagged conservative similarity candidates requiring visual review; physical-site independence remains unknown. Fixed raw/publisher-box sheets reveal render/masonry/paving material in some sequences and ambiguous coating loss in another. These cannot silently become twelve independent concrete-spall sources. Acquisition/acceptance scope is documented in [the future-data record](construction-spall-future-data-r4.md); current training/calibration/held-out partitions remain unchanged by that acquisition.

The additive acceptance review subsequently inspected all121 flagged comparison pairs across16 raw/raw sheets and found no confirmed duplicate damage layout among those pairs. Its explicit recommendation is **nine concrete-surface captures/216 views** for a separate future fit, two different-material captures/48 views excluded from a concrete-only fit, and one ambiguous capture/24 views quarantined. This bounded visual review does not establish project independence or clear unseen frames. The original protocol/manifest remain unchanged; acceptance has its own version/hash and does not restart or retroactively alter the completed fit.

The distinct round-four dataset now contains832 fitting views and the same122 calibration source identities/labels. It combines those216 accepted real views with the previous616. A shared whole-frame exposure policy applies bounded gain only when grayscale p99<64; all954 separately prepared fitting/calibration input PNGs were checked against that exact function, leaving raw sources untouched. Five fitting and32 calibration inputs receive gain. Future reports and runtime decisions retain actual per-frame gain, and recording panels distinguish the raw camera frame from the native input given to the model before letterbox. This is image preprocessing, not added damage or a changed annotation.

The round-four head fit is fixed at four epochs,640 input, freeze10, batch/nominal batch4, AdamW.0005 and three CPU threads. It retains the first learned two-class head. Protocol SHA256:`53c390e9f4fd23f24c2329d10d1296c4ddfaa8f3259ba0eedef71b150fb3a532`. No wall-time cap or silent continuation is used. Its TRAIN-only batch probe measured2.55/2.19/.85 training images/s for batch4/8/16; these contended optimizer measurements are neither inference FPS nor a full-fit duration prediction. Complete calibration and serving parity follow the fit; a failed outcome cannot be promoted or credited with fresh held-out accuracy.

An [independent static/data-contract review](construction-round4-independent-static-review.md) found no class-swap, native-box normalization or repeated-gain defect. The original pretrained source actually maps0=joint,1=crack; the first fit resets only the class0 final classifier, and the current fit preserves its learned0=spalling,1=crack mapping. All1051 TRAIN boxes across832 views agree with their recorded native metadata within.00000254px;20 views are negative. There is no exact raw-file hash overlap with calibration, which does not prove physical-site independence. Prepared gain is applied once; final calibration starts from original raw sources and invokes the same serving function once. This review ran no NN, held-out predictions or calibration-pixel inspection. It validates contracts, not detection accuracy.

## Inspectable failed-model video diagnostics

### Completed R4 head and additional real low-light inputs

The four-epoch head fit completed in4193.90 seconds,4196.17 through export, including the recorded2054.605-second Windows sleep interval. Its frozen ONNX SHA256 is `33e7ab167a93c7ee9c9e80e544951ec2e61fcdaf977377ea56e321dd4fea41ad`. On all122 unchanged raw calibration sources, actual serving inference matches the cached decoder. At its failed fallback operating point (crack.45/material-loss.10), crack localization is5/125 with6 unmatched boxes; material loss is4/38 with28 unmatched boxes. The geometric-thin subset is0/62 and3/10 negative photographs trigger an alarm. A restricted common material-loss annotation scope has5 unmatched boxes; the23 additional spall-class alerts on crack exposures remain in the full-scope report. Every accuracy/false-alarm minimum fails. No heldout NN evaluation or default promotion follows.

Additional acquisition and review target a measured input gap: only5 of832 old TRAIN inputs received exposure gain, versus32 of122 calibration inputs. The separately accepted Dec31 source contributes250 genuine dark camera exposures with405 publisher crack boxes, from13 reviewed accepted sequences; seven sequences/160 proposed views remain quarantined. Its acceptance receipt SHA256 is `e3afb0d2d765238a7a0c240198a89775122746cfa1aec597aa0c26fec6a2bf7d`. All20 fixed RAW/GAIN/GT sheets were visually reviewed; conservative pair-review exclusions remain. This is future TRAIN data, not new independent test accuracy or synthetic damage.

The author-linked LCSD release contributes41 real low-light400×400 photographs, whose preserved source masks use0=background and2=crack. Their92 connected source-mask components supply derived training regions, with the original masks retained. Root reviewed fixed first/middle/last RAW/MASK/GAIN sheets and the sole flagged similarity pair. The flagged LCSD diagonal branching crack and old mottled material-loss frame have different visible layouts; no confirmed duplicate was found in that pair. This bounded review does not mean every photo was manually inspected or establish independent physical sites. Repurposing the author's validation/test stills as our TRAIN transfer data forfeits any LCSD test-accuracy claim. See [the pinned source, rights, mask-code and acquisition record](construction-lcsd-future-data.md).

A distinct combined preparation is authorized for1171 TRAIN views (990 real camera exposures,161 positive photographic views from121 unique photographs,20 negatives), retaining all122 original calibration sources/labels and the existing heldout boundaries. It combines the approved880-view broader-material version with250 Dec31 exposures and41 LCSD photos; it must not rewrite the historical832/880 versions. The unchanged gain function is applied once to separately prepared native inputs. This preparation is not evidence that cracks or spalling are fixed. A bounded TRAIN-only layout/thread probe and a separately frozen four-epoch freeze4 continuation precede complete calibration and serving parity.

Preparation subsequently completed with296 gained TRAIN inputs and32 gained CAL inputs. Its data-protocol SHA256 is `c298085a4fd7ab4b1051f432c807c73f67f0d3902948f47b5f51c5d7430882a0`. The four fixed TRAIN-only layout/thread trials select channels-last/CPU4 at2.581 training images/s, versus2.213 for contiguous/CPU4 (both batch4). This small optimizer probe is not inference FPS or a full augmented-fit prediction. The distinct four-epoch, freeze4,640-input, batch4/nominal4, AdamW.0005 fit starts from the retained learned R4 best checkpoint; protected protocol SHA256 is `f77168288653f986ddf23a8c601141767d84a172ee8e0d31f9d5059c95da625d`. It overlaps factory accuracy work, with no wall cap or hidden extension.

The [independent combined-data audit](construction-combined-lowlight-independent-audit.md) verifies585 material-loss and1016 crack TRAIN boxes, exact accepted source IDs, all quarantine exclusions, retained original masks and normalized geometry within.00000254 native pixels. Historical880 TRAIN and all122 CAL input/label contracts remain unchanged. This audit reads metadata/label text, not image/mask pixels or NN output. A separate additive evaluation receipt, SHA256 `52d24e01b187c47c57e95702fda829b259add2575ede35d96214f849416d7002`, records the unchanged minimum class/thin recall, precision, negative-alarm and serving-parity checks while the first epoch is already fitting but before postfit operating-point scoring. It does not rewrite the prefit protocol or turn50% recall into a production target.

The next whole-video development comparison is also fixed before new postfit scoring: exposed thin-crack89 plus calibration daytime crack129, material-loss75 and night `ros_crack_21`. It compares the latest frozen R4 head with the new frozen candidate on all original exposures/PTS. Protocol SHA256 is `0006e730fc4951044d395a7921796b1c226884d574ae10ed4d8a09604d19ee7b`. An explicit whitelist excludes heldout sources before pixel access. RAW camera, actual gained input, external GT and model panels remain separate. These reused/exposed sources provide inspectable development evidence, not independent accuracy credit. Video exports wait for factory calibration to release the GPU.

Six whole-source recordings now preserve the same fixed seen-training spalling source54 and calibration controls spalling75/crack129 under the historical and adapted failed models. They are available through **Include failure diagnostics and context footage**; they are not default success examples or primary category model references. Matching local raw-source presets use sequential replay, exact registered model hashes and the original per-class frozen thresholds. The registration script checks every video/report SHA before adding rows and refuses to overwrite a changed existing row: `scripts/register_construction_round3_diagnostics.py`.

On the whole seen-training source, the historical model matches65/250 publisher spalling instances with88 unmatched spalling boxes plus5 unmatched crack boxes. The adapted model matches0/250 at its frozen spalling cutoff. Both miss all77 calibration spalling instances; the fixed thin-crack control changes from4/71 to17/71 localized instances, while remaining a failed calibration model. These counts include correlated video observations, not independent defects. No threshold was lowered to make a nicer demonstration. Actual source/material uncertainty remains explicit: the scene proves publisher-labelled surface material loss, not structural concrete-cover loss or cause. Live spalling wording therefore asks the operator to verify the surface material before a concrete-specific assessment.

An independent, JSON-only [recording audit](construction-recording-count-audit.md) recomputed all eleven registered/new evidence reports' actual box IoU, unique one-to-one/class-aware matches, per-frame/per-class/global counts, unsupported exclusions and declared complete source-frame/PTS timelines. All eleven passed consistency checks;17 corruption/exclusion fixtures passed. This verifies report consistency and does not re-decode containers, inspect pixels, measure pixel accuracy or make poor detections correct. `FP` here means unmatched boxes under publisher annotation scope; partially annotated real surfaces do not establish that every such box is a verified material false alarm.

Failed models carry `diagnostic_only` in registry/API/runtime decision metadata and a failed-calibration preview banner. They cannot become the automatic damage-model default or be upgraded to an experimentally supported category merely through catalog wording. The DACL photo model remains the ordinary damage-selector default. Primary category evidence and all older raw reports remain preserved. All six new evidence endpoints returned HTTP200 with their complete frame records;60 affected construction/API/video contracts passed in3.23seconds, with the existing upstream deprecation warning, and JavaScript syntax validation passed. The extra failed-model promotion/provenance guard passed four focused fixtures in0.76seconds.

The actual dashboard was checked on both a publisher/model match and an actual miss from the complete historical seen-training spalling54 recording. The localized moment is source frame179 at4.499767s; a model spalling box matches the external publisher box at IoU.77, while an unmatched crack candidate remains visible. The missed moment is frame31 at.800064s: the publisher region is missed and the model predicts a different location. The viewer preserves the complete65/250 match result,185 misses and93 total unmatched boxes. It labels the source **SEEN TRAINING** and the model **FAILED CALIBRATION**. These are inspectable failure diagnostics, not a new successful model or independent validation. The encoded historical wording is preserved; persistent notes clarify that the material/substrate cannot be confirmed from this grayscale footage.

![Actual localized material-loss moment with failure counts and source role](C:/Users/IAM/Documents/hackathon/opencv/project1/output/evaluation/construction-round3/ui-real-spall-diagnostic.png)

![Actual missed material-loss region and wrong-location model box](C:/Users/IAM/Documents/hackathon/opencv/project1/output/evaluation/construction-round3/ui-real-spall-miss.png)

## Regression verification checkpoints

After the category API/model-identity guards, photo presets, shared publisher-timestamp helper and factory calibration-tool changes, the full runtime suite passed **208 tests in 49.24 seconds**. The sole warning is the upstream Starlette test client's deprecated AnyIO `BlockingPortal` import. These tests exercise software contracts and fixtures; they do not establish model accuracy. This run preceded final construction-model registration and final application acceptance, which are recorded separately below.

A subsequent provenance fix retains `damage_variant`, actual per-class confidence thresholds, selected construction stage and confirmation count in serialized runtime decisions/events. Previously the full profile description carried them but the result whitelist discarded them. A focused serialization-contract regression passed without loading a model. This changes decision metadata, not NN outputs or the already measured factory policy.

A later full-suite checkpoint passed **254 tests in106.45seconds**, with the same single upstream Starlette/AnyIO deprecation warning; JavaScript syntax validation also passed. This includes the additional actual-runtime bbox audit fixtures, task scope, construction worker preparation and reversible-pause mocked recovery checks. Construction's next optional per-class DACL threshold change and final model/preset registration require their subsequent affected-contract checks. Tests ran while training continued and are not performance measurements.

After the optional DACL per-class threshold implementation, the affected DACL/task-scope/provenance/capability/worker-preparation contracts passed **15 tests in0.59seconds**. The model/calibration parity audit is a separate check on actual cached NN outputs; fixture passes do not establish detection accuracy.

## Explicit construction task scope

A later regression checkpoint, after the gain, diagnostic-registration and admission-provenance changes, passed **303 tests in33.40 seconds**, with the same upstream Starlette/AnyIO warning. JavaScript syntax validation passed. This is software verification, not model accuracy or speed; later changes require their own affected checks.

The dashboard now offers **Condition to inspect** for the selected damage model. Its choices come from that registered model's capabilities: all named conditions, or one explicit operator-selected condition. This passes `damage_classes` to the installed adapter. The selected task and narrowed `supported_capabilities` remain in the profile/result metadata; excluded conditions are unassessed. The nineteen-channel DACL network still computes its normal output, and the existing class thresholds/geometry remain unchanged. The optional filter controls the review task and overlay clutter; it is **not an accuracy improvement** or a mask/source annotation input.

The default remains all supported conditions. All fourteen unfiltered photo presets, the55-photo class audit, fixed-case gallery and saved errors remain unchanged. A separate real-photo UI check on the first exposed-reinforcement case produced19 unfiltered candidate regions and is saved in `output/evaluation/construction-dacl19-round3/ui-all-conditions-state.json`; repeated observations of that still do not add independent accuracy samples. Six focused local-model fixtures verify selected/default scope and invalid-task rejection; together with construction/provenance fixtures,20 tests passed in1.19s. JavaScript syntax verification passed.

A fresh UI inference of the same real source after explicitly selecting **Exposed reinforcement** returns2 reinforcement candidates; the saved runtime metadata declares exactly `selected_damage_classes=["exposed_rebar"]` and `supported_capabilities=["surface_exposed_rebar"]`. State proof: `ui-exposed-rebar-task-state.json` alongside the all-condition proof. This is task filtering, not nineteen false positives becoming two true positives. Actual per-class accuracy remains the unfiltered publisher-mask audit. The station was stopped after both checks. The photo source badge/caption explicitly distinguishes still input from recorded video.

The stopped inspection's last real preview is restored once after a dashboard refresh; it stays labelled **STOPPED / LAST PREVIEW**, without restarting inference. This prevents a retained finding from appearing beside an empty image panel.

![Real-photo condition task, selected before local inference](C:/Users/IAM/Documents/hackathon/opencv/project1/output/evaluation/construction-dacl19-round3/ui-exposed-rebar-task.jpg)

## Completed curvature factory comparison

The fixed600-step curvature fit completes, and all three200/400/600 calibration points are scored before global selection. The unchanged strict freezer independently re-verifies source, metadata, policy and original80-source calibration identity. Selected200 model SHA256 is `7da051dabc4eefd024fbc7b0a1482b040e00087252a33a278cc086f42dbe7e81`, metadata `118f69cac35bf96add6d6b937fac03620d84bc44978c710a2eca1b4a98ff3505`, freeze `863441430b420175f114b6b4c4d47f9fcc66eeaef5b42e6436c32fb67560cef3`. Its serving point is high.99/area1/growth.8. The losing400/600 checkpoints receive no original comparison.

The coordinator's actual original-mask and runtime-box audit yields10/11 meaningful localizations,2/69 normal-view alarms,0 abstentions,8/11 half-pixel coverage, pixel recall71.5898%/precision61.3978%, micro-IoU.493688, strict mask-region matches3/11 and actual-runtime-box matches8/11 at IoU.50. Report `output/optimization/factory-round3j/after-localization-audit.json`, SHA256 `436bbee305899d3f17a94f248520e84b6fb72f4c313cda09ef0f52c617188b69`, retains every source/mask identity and every defective/false-alarm visual. The coordinator visually inspects the persistent missed thin defect: it has zero predicted pixels, not merely a weak overlap. Offline grouping still gives9/10 defective items localized; there are no fully normal items to measure good-product rejection. Construction CPU training continues during this accuracy-only execution, which is not credited as a speed measurement. The recall target remains unmet, C has fewer normal alarms, no default changes and no official SDD2 test opening follow. Final isolated capacity/runtime timing remains pending.

Before any final timing, rejected-curvature characterization is fixed to this same200 checkpoint, using three paired complete-profile passes over80 CAL views and sequential60-second baseline/curvature replays at16 inputFPS with100ms deadlines. Plan `curvature-rejected-calibration-capacity-plan-v2-before-timing.json` SHA256 `db0ed94a0b7b0175be116a818b545407e87ddfe4fa424b40d4449dd39809689c` supersedes an unexecuted wording-only v1. The separate rejected-density ID1 CAL-only plan remains unchanged. No losing checkpoint, original-image speed selection or better-tail retry is authorized by these plans. All project fits/probes/inference/data work must finish or hold first; ordinary host processes remain uncontrolled and are sampled outside timing.

## Completed project-quiescent curvature and rejected-density timing

After all construction/curvature fits, calibration jobs and the bounded pixel gradient probe exit, all agents hold NN/tests/dataset IO. The dashboard station is verified stopped. The coordinator runs the exact two previously recorded plans sequentially, retaining every first run; ordinary host services remain uncontrolled. Read-only before/after snapshots in `output/optimization/factory-round3j-capacity/` retain the unrelated Python process26076 at102.5/105.5% of one logical core plus normal desktop activity. Transient snapshot collectors run outside timing. This is project-quiescent, not whole-host-exclusive.

| Paired CAL workload (240 timed real-image calls/model) | Baseline capacity | Frozen candidate capacity | Candidate mean/P95/P99/max process ms |
|---|---:|---:|---|
| Curvature selected200 |27.8288 calls/s|21.1491 calls/s|47.2833/56.2139/59.7047/64.1620|
| Globally rejected F2 ID1 |28.9362 calls/s|26.4618 calls/s|37.7903/46.1942/53.1325/57.4604|

Both paired runs use the unchanged80 CAL sources and three timing repetitions, with predecoded inputs. These capacity timings include full profile calls but exclude disk decode, capture, queues, UI/evidence and actuation. They do not add accuracy samples. F2 ID1 is the lowest-ID eligible noncontrol chosen before timing, not a replacement for its global rejected outcome. No original comparison is opened for F2. Its CAL accuracy remains7/10 meaningful/1 normal-view alarm; capacity does not repair that failure.

| Sequential60-second CAL replay at16 inputFPS | Captured/processed | Drop/late/error | Decision P50/P95/P99/max ms |
|---|---:|---:|---|
| Curvature-paired baseline |960/959|1/8/0|30.8/66.0/85.7/156.6|
| Curvature selected200 |960/960|0/0/0|44.2/55.8/67.4/88.2|
| Density-paired baseline |960/960|0/0/0|30.0/39.1/45.4/72.3|
| Rejected F2 ID1 |960/960|0/0/0|34.0/42.6/54.5/82.9|

Every source-skip and inspection-error count is zero. All four runs retain60.2-second reported uptime; processed rates round15.9–16.0/s. Late means decision age above100ms. The first baseline's8 late counter comprises7 traced late profile returns and1 stale/drop; its27 calls exceeding the62.5ms input period and90.08ms maximum pre-profile queue wait remain visible. Six of its7 traced late returns have profile calls individually within100ms. The other runs have8/1/3 calls exceeding the input period and maximum queue waits33.05/2.49/15.50ms, respectively; none is late. This describes measured queue contribution without assigning cause to a specific host process. No tail is removed, corrected or rerun for a better result.

Source reports are `factory-round3j-capacity/{paired-calibration-capacity,baseline-calibration-runtime-16fps-60s,failed-curvature200-calibration-runtime-16fps-60s}.json` and corresponding `factory-round3i-density-capacity/` paired/baseline/failed-density-ID1 files. `fixed-window-trace-analysis-v1.json` recomputes supplementary trace counts and preserves sourceSHA. All are under `output/optimization/`. These are software replays of real public stills, **not moving production-line video or verified physical-item coverage**. Curvature's10/11 original recall still fails the requested improvement; neither model is promoted and the official SDD2 test stays sealed.

## Crack-only pixel supervision: preparation and local capacity

The separately protected preparation contains141 real TRAIN sources:41 accepted LCSD masks (explicit value2 crack/255 ignore),40 DACL crack-positive and60 verified crack-target-negative photographs. Complete saved DACL polygons are checked against pinned original metadata before negative assertions. Native whole-image gain applies once, followed by the existing384-square/RGBfloat0..255 serving convention; no camera bounding box becomes a semantic pixel mask. All projected positive masks retain foreground. Prepared manifest SHA256 is `45bde41e6d8a11762e29396895009503c111a2d0ac70e30b24633fd99c820219`.

Protected proposal/probe protocol SHA256 `1f893e2bdcf025dbb5513a680a72b052729ea66405880900549f76afd24e32e4` precedes all six actual TRAIN gradient updates. Probe receipt SHA256 `416a65b137830f73ae058f06694dee0aa4ad2b26071ce05d23533b87f15206e8` reports finite logits/loss/gradients/full updated state,326 nonzero gradient tensors each step, no frozen-encoder gradients and unchanged encoder/BN-buffer identities. Source sigmoid versus main raw-logit parity is exactly zero for both batch trials. The1,987,986 trainable and951,024 frozen parameters fit within peak RSS1,235,030,016 bytes versus the6GiB guard. Batch1 complete-step timings3.044/2.848 seconds yield.33947 TRAIN images/s; batch2 timings4.810/4.906 yield.41166. Each rate uses only two complete updates under concurrent construction CPU4 training; no inference FPS or whole-fit duration claim follows. Probe weights are discarded, no full fit starts, and no GPU inference occurs. The [independent source review](construction-pixel-supervision-independent-review.md) separates implemented preparation/probe from the proposed fixed300-step/all100-200-300 selection tooling.

An author-linked metadata inventory separately identifies102 well-lit LCSD image/mask pairs that could supply actual reserved photo pixel GT. Their pixels remain unopened. The prospective protocol pins expected normal-light class1 encoding, unknown-code quarantine, exact/near-duplicate review against all TRAIN assets, possible day/night same-scene leakage, model/policy freeze before opening and supplementary photo-only interpretation. This is not an author test benchmark or video pixel annotation. No acquisition follows from metadata availability alone.

## Construction initialization before video capture

`ConstructionProfile.prepare_worker()` performs three discarded raw-model calls on a constant initialization tensor on the inference worker. It never invokes the inspection/tracking pipeline or creates an observation, source frame or evidence image. This is engine initialization, not simulated inspection data. The existing shared-runtime ready barrier keeps capture closed until preparation succeeds; startup failures/timeouts therefore do not consume real video frames. Startup time and zero inspected warmup observations are retained under `model_initialization`.

The same initialization applies to both profiles in the later before/after video runtime benchmarks. Offline localization metrics remain unchanged. A focused test verifies all three raw calls, zero tracker/sequence changes and the exact initialization report; the construction/task fixtures passed10 tests in0.90s. Actual input, queue drops, latency and throughput still require their separate real-video runtime measurements.

## Later completed measurements: failed candidates remain visible

The preparation-only paragraphs above preserve their historical state. Subsequent actual work is summarized here; [current model status](current-model-status.md) is the entry point for the latest outcome.

- **Factory K C200+density:** all9 fixed calibrations finish and globally select candidate3. The unchanged original-mask audit gives6/11 meaningful localizations,2/69 unaffected-view alarms, foreground pixel precision65.79%/recall54.14%, mask-region IoU.50 matches5/11 and actual-box matches6/11. It is rejected; no losing candidate or alternate point is evaluated on the exposed comparison. The19.216calls/s receipt is an architecture preflight, not fitted-winner qualification. [Exact preparation, all9 results and failed original audit](factory-c200-density-preparation-round3.md).
- **Factory L normal-only teacher–student:** the actual first64 complete-profile calls measure18.388/s, but the GPU-versus-Torch maximum score error.09535 fails the protected.01 limit. No600-step fit or CAL opens. Separate CPU diagnosis matches Torch within about1.9e-6 on both ONNX Runtime and OpenVINO, and CPU half precision differs by.00759; requesting GPU FP32 still fails at.09573. All initial receipts remain preserved. A single separately declared no-Winograd GPU version is under numerical investigation, with no relaxed guard. [Implementation, source pins and actual measurements](factory-stfpm-mobile-round3.md).
- **Construction pixel supervision:** the original100-step fit fails only during metadata export. A separately reviewed restart repeats the source, seed and fresh optimizer and completes the same300-step recipe; its first100 PT/ONNX bytes match the old finite artifacts. All100/200/300 numerical and actual122-serving checks pass, but **all3 accuracy gates fail**. Their failed diagnostic points give14/19/25 crack matches out of125,560/584/631 unmatched boxes, thin2/2/4 of62 and negative-photo flags10/10,10/10,9/10. Global selection rejects all checkpoints. These diagnostic points are not adopted cutoffs. [Recovery, all checkpoints and source archives](construction-pixel-supervision-probe.md).
- **Real camera videos:** six complete original-TUT versus fixed-final300@.50 crack traces retain original exposure timing, raw and served masks, every box and complete counts. All700 saved records pass an independent source/mask/count/PTS audit. Thin89 remains0/8; day129 regresses8/71→7/71; night68/442→136/442 matches comes with unmatched boxes1,667→3,563. Separate two-class spalling75 is0/77 before and after. [Watch all real before/after comparisons and read the audit limits](real-before-after-video-results.md).
- **Actual reserved pixel GT:** all102 fixed LCSD photographs pass pre-inference screening and finish the original-versus-final300@.50 audit. Precision59.47%→65.25%, recall67.64%→43.35%, IoU.46296→.35219. All102 raw/GT/before/after panels are retained. The model regresses; these crack-positive photos do not estimate healthy-wall false alarms.
- **Finishes expansion:** a separately fitted tile texture specialist is now executable through the real app with6 fixed photo presets and research-only status. All117 test photos and all117 actual-serving comparisons are retained: crack pixel precision31.14%/recall59.71%/IoU25.73%, derived crack-region matches0/17, normal flags3/33. Two gray-stroke pixels differ after the original backend component filter; crack/normal results do not.230 additional real MBDD wall photos receive an independent source/XML/geometry/split review;181TRAIN/48CAL remain eligible after one quarantine. Related wall supervision is not yet a validated paint-only or humidity measurement model. [All datasets, rights, metrics and unsupported finish tasks](construction-finishes-expansion.md).

The app integration checkpoint passes375 software tests with1 skipped in79.16s, plus JavaScript syntax validation. Optional factory precision additions have focused independent checks; later executed source and final software verification must be recorded separately. No software-test count is evidence of defect accuracy. All official SDD2 test pixels remain sealed, and no model is promoted as satisfying the requested better-than90.9% recall with fewer alarms at15+FPS.


## CPU runtime result and software checkpoint

The fixed L5 CPU FP32/four-thread runtime passes three numerical checks (worst error2.68e-6) and actual configuration checks, but its first64 full-profile calls measure12.676/s, mean78.887ms/P9591.529ms/max94.554ms. The unchanged15FPS gate fails; no600-step fit, CAL or original accuracy audit follows. See [full CPU record](factory-stfpm-cpu4-round3.md). The complete software suite after these hooks passes400 tests,1 skipped, in73.91s; JavaScript syntax passes. This is software verification, not model accuracy.


## Actual new camera and wall-photo results

Four complete post-hoc known-hit videos now retain706 real source exposures in total. Crack129 matches1→5/71 with unmatched1→12; material-loss76 matches48→50/207 with unmatched4→35. All source/native-input hashes, lossless pixels, encoded microsecond timelines and independent report-box counts pass a separate root audit. These deliberately disclosed post-hoc illustrations do not change the failed fixed cases or provide independent improvement credit. [Watch both full pairs](real-before-after-video-results.md).

The five-class wall continuation completes12 additional epochs(total16). All category guards still fail; CAL matches crack0/19, leakage3/22, detachment8/60, corrosion1/23, bulge1/10, with0/12/1/2/0 unmatched predictions. Four appearance-only research categories and five original-photo presets are registered; quantitative humidity, paint-only identification and installed-tile alignment remain unsupported. Ten photo evidence buttons were corrected to name their actual source presets. [All wall measurements](construction-wall-five-class-plan.md).


## Factory output-copy optimization: measured but still below15FPS

The exact75-output initial graph passes all three GPU numerical checks under both runtime variants. In a project-quiescent first64-call full-profile comparison, ordinary host copies measure7.654/s (mean130.647ms/P95143.850/max148.881), shared output views11.810/s (mean84.675ms/P9597.262/max100.349). Both complete64/64. Shared views lower mean latency35.2% but neither passes15; no learned75 graph or CAL is opened. [Every first-run receipt and method](factory-stfpm-tapped75-round3.md) remains preserved.


## Completed six-feature learned runtime and author construction candidate

Factory L9 replaces the GPU score head with six raw GPU feature outputs and the same float32 CPU arithmetic, reducing output volume from 134,420,480 to 23,756,800 bytes. The BN-only initial model fails numerical parity; no speed is claimed. Under the pre-inference amendment, all three already-trained L7 checkpoints are deterministically transformed without another fit. All three then pass strict numeric/configuration checks, but every fixed 80-view calibration fails eligibility (96 raw +72 grown policies per checkpoint). Under normal alarms≤1, the best policies have 0/10 meaningful localization and 0 pixel precision. Global winner is null, so there is no fitted speed or original-comparison run. The unexecuted deployed-precision fallback and winner benchmark drafts are explicitly inapplicable; they are not additional results. [All learned results and archive hashes](factory-partitioned-learned-round3.md).

The independently sourced CUBIT author segmentation model passes strict weight transfer, CPU/GPU checks and full 122-input actual-serving parity. All accuracy checks fail: crack12/125+81 unmatched, spalling0/38, thin7/62, verified target-negative flags5/10. Five complete genuine camera videos preserve683 exposures and270 actual prototype masks, independently audited with exact timestamps. Daytime crack129 improves1/71→38/71, but thin89 remains0/8, night regresses10/442→0/442, spalling75 remains0/77 and supplementary post-hoc spalling76 regresses48/207→0/207. No overall fix or video pixel-accuracy claim follows. All23 registered comparison videos/reports pass local HTTP checks; actual UI mask+sourceframe evidence is verified. [Study](construction-cubit-author-candidate.md) and [watch all before/after recordings](real-before-after-video-results.md).

An unrelated source-routing defect in24 earlier construction presets is repaired: lossless/video paths now explicitly use source_kind=file rather than defaulting to the image reader. Matching source/model settings were verified through the real dashboard without starting inference. This changes usability, not accuracy. [Repair receipt](../output/evaluation/construction-source-type-repair/repair.json).


Final integration verification:447 tests pass,2skip,1upstreamwarning,in37.15s; five training-environment Torch arithmetic tests pass separately (.041s test time), JavaScript syntax passes, all23 comparison recordings/reports are served correctly and96 principal-document local links resolve. Optional Torch and the preserved historical pre-NN-only assertion explain the skips; no accuracy guard was skipped. The service is running locally with the inspection station stopped and the new actual crack-mask evidence paused in the browser. [Detailed verification](verification.md).
