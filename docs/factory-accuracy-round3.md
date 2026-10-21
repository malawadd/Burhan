# Factory round three: higher true-defect recall with fewer normal alarms

The preserved v10h model locates ten of eleven exposed comparison defects with pixel IoU at least 0.1, but flags five of 69 normal comparison views. This round seeks **more than 90.91% genuine localized recall, fewer than five normal false alarms and at least 15 processed observations/s on this laptop**. A favorable image flag, a box away from the real mask or a faster run that skips observations does not satisfy those goals.

The old model, reports and its known failure remain preserved. This study records new training/calibration protocols before experiments. The original 80-view comparison is already exposed development data, not a new blind holdout. Its filenames, masks and old error list are excluded from training, mining and checkpoint/threshold selection.

## Candidate protocol before fitting

Two controlled learned candidates start from the existing SDD2-transfer v10 step-300 checkpoint. Both use only the original thirty SDD1 training physical items for this fine-tune. The ten calibration items select architecture/checkpoint/seed/growth policy under a stricter **at most one normal false alarm among seventy calibration normal views**. The ten original comparison items are not opened until the final candidate is frozen. SDD2's reserved inner-calibration and official-test partitions remain excluded from fitting.

The shallow ablation retains the existing seven-block encoder and spatial decoder. The context candidate adds ImageNet-initialized MobileNetV3 Small blocks seven through eleven, extending spatial context through stride 32. A zero-initialized 96-to-40-channel residual projection initially leaves the old representation unchanged; all new context and existing spatial parameters then train. The full-shape capacity probe uses a nonzero projection so constant folding cannot remove the extra branch and understate its cost. The runtime shape remains 512×1280 BGR input / 256×640 output logits, compatible with the same explicit Intel GPU runtime.

Both candidates use 640×256 training patches. The initial capacity plan used batch size four: two positive-centered patches, one original-training hard-location patch and one unbiased training-view patch. Before fitting, measured CPU cost led to batch size two, with one positive plus an alternating hard-location or unbiased patch. This retains the same long-run 50% / 25% / 25% sampling and the same 327,680 input pixels per step as the previous batch-eight 320×128 fit. Hard locations are re-mined from training images using v10 predictions, never calibration or comparison errors. The larger field gives the learner more surrounding surface/edge context.

Training masks are projected **directly from original annotations into binary output-cell occupancy**: INTER_AREA values greater than zero mark a cell positive if any annotation area overlaps it. These are **binary any-overlap targets, not fractional-area targets**. The cells are duplicated 2×2 for even-aligned input crops. This avoids aliasing from nearest upscaling followed by pooling, which can shift very thin annotation support relative to original-resolution evaluation. Matching flips retain the output-grid alignment. Evaluation masks remain untouched original publisher annotations.

The loss averages positive softplus error, twice the average top-ten-percent negative softplus error (at least 1,024 negative pixels), plus soft Dice on positive patches. Hard-negative pixel mining concentrates background gradients on actual confusing regions instead of overwhelming them with easy background. AdamW uses encoder learning rate 0.0001, decoder 0.0003, new context 0.0002, weight decay 0.0001 and frozen batch-normalization running statistics. Seed 20261021 and checkpoints 200 / 400 / 600 are fixed before fitting. Matching flips, contrast 0.6–1.4, brightness ±30 and occasional mild blur are training augmentation only, never validation or demonstration footage.

## Calibration and freeze rules

Seed thresholds are fixed at 0.5 / 0.7 / 0.9 / 0.95 / 0.98 / 0.99 / 0.995 / 0.999 / 0.9999 / 0.99999 / 0.999999; minimum component areas 1 / 2 / 4 / 8 / 16 / 32. For each threshold, the eligible area maximizes any-mask localized recall, foreground recall and pixel IoU under the one-normal-alarm budget. The top four distinct seed thresholds proceed to connected growth thresholds no growth / 0.5 / 0.8 / 0.9 / 0.95 / 0.98 / 0.99. Growth can retain only neural foreground connected to an already-qualified high-score seed; it never creates a new flagged normal view or modifies scores.

The final checkpoint/policy rank is meaningful per-view pixel-IoU-at-least-0.1 recall, fewer normal false alarms, micro pixel IoU and foreground recall. This same calibration-only rule applies across both architectures. Complete-profile capacity must exceed 15 observations/s before comparison scoring. Original-mask audit, PR/coverage, false alarms, abstentions and complete-runtime drops/latency tails are reported independently after freeze.

Training/research/UI work may proceed through parallel agents, but CPU/GPU training and inference timing windows on the shared laptop are serialized. Startup/steady-state costs, all rejected models and failed timing trials remain visible. The study will not substitute the easier score of another dataset for the original defect problem.

Scripts: `scripts/train_factory_round3.py`, `scripts/calibrate_factory_round3.py`, and the existing independent `scripts/factory_localization_audit.py`. New artifacts/reports use `spatial_round3_*` and `output/optimization/factory-round3/`; prior round-two files stay unchanged.

## Actual capacity and training-only mining

Mining the existing v10 raw predictions on the original 239 training views took **26.06 seconds**, producing **3,788** spaced hard locations. True annotations and a surrounding margin were excluded. The report is `output/optimization/factory-round3/v10-train-hardneg.json`; no calibration/comparison pixels were opened.

The learned context graph has **876,601 parameters**. Its nonconstant full-shape GPU capacity probe measured warm median **36.09 ms**, P95 **42.19 ms**, **26.94 serial neural predictions/s** on Intel GPU.0. Graph/request initialization took **17.06 seconds**. This includes input preparation, neural inference/readback, sigmoid and raw components but excludes full-profile quality, runtime queues/media and browser work. It is a capacity screen, not a sustained 15-FPS acceptance result.

CPU batch-four 640×256 forward/backward preflight measured warm median **0.990 seconds/step**. Batch two measured **0.409 seconds/step**, so the complete 600-step context fit uses batch two with fixed checkpoints 200 / 400 / 600. Batch-size selection occurred before fitting or new calibration scoring; it was not chosen to improve a result on exposed comparison defects. Both probe reports remain available, as does `context-gpu-capacity.json`.

The first fit launch aborted before its first checkpoint because Windows backslash keys in the saved mining records did not match the trainer's slash-normalized image keys. No model/checkpoint from that failed process was saved. The trainer restarted from the frozen warmstart and fixed seed after canonicalization, with an explicit nonempty key-match guard. The captured error is preserved in `context-fit-initial-key-mismatch.txt`; two focused key/empty-pool regression tests passed. A short, single-thread construction-asset preparation job ran during part of the subsequent fit, so fit wall time is not a pure CPU-capacity benchmark. All NN capacity/speed comparisons retain exclusive compute windows.

## Fresh before measurement

The independent round-three before audit reproduced v10h's **10/11 meaningful localized defects and 5/69 normal false alarms**. The clean before runtime processed **960/960 observations at a requested 16 FPS for 60 seconds**, with zero drops, **nine observations later than 100 ms**, host P95 **56.2 ms** and maximum **138.4 ms**. This remains throughput evidence with latency tails, not a guaranteed production-line deadline. Raw reports are `before-localization-audit.json` and `before-runtime-16fps-60s.json` in the round-three output directory.

An earlier before trial ran while an RF dashboard station was active. It is retained as `before-runtime-dashboard-overlap-16fps-60s.json` and excluded from paired throughput conclusions. The dashboard station was stopped before the clean rerun and subsequent exclusive model windows.

## Context fit and preliminary calibration

The complete 600-step context fit took **294.28 seconds**, opening 239 training views / 31 defective views and no calibration or comparison images. Checkpoints 200, 400 and 600 preserve all three predetermined candidates. Loss remained finite throughout; variable hard-negative batches make individual loss values unsuitable as accuracy claims.

The first calibration candidate, step 200, locates **10/10 defective calibration views at pixel IoU ≥0.1 with 1/70 normal false alarms**. Its selected seed threshold is 0.99, minimum area 32 cells and connected growth threshold 0.9. Foreground pixel recall is **75.46%**, precision **76.78%**, micro IoU **0.61435**; nine of ten defective views cover at least half their true defect pixels. Actual raw prediction median was **41.93 ms**, P95 **48.19 ms** over these calibration images, including the first predictions but excluding graph compilation. This is preliminary calibration evidence: the remaining checkpoints, planned shallow ablation, independent comparison audit and sustained runtime gate are still pending.

Calibration caches preserve every score map and image key, including all normal views, in `spatial_round3_*-calibration-scoremaps.npz`; comparison images are absent. The full raw and growth grids remain in each calibration JSON. A subsequent calibration-only optimization precomputes each analysis cell's exact original-pixel and GT-pixel weights instead of repeatedly projecting connected-component labels to full resolution. A focused test verifies bit-identical count arrays and metric dictionaries across rectangular up/downsampling geometries. It does not alter prediction masks or NN/runtime code; the initial context calibration process used the original unoptimized metric implementation.

The first grid implementation took 81.64 seconds for context step 200. The controller then interrupted redundant enumeration to resume from cached scores using exact weighted counts. The initial resume indexed a compressed NPZ repeatedly, unnecessarily decompressing the full tensor and retaining large backing arrays; it was stopped and fixed to load that tensor once. The interrupted process also left an incomplete step-600 NPZ: the next resume successfully calibrated step 400, then raised `zipfile.BadZipFile: File is not a zip file` when opening that incomplete cache. The incomplete file is preserved with `.interrupted.npz`, and future score caches write to a temporary file before atomic replacement. Step 600 is recomputed from its unchanged NN checkpoint. These are calibration-tool failures, not hidden model trials; no model weights, score policy grids or comparison-data boundaries changed.

Step 400's completed calibration retains **10/10 meaningful defects with 0/70 normal alarms**, using threshold 0.9, area 16 and no growth. Foreground recall is **76.65%**, precision **78.88%**, micro IoU **0.63598**, with nine of ten defects covering at least half the GT. The exact cached-score grid took **14.48 seconds**, including loading and original-GT weight preparation; it is not an NN timing measurement.

Reading the interrupted process's remaining stdout subsequently recovered a completed **original-implementation step-400 result**: it had already finished this checkpoint before interruption of the step-600 cache write. Its selected policy and every printed metric match the weighted resume exactly, as do model and metadata hashes. That original pass took **57.09 seconds**, with raw NN prediction median **45.41 ms**, P95 **54.16 ms**. The recovered result is preserved in `context400-original-grid-completed-before-interruption.json`; it provides an actual-data selected-point equivalence check beyond the rectangular-grid unit test. The weighted pass's null `raw_predict_ms` correctly means it reused scores, not that the model ran instantaneously.

All three context checkpoints are now complete. The predeclared meaningful-recall / fewer-FP / micro-IoU ranking selects **step 400 within the context architecture**. This is not yet the global winner: the planned shallow ablation must complete before comparison scoring.

| Context checkpoint | Meaningful IoU≥0.1 views | Normal FP | Pixel recall | Pixel precision | Micro IoU | ≥50% GT coverage views | Seed / area / growth |
|---|---:|---:|---:|---:|---:|---:|---|
| Preserved v10h calibration | 10/10 | 3/70 | 74.00% | 75.28% | 0.59537 | 9/10 | .999 / 16 / .95 |
| 200 | 10/10 | 1/70 | 75.46% | 76.78% | 0.61435 | 9/10 | .99 / 32 / .9 |
| **400** | **10/10** | **0/70** | **76.65%** | **78.88%** | **0.63598** | **9/10** | **.9 / 16 / none** |
| 600 | 10/10 | 1/70 | 73.04% | 78.39% | 0.60800 | 8/10 | .995 / 32 / .8 |

The v10h row comes from its preserved `hysteresis_calibration_point` metadata and uses the same eighty calibration views and original GT pixel count (30,852). The context-400 improvement on calibration is therefore measurable beyond moving a threshold on the old model: normal alarms fall from three to zero, foreground recall rises by 2.65 percentage points and micro IoU by 0.04061. This does not predict an independent comparison outcome; no comparison result has selected this checkpoint.

Step 600's full calibration took **42.12 seconds**, including graph compilation, inference, saving the full score cache and exact grid enumeration. Raw prediction median was **40.09 ms**, P95 **48.17 ms**, not a complete-profile/runtime result. Its extra training did not outperform step 400, so it is retained as a rejected checkpoint. Context step 400 SHA-256 is `c34255ba62970a0df5e9bcf279bc9bc51478269466a98409801a1f6327c4c817`; its calibration-time metadata SHA-256 before descriptive geometry finalization is `f1f50719a362f4119abbc7ee5cd194be5e7cc07bff047f846be84f91eb9a963f`.

These calibration predictor timings include resize/input preparation, inference/readback, sigmoid and the initial `.999` / area-16 component call used to obtain scores. They are not isolated GPU-kernel timings and do not measure the subsequently selected policy's complete profile. Only the later isolated full-profile/runtime runs can qualify the final model's speed.

Score-cache resume verifies image order, defect labels and the checkpoint hash (embedded in new caches, or matched to the completed historical report for initial caches). It refuses stale-cache/model combinations. The complete context JSON preserves raw and grown policy grids for every checkpoint. No independent comparison views have been scored for any round-three candidate at this stage.

## Shallow ablation execution amendment, before its fit

The shallow candidate retains the exact predeclared 600 steps, checkpoint positions, seed, sampling, loss, augmentation and calibration rule. A coordinated execution amendment adds `--cpu-threads 1 --contended-training` so this CPU fit may overlap the separate construction CPU fit **only after both jobs' owners agree**. This is training scheduling, not an accuracy-data or inference-policy change. Its elapsed fit time will be labelled contended and cannot establish isolated CPU capacity. No GPU calibration, full-profile timing or runtime benchmark may overlap either fit. The one-thread count and contention flag are recorded in the shallow protocol before fitting; the completed context artifacts/protocol remain unchanged.

Both owners agreed before launch. The actual shallow command sets `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, Torch intra-op/inter-op threads to one and OpenCV threads to one:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.venv-roads-export/Scripts/python.exe -X utf8 scripts/train_factory_round3.py --variant shallow --batch-size 2 --steps 600 --save-at 200 400 600 --cpu-threads 1 --contended-training
```

Before shallow calibration, the coordinator additionally permits **accuracy-only GPU calibration** while the construction CPU fit continues, provided construction is not using the GPU. The amended command uses `--contended-calibration --cpu-threads 1`; the report and model metadata explicitly mark this execution scope. Data, mask metrics, grids and checkpoint ranking stay fixed. Any `raw_predict_ms` collected under that overlap is diagnostic contended latency and **cannot qualify capacity or compare model speed**. The final capacity/runtime before-and-after block must remain fully isolated after all fits.

Both owners agreed to this accuracy-only GPU reservation. It supersedes the earlier scheduling restriction solely for calibration; no inference/runtime speed benchmark can overlap either fit.

The shallow fit completed all **600 steps in 334.44 seconds**, with all three checkpoints saved. It used the same 239 training views / 31 defective views and opened no calibration or comparison pixels during fitting. The first 25 optimization steps took 13.14 seconds; the last checkpoint was reached at 333.49 seconds before final export. These are actual **one-thread contended fit times**, not an architecture-capacity comparison with the four-thread context fit. Every recorded loss was finite; calibration, rather than loss values, will determine the winner.

## Completed shallow control and global freeze

All shallow checkpoints finished the unchanged grid using actual GPU outputs while construction CPU training continued. These rows are accuracy results; raw prediction latencies collected in this phase are explicitly contended and are not capacity evidence.

| Shallow checkpoint | Meaningful IoU≥0.1 views | Normal FP | Pixel recall | Pixel precision | Micro IoU | ≥50% GT coverage views | Seed / area / growth | Calibrate wall seconds, contended |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| 200 | 10/10 | 1/70 | 76.33% | 75.16% | 0.60950 | 9/10 | .99 / 8 / .9 | 60.61 |
| 400 | 10/10 | 1/70 | 72.09% | 79.37% | 0.60710 | 8/10 | .995 / 8 / .95 | 58.97 |
| 600 | 10/10 | 0/70 | 74.73% | 75.96% | 0.60438 | 8/10 | .999 / 1 / .8 | 47.02 |

The shallow model's strongest eligible point by the fixed rank is step 600: its zero normal alarms precede pixel IoU in the declared rank. **The global winner is context step 400**, which matches the shallow winner's 10/10 meaningful localized views and zero alarms, then wins on micro IoU **0.63598 versus 0.60438**, foreground recall **76.65% versus 74.73%**, and precision **78.88% versus 75.96%**. The deeper context therefore improves actual calibration masks under the same data/loss/sampling recipe, rather than being accepted from architectural expectations alone. This comparison isolates the additional learned context branch between the two round-three variants; comparison with v10h also changes the target projection, patch sampling and loss together, so it is not a single-loss causal ablation.

`scripts/freeze_factory_round3.py` verified all six planned checkpoints, matching seed/data/split/grids/warmstart hashes, every model/metadata SHA and each selected policy. It then wrote **`output/optimization/factory-round3/winner-freeze-before-comparison.json`**. The protected report contains all six ranked candidates and hashes of both calibration reports/protocols. It refuses replacement by a different winner. **No original comparison image or official SDD2 test image was opened for selection.** The independent capacity gate and original-mask comparison remain pending at this freeze.

- Frozen model: `models/factory/spatial_round3_context-step400.onnx`.
- ONNX SHA-256: `c34255ba62970a0df5e9bcf279bc9bc51478269466a98409801a1f6327c4c817`.
- Final metadata SHA-256: `c1f00d1844192615dd7a2844b7d549d9b7158f901d3e98a1c952aede6aca9e9a`.
- Frozen decision policy: **score ≥0.9, minimum 16 native output cells, no growth**. Scores remain unchanged learned sigmoid values, not calibrated probabilities.

Before freeze, `scripts/finalize_factory_round3_metadata.py` corrected a descriptive ambiguity only in the six **new** sidecars. The inherited `input_width:256` historically described the analysis grid; new metadata now explicitly identifies NN input **512×1280** / NCHW `[1,3,1280,512]` and output **256×640** / `[1,1,640,256]`, plus the correct architecture. The loader had already checked these actual graph shapes during calibration and serving never used the misleading legacy field in its neural branch. All old metadata bytes and old/new hashes remain in `metadata-geometry-finalization.json` and six archived JSONs. The calibration report metadata hashes were refreshed; model hashes, thresholds, masks and metric values did not change. Prior v10h files and runtime/model-processing code stay unchanged. The existing finding method is already the generic `local_neural_surface_segmentation`.

The next gate runs **only during an isolated compute window**, before original-80 audit:

```powershell
.venv/Scripts/python.exe -X utf8 scripts/benchmark_factory_accuracy.py --before models/factory/spatial_unet_v10h_sdd2-step300.onnx --after models/factory/spatial_round3_context-step400.onnx --split calibration --repeat 3 --output output/optimization/factory-round3/calibration-paired-profile-capacity.json
```

It checks 80 unique calibration views ×3 timing repetitions with alternating before/after profile order. The frozen winner must exceed 15 serial full-profile calls/s; disk decode, capture/queue, UI and source coverage remain outside this capacity check. The independent original-mask audit and paired 60-second runtime runs then establish spatial outcomes and actual end-to-end coverage/tails. Contended calibration times above cannot substitute for this gate.

## Passed isolated complete-profile capacity gate

After both fits and construction calibration released compute, the independent paired gate completed three repetitions of all eighty calibration images: **240 measured profile calls per model**, alternating model order. The frozen context-400 model passed **25.345 serial calls/s**, above the predeclared 15/s gate, before any round-three original-80 comparison audit. Full-profile quality/domain processing produced **zero defective or normal abstentions** for either model; v10h flagged ten defects plus three normals, while the winner flagged ten defects and no normals, consistent with the fixed calibration points.

| Model | Mean profile ms | Median ms | P95 ms | P99 ms | Maximum ms | Serial profile capacity/s |
|---|---:|---:|---:|---:|---:|---:|
| Preserved v10h | 39.145 | 37.019 | 50.773 | 67.319 | 84.460 | 25.546 |
| Frozen context 400 | 39.456 | 37.281 | 51.546 | 67.297 | 101.945 | **25.345** |

This is the actual configured profile tradeoff, including the different frozen component policies, rather than an isolated architecture FLOP comparison. The extra context does not improve speed; its measured average capacity is slightly lower. Both models use the existing clear-state fast path, and the new policy omits growth; this experiment does not isolate their individual costs. All inference still runs on every accepted observation. The maximum and P99 show that average capacity does **not** establish a per-frame66.7ms or100ms deadline guarantee. Sustained source/queue/preview coverage is measured separately next. Raw report: `output/optimization/factory-round3/calibration-paired-profile-capacity.json`.

## Attempt 1 fails the original comparison goal

The unchanged frozen context-400 candidate **does not satisfy the requested improvement**. Its independent 80-view mask audit finds **9/11 views at IoU ≥0.1 (81.82%), down from v10h's 10/11**, despite reducing normal alarms from five to one. Any-pixel hits remain 10/11, so presenting that unchanged 90.91% score as improved localization would hide a regression. All source masks, model hashes and policies remain frozen; this failed result is preserved rather than used to select another exposed checkpoint.

| Original80 metric | Preserved v10h | Frozen context400, rejected for requested goal |
|---|---:|---:|
| Any-pixel localized defect views | 10/11 | 10/11 |
| Meaningful per-view IoU≥0.1 | 10/11 | **9/11** |
| At least50% GT pixels covered | 7/11 | 8/11 |
| Normal false-alarm views | 5/69 | 1/69 |
| Foreground pixel recall | 54.44% | 60.12% |
| Foreground pixel precision | 68.73% | 78.19% |
| Micro pixel IoU | 0.43629 | 0.51483 |
| One-to-one region matches atIoU0.50 | 5/11 | 6/11 |
| Abstentions | 0 | 0 |

This is a useful counterexample to aggregate-only selection: foreground masks improve overall while a low-count localized-view goal worsens. Full raw outcomes remain in `before-localization-audit.json` / `after-localization-audit.json`; all original positive and flagged-normal evidence remains available. The official SDD2 test stays sealed.

The coordinator also reran both frozen models under an isolated post-training16FPS/60-second clock with preview6FPS and deadline100ms, preserving the original before trials:

| Isolated runtime | Captured / processed | Dropped | Late>100ms | Median host ms | P95 ms | P99 ms | Maximum ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Post-training v10h | 960/960 | 0 | 0 | 36.3 | 48.1 | 63.3 | 87.1 |
| Context400 | 960/959 | **1** | **5** | 37.5 | 51.3 | 80.3 | 209.7 |

The candidate processes roughly 15.93 observations/s including drain, so average throughput exceeds 15 but a drop and five deadline misses remain. This is clocked replay of genuine public still images, not a captured moving production line or proof of physical-item coverage. The raw files are `paired-before-runtime-16fps-60s.json` and `context400-runtime-16fps-60s.json`. Both the spatial failure and latency tails are retained; the candidate is not declared production ready.

## Attempt 2 protocol, frozen before its fit/calibration

Attempt2 keeps the compact context graph and original input/output geometry. It restarts from the same original v10 warmstart plus ImageNet-initialized new context branch, **not a comparison-selected checkpoint**. The added context layers first receive300 steps on the already frozen SDD2 **inner fit only:1,865 images /197 defective**. Each two-patch batch contains one true-positive-centered patch and one randomly cropped actual normal image. The reserved466-image inner-calibration partition remains excluded from this fitting/selection; its metadata was previously used to form the stratified split. The official SDD2 test stays sealed.

Fine-tuning then runs600 fixed steps: every fifth batch replays SDD2, while the other80% use the original SDD1 thirty fit items /239 views /31 positives with the existing50/25/25 positive/hard/unbiased sampling. This rehearsal and a lower encoder learning rate reduce reliance on only31 SDD1 positive views. Pretrain AdamW learning rates are encoder1e-4/context2e-4/decoder3e-4; fine rates5e-5/1e-4/2e-4, weight decay1e-4, frozen BN statistics, gradient clipping5. The optimizer resets between stages. Seed20261022, patch640×256, batch2 and fine checkpoints200/400/600 are fixed.

The new sensitivity loss is **2×positive softplus +2×top-ten-percent negative softplus (minimum1,024 pixels) +positive-patch Tversky withFP weight0.3/FN weight0.7**. Train-only contrast0.4–1.4, brightness±30, matching flips and occasional mild blur operate on actual training inputs; evaluation images remain untouched and no damage is synthesized for evidence. The bounded64-view SDD2 cache prevents preloading several gigabytes of resized images. The one-thread fit may overlap the independently coordinated three-thread construction fit; all training wall time is explicitly contended and never a speed qualification.

The predeclared attempt2 calibration guard is **at most1/70 normal alarms and aggregate foreground precision≥0.65**. Eligible points rank: meaningful IoU≥0.1 view recall, **worst-positive-view IoU**, number of positives with≥50% GT coverage, worst-positive GT coverage, microIoU, fewer normal alarms, then foreground recall. This changes the prior aggregate-first preference explicitly before new calibration; comparison filenames/masks cannot select training examples or settings. High thresholds are.05/.1/.25/.5/.7/.9/.95/.98/.99/.995/.999/.9999/.99999/.999999, minimum areas1/2/4/8/16/32, and growthnone/.05/.1/.25/.5/.7/.8/.9/.95/.98/.99 (growth≤seed). Each high selects its best FP-eligible area by this weakest-view rank; the top six distinct high thresholds proceed to the fixed growth grid. No-growth remains available. Final eligibility applies both FP and precision guards.

The previous winner's **calibration** records already reveal this weak tail: minimum positive-view IoU **0.1463** and minimum GT coverage **14.76%**, despite aggregate microIoU0.636 and ten views nominally passing IoU0.1. These are minima from the ten existing calibration records, not an analysis of a comparison failure's identity or pixels. The new rule seeks stronger support for every calibration defect rather than allowing large masks to dominate a foreground-pixel average.

A separate **training-only** annotation geometry audit found33 true source-mask components across31 positive views. Their smallest source area is422 pixels; after exact any-overlap projection to256×640, the smallest occupied component is129 cells, median1,079, and **none is below32 cells**. Thus there is no training-label evidence that a16-cell minimum is inherently larger than complete true defects. A model's high-score seed can still be smaller than its full target, which is why the declared area grid retains1–32 rather than imposing16. These are ground-truth geometry counts, not NN accuracy or a diagnosis of any exposed miss. Report: `attempt2-train-mask-geometry.json`; script: `scripts/inspect_factory_train_geometry.py`.

New learned artifacts use `spatial_round3b_context_sdd2_fn*`; the failed attempt1 freeze, reports and model files remain unchanged. Only train/calibration data may choose the next winner. An isolated capacity gate, independent original80 mask audit and sustained paired runtime must all repeat before claiming improvement. The official SDD2 test will not open while this search remains active.

The1,865-image SDD2 count denotes the available frozen fit pool; stochastic positive/normal patch sampling does not imply every image is visited. The complete reserved466-view partition and official test remain outside the loader. A brief, separately coordinated real-photo DACL UI accuracy check may also overlap this CPU fit; all fit durations remain labelled contended and no inference throughput is measured during that overlap.

### Attempt 2 fit completed; accuracy-only calibration

The fixed run completed all **300 SDD2 pretraining steps and 600 mixed fine-tuning steps**, saving all three planned checkpoints. Pretraining optimization reached step300 at238.40 seconds; the fine phase reached step600 after488.21 seconds, or726.73 seconds from the first phase. Total wall time including final export was **728.16 seconds**. These are one-thread contended fit times alongside the separately coordinated construction CPU fit and brief UI accuracy checks, not inference-capacity measurements. Every optimization loss was checked finite. Calibration/comparison pixels remained outside the training loaders and the official SDD2 test stayed sealed.

The console prints an instantaneous loss every25 steps. Because every fifth fine-tuning batch is an SDD2 rehearsal batch, **all printed fine-phase losses happen to describe rehearsal batches**; they are neither mean SDD1 losses nor a performance metric. The actual fixed80% SDD1 /20% SDD2 training schedule remains in the script and protocol.

Five focused, non-neural tests passed: canonical Windows training keys and empty-pool guards, exact weighted original-resolution mask metrics, and the old/new selection ranks plus false-alarm/precision boundary guards. Accuracy calibration now evaluates the three fixed checkpoints on the existing80 calibration views under the predeclared weakest-positive rank. The construction owner explicitly reserved the GPU while its CPU fit continues; `--contended-calibration --cpu-threads 1` records that scope. These raw inference latencies cannot qualify speed. A protected new freeze, isolated capacity gate and independent comparison must follow before any improvement is claimed.

### Attempt 2 calibration and protected freeze

All three planned checkpoints completed the exact original-mask calibration grid. All pass the same **normal FP≤1/70 and pixel precision≥65%** guards. These are actual localized mask results, not box overlap or image labels. The ranking selects **step400** because all three localize10/10 views at pixelIoU≥0.1, then step400 has the strongest **worst-positive-view IoU0.29438**, versus0.27445 and0.24761. The earlier rejected attempt1 winner's calibration minimum was0.14631; this gain in weakest-view support is measurable on calibration, but it cannot establish the yet-unseen result of the independent reused comparison.

| Attempt2 checkpoint | Meaningful IoU≥0.1 views | Normal FP | Worst positive IoU | Worst GT coverage | ≥50% coverage | Pixel recall | Pixel precision | Micro IoU | Seed / area / growth | Grid wall seconds, contended |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 200 | 10/10 | 1/70 | .27445 | 29.40% | 9/10 | 84.08% | 67.14% | .59570 | .99 /32 /.9 | 98.78 |
| **400, frozen** | **10/10** | **1/70** | **.29438** | **34.43%** | **9/10** | **83.86%** | **70.00%** | **.61690** | **.995 /32 /.9** | **106.75** |
| 600 | 10/10 | 1/70 | .24761 | 26.29% | 9/10 | 83.53% | 68.51% | .60357 | .995 /32 /.95 | 103.40 |

All80 learned score maps per checkpoint are retained in atomic, model-hash-validated NPZ caches, and every raw/grown grid point and selected per-view mask metric is retained in `spatial_round3b_context_sdd2_fn-calibration.json`. The wall times include compile, inference, caching and policy evaluation under CPU contention. Raw predictor medians54–59ms and P9565–72ms from this phase are **not capacity evidence**.

After all three checkpoints completed, `scripts/freeze_factory_round3b.py` verified exact model/metadata hashes, thresholds and NN input/output shapes and wrote **`winner-attempt2-freeze-before-comparison.json`** before any new original80 scoring. It records the full ranked set, fixed protocol/report hashes and preserved failed-attempt1 freeze hash. The official SDD2 test remains sealed.

- Frozen model: `models/factory/spatial_round3b_context_sdd2_fn-step400.onnx`.
- ONNX SHA-256: `ef90893388e596283cf93c75923ff9b0b3479b29f15cfcd75e0090a347051629`.
- Metadata SHA-256: `2b49900d0ec6b21cd84084d9ca4e85a5dd6ae7e91eb1f9682931488a1c2b8e52`.
- Frozen policy: high seed≥0.995, minimum32 output cells, grow only connected≥0.9 regions touched by retained high seeds. Raw learned scores remain unchanged.

The model keeps the same compact context graph/input/output shapes. The isolated full-profile calibration capacity gate must exceed15 serial calls/s before the independent original80 mask audit; paired sustained runtime follows separately. GPU/CPU were explicitly released after calibration. No average-FPS or production-readiness conclusion is inferred from these contended results.

### Execution-order amendment before attempt2 original scoring

The coordinator's two read-only watchdog readiness probes failed closed because the proposed recovery watchdog remained in a Windows process job. **No suspension was attempted** and the construction CPU fit kept running. Before attempt2 original80 scoring began, the coordinator and model agent therefore recorded an execution-order amendment in `attempt2-capacity-execution-order-amendment-before-original.json`, referencing the immutable attempt2 freeze SHA. The immutable freeze itself stays unchanged.

The amended order first measures a **provisional paired complete-profile calibration gate under the recorded three-thread construction CPU fit**, requiring the frozen candidate to exceed15 serial calls/s. Passing permits the accuracy-only original80 mask audit while its graph, metadata, thresholds and selection rules stay frozen. Those contended timings **cannot qualify speed or supply the final before/after speed headline**. After construction fitting ends, the same baseline/winner must still complete the mandatory isolated paired calibration-capacity and sustained16FPS/60-second runtime qualification. This changes scheduling only; neither the selected winner nor its selection evidence changes, and the official SDD2 test stays sealed.

### Attempt2 independent comparison also fails the requested conjunction

The provisional contended complete-profile calibration gate passed: frozen attempt2 **17.4137 calls/s**, mean57.426ms andP9569.10ms, against v10h18.6755/s under the same recorded construction CPU fit. This permits the accuracy-only audit under the explicit amendment, but remains **provisional contended timing rather than final speed qualification**. Raw capacity report: `output/optimization/factory-round3b/contended-calibration-paired-profile-capacity.json`.

The exact frozen independent audit finds **10/11 meaningful IoU≥0.1 localized views**, matching v10h's90.91% rather than exceeding it. The1/69 normal alarm count is lower than v10h's5/69, and coverage improves, but the requested recall increase is still absent. The failed candidate/freeze stay preserved; no losing checkpoint is chosen from comparison results.

| Original80 metric | v10h | Attempt1 frozen context400 | Attempt2 frozen rehearsed400 |
|---|---:|---:|---:|
| Any-pixel localized views | 10/11 | 10/11 | 10/11 |
| Meaningful per-view IoU≥0.1 | 10/11 | 9/11 | **10/11, unchanged** |
| At least50% GT coverage | 7/11 | 8/11 | **9/11** |
| Normal false alarms | 5/69 | 1/69 | **1/69** |
| Pixel recall | 54.44% | 60.12% | **71.71%** |
| Pixel precision | 68.73% | 78.19% | 65.77% |
| Micro pixel IoU | .43629 | .51483 | .52218 |
| One-to-one regionIoU0.5 matches | 5/11 | 6/11 | 5/11 |
| Abstentions | 0 | 0 | 0 |

Raw masks, all positive/flagged-normal evidence and exact outcomes are retained in `output/optimization/factory-round3b/after-localization-audit.json`. These80 views are an exposed development comparison, not independent blind validation. The official SDD2 test remains sealed while further train/cal-only work continues.

## Attempt3 protocol: learned full-resolution surface detail

The next attempt changes the architecture and supervision resolution. `scripts/train_factory_round3c.py` adds a lightweight learned residual refinement branch at **512×1280 output**, before any serving mask threshold. It combines12 upsampled learned decoder features, raw input channel-mean intensity, and intensity minus its5×5 local average; a14→8 pointwise projection,8-channel depthwise3×3 convolution, pointwise8 and pointwise1 head predict a residual added to bilinearly upsampled coarse logits. It receives image pixels/features only. The final head starts at zero for fitting; a separate capacity probe uses nonconstant head weights so compiler elimination cannot hide its cost.

The source is the **immutable attempt2 calibration-selected step400 checkpoint**, not a checkpoint selected from the failed original comparison. The fixed new seed is20261023,600 optimization steps with checkpoints200/400/600, batch2 of640×256 patches, one CPU thread and the same80% SDD1 /20% frozen SDD2-fit rehearsal. Direct original training annotations project into512×1280 binary any-overlap cells without the old stride-two pooling. Validation ground truth is unchanged. Loss remains the declared FN-aware positive/OHEM/Tversky objective and matching train-only augmentation; AdamW rates are encoder/context5e-5,decoder1e-4,newrefiner3e-4, weightdecay1e-4, frozenBN and gradient clip5.

The calibration guard/rank stays normalFP≤1/70 and aggregate pixelprecision≥.65, then meaningful-view recall, worst-positive IoU,50%-coverage count, worst coverage, microIoU, fewer normal alarms and pixel recall. The high/growth grids remain the attempt2 grids. The full-resolution area grid is **1/2/4/8/16/32/64/128 native output pixels**: the output has four times as many cells, so64 corresponds in physical image area to16 old-grid cells and128 to32. Small values stay available under the same false-alarm guard. The protected protocol is `spatial_round3c_fullres_refine-protocol-before-fit.json`.

The separate training-only full-resolution annotation audit retains33 components: minimum478 output pixels, median3,950, maximum6,378; none is below128. Source annotations remain the same33 components/minimum422 source pixels. These are native output-cell counts from direct projection, not inferred seed areas or evidence about comparison misses. Raw record: `attempt3-train-mask-geometry.json`.

Serving now validates exactly two allowed graph-output shapes: the old `[1,1,640,256]` or new `[1,1,1280,512]`, with the unchanged `[1,3,1280,512]` input. Bounding boxes/grid telemetry derive from actual score-map shape; old-grid arithmetic stays identical. The hardware-free coordinate test covers both grids with the same expected source box. Eleven focused geometry, growth, metric and selection tests pass. There is no frame skipping, preprocessing, score mutation, gate or threshold change for any prior model.

The nonconstant full-shape preflight must pass before committing this fit. A provisional contended complete-profile calibration gate may again permit only the accuracy audit if construction training is active; this order is declared in the attempt3 protocol before fitting. Final isolated paired complete-profile capacity and sustained runtime remain mandatory for any ≥15FPS claim. Neither training nor selection opens official SDD2 test pixels or uses comparison failure filenames/masks.

### Full-resolution refinement A rejected at capacity screen

The first nonconstant graph has876,882 parameters, only281 more than compact context, but **parameter count did not predict runtime**. Four contended single-thread patch forward/backward probes took1.902/1.362/1.784/1.543 seconds. A separate actual full-profile preflight used16 genuine TRAIN views,64 calls, zero GT reads and no comparison images. It measured **8.51 calls/s**, mean117.501ms,P95125.008ms,P99130.265ms,max132.961ms. Startup including compilation/warmup was19.730 seconds. Under the recorded construction CPU3 fit, mean neural stages were preprocessing12.257ms, GPU/readback55.540ms and CPU postprocessing39.309ms. These are contended capacity-screen measurements rather than isolated qualification; they are sufficiently poor to reject this candidate **before its planned600-step fit or any calibration**.

The unchanged protocol, nonconstant ONNX and exact reports remain in `spatial_round3c_fullres_refine-{training-capacity-probe,fullprofile-preflight}.json`. Synthetic initialization tensors are never accuracy evidence. No trained refinement-A checkpoint exists and no comparison result is attached to it.

### Refinement B protocol before its capacity screen

`scripts/train_factory_round3d.py` preserves the same real data, source checkpoint, full-resolution supervision,600 steps/checkpoints and calibration guard/rank, but moves learned filtering back to the256×640 feature grid. Exact2×2 intensity/local-contrast pixel phases are packed into eight channels, concatenated with12 decoder channels;20→8 pointwise,8-channel depthwise3×3,pointwise8 and four phase-logit outputs are shuffled to full resolution and added to coarse logits. This preserves subcell raw-pixel information while removing the expensive full-resolution12-feature interpolation and8-channel filtering. Its distinct protected protocol/artifact prefix is `spatial_round3d_packed_refine*`; the failed A graph/protocol remain untouched. A separate nonconstant preflight must establish plausible headroom before training. No inference is scheduled during the construction owner's intervening GPU accuracy window.

### Packed full-output refinement B screen and phase-max serving C

The nonconstant B graph has876,957 parameters. Its four CPU patch forward/backward probes took1.253/.783/.837/1.144 seconds. The actual16-TRAIN-view/64-call full-profile screen measured **9.86 calls/s**, mean101.406ms,P95117.992ms,P99120.143ms,max120.725ms, startup30.173 seconds. Mean neural stages were preprocessing12.137ms,GPU/readback39.479ms and CPU postprocessing39.633ms. Packed features reduce the observed GPU/readback cost versus A, while the4× output map remains costly on the CPU. B is therefore not fitted at this stage.

After the screen, the coordinator reported an additional one-thread extraction of12 video/label pairs (about261.5MB compressed) overlapping the construction CPU3 fit. The SHA-linked `packedB-preflight-workload-supplement.json` records this extra workload without editing the raw timing/model/protocol. These contended screens **do not prove that isolated15FPS is impossible**; they prevent spending a lengthy fit on currently poor measured headroom. Raw B reports remain `spatial_round3d_packed_refine-{training-capacity-probe,fullprofile-preflight}.json`.

Refinement C (`scripts/train_factory_round3e.py`, prefix `spatial_round3e_phase_max_refine*`) keeps B's learned raw-pixel phase features and **full-resolution training loss**, but its exported serving graph takes the maximum logit in each2×2 output cell and emits the original **256×640 grid**. Sigmoid is monotonic, so this is the largest learned subcell score, not a forced score increase or calibrated probability. The model may recognize subcell detail, but its displayed/validated localization is explicitly coarse at the original grid. The area grid returns to1/2/4/8/16/32 old-grid cells. All other fixed training/data/guard/rank/checkpoint choices remain the same; the distinct protocol is recorded before its nonconstant capacity screen and before any calibration. The change has **not** been presented as exact-mask equivalence with the full-output graph. C must itself pass a new capacity screen and calibration, and final isolated speed/independent spatial evidence remain mandatory.

A subsequent coordinator read-only host sample also found a non-project Python service(PID26076) using about99% of one logical CPU. It was not controlled or stopped. The configured three-thread construction fit used about412% of logical-CPU capacity in that short sample; a thread setting is not a guarantee of exactly three continuously occupied logical CPUs. `host-workload-sample.json` preserves the measurement. All earlier/future references here to **isolated** mean project fits and competing project inference are quiescent, rather than exclusive ownership of the entire host. Final paired timing must snapshot actual background workload and state this practical scope.

### C capacity screen supports fitting, without speed qualification

The nonconstant C model has876,957 parameters, unchanged from B. CPU patch backward probes took.976/.761/1.227/1.198 seconds. Its genuine16-TRAIN-view/64-call full-profile screen measured **14.9708 calls/s**, mean66.7965ms,P9574.9683ms,P9976.7280ms,max77.5191ms; startup21.0874 seconds. **This is below15/s and is not a passed speed qualification.** It is sufficiently close under active construction fitting/documented host contention to justify the fixed600-step learned experiment, whereas A/B had much poorer measured headroom. The final selected, actually trained graph must still exceed15/s under the stated project-quiescent qualification and sustained runtime; failure remains possible. Raw records are `spatial_round3e_phase_max_refine-{training-capacity-probe,fullprofile-preflight}.json`. No GT was read and no comparison was scored for these capacity screens.

After GPU release, C's one-thread CPU fit starts from the protected protocol/warmstart. It keeps the fixed600 steps and all three checkpoints; its wall time remains contended. Other agents may use the GPU for separately coordinated accuracy work while this CPU fit runs. No serving model or old benchmark is replaced by the untrained capacity probe.

Independent static review confirmed the active warmstart is the protocol's step400 source. The initializer now additionally asserts that the dynamically frozen winner's actual `.pt` SHA equals the protected protocol's warmstart SHA before loading; this hardening does not interrupt or alter the already-correct active fit. The calibration search is explicitly **heuristic**: seed-area eligibility applies the normal-FP guard, then the best area per high/top six highs enter growth search; only final candidates apply both the FP and≥.65 precision guards. It does not claim an exhaustive jointly guarded optimum across every possible high/area/growth combination. The rules and thresholds remain unchanged.

The coordinator approved an accuracy-only pipeline before C calibration: each immutable checkpoint can be scored after its export completes while the **unchanged fixed600-step CPU fit continues**. No calibration result changes data, steps, learning rates or later checkpoints. Resume-cache verifies checkpoint hashes/order and all three results must finish before the global freeze; no original comparison opens early. `--contended-calibration --contention-note ...` records the actual active factory CPU1 fit. Construction's CPU fit ended during C fitting, so overall fit time has mixed background contention; separately coordinated construction GPU accuracy jobs are serialized between factory calibration windows. No pipeline latency qualifies final capacity or runtime.

C checkpoint200's initial pipeline result passes both guards:10/10 meaningful views,0/70 normal alarms,worst-positiveIoU.30239,minimumGTcoverage35.22%,9/10≥50%coverage,pixelrecall86.98%/precision66.90%/microIoU.60813, atseed.999/area1/growth.8. This is a calibration result **before the remaining fixed checkpoints finish**, not a chosen final winner or a comparison improvement. The single-cell high seed is followed by actual connected learned foreground; the reported original-mask IoU/coverage measures the resulting mask rather than rewarding that one seed alone. The raw learned score map stays unchanged. The200NN phase completed/cache was written before the GPU was handed to construction; only its CPU policy search continued during the next construction GPU accuracy window.

The fixed C fit completed all600 steps and all three exports without calibration feedback. Step600 finished at555.808 seconds and total wall time including final export was **557.236 seconds**. Every optimization loss was finite. All printed25-step instantaneous losses again correspond to the every-fifth-batch SDD2 rehearsal schedule; they do not summarize SDD1 or estimate validation accuracy. The first part overlapped construction fitting; later parts overlapped serialized GPU accuracy jobs and CPU-only calibration, so this is mixed-workload training wall time, not a speed/architecture comparison.

Checkpoint400's independent calibration also passes both guards but ranks behind200:10/10 meaningful,1/70 normal alarms,worstIoU.26225,minimumcoverage29.14%,9/10≥50%coverage,pixelrecall86.06%/precision66.48%/microIoU.60017, atseed.999/area1/growth.9. The unchanged selection rule, rather than the latest training step, determines the eventual winner after600calibration. All earlier model files/reports remain untouched.

### All C checkpoints completed; step200 frozen before comparison

All three planned C checkpoints completed actual GPU calibration. Their top points are:

| C checkpoint | Meaningful IoU≥0.1 | Normal FP | Worst positive IoU | Worst GT coverage | ≥50% coverage | Pixel recall | Pixel precision | Micro IoU | Seed / area / growth | Grid seconds, accuracy-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| **200, frozen** | **10/10** | **0/70** | **.30239** | **35.22%** | **9/10** | **86.98%** | **66.90%** | **.60813** | **.999 /1 /.8** | **101.31** |
| 400 | 10/10 | 1/70 | .26225 | 29.14% | 9/10 | 86.06% | 66.48% | .60017 | .999 /1 /.9 | 74.79 |
| 600 | 10/10 | 0/70 | .24799 | 26.59% | 9/10 | 86.89% | 66.76% | .60646 | .995 /1 /.5 | 82.05 |

All pass the fixed normal-alarm and pixel-precision guards; step200 wins the next fixed criterion, worst-positive IoU. Later steps are not automatically preferred. The final600 scoring happened after factory fitting/construction inference ended, with possible brief data preparation and ordinary host service recorded; its raw latency still does not qualify speed. Immutable score-map caches and all raw/grown policy rows remain in `spatial_round3e_phase_max_refine-calibration.json` and checkpoint NPZs.

`scripts/freeze_factory_round3e.py` verified all three checkpoints, model/metadata SHAs, protected training protocol, search grids/rank, exact selected-point records, thresholds and NN input/output shapes. It wrote **`winner-attempt3-phase-max-freeze-before-comparison.json`** before original80 scoring. It records earlier failed-attempt freeze hashes without replacing them.

- Model: `models/factory/spatial_round3e_phase_max_refine-step200.onnx`.
- ONNX SHA-256: `ca9b90e33f1cacc8cd83ccc8e406b32faa8f45947fd2d672dc48e678b20d7318`.
- Metadata SHA-256: `6d727b95adef756399dd72a9d0a6d814c4ce2a4b72c7cf4ee15abd4a3b702bdb`.
- Frozen policy: seed≥.999, minimum1 native256×640 cell, connected growth≥.8. The resulting mask is judged against all original GT pixels.
- Serving resolution:256×640, explicitly max-pooled learned phase logits; training supervision resolution:512×1280.

### Final C complete-profile calibration capacity passed

After **all project fits, competing inference, exports and CPU policy/data jobs exited**, the final paired gate measured all80 calibration views three times, alternating model order:240 timed profile calls/model. Both models had zero quality/domain abstentions. Baseline flagged10 positives plus3 normals; frozen C flagged10 positives and zero normals. All model/policy values remained fixed.

| Frozen model | Mean profile ms | Median ms | P95 ms | P99 ms | Maximum ms | Serial full-profile calls/s |
|---|---:|---:|---:|---:|---:|---:|
| v10h | 39.580 | 38.188 | 47.786 | 56.859 | 64.339 | **25.265** |
| C200 | 50.859 | 50.085 | 58.641 | 65.796 | 73.597 | **19.662** |

C is slower than v10h, but passes the mandatory **>15 serial complete-profile calls/s** gate before original comparison. These predecoded profile calls exclude capture, decode, queue, UI/evidence and actuation. The73.6ms maximum also precludes claiming every call meets a66.7ms deadline. Actual sustained16FPS/60-second runtime and independent original-mask/source-bbox outcomes remain the coordinator's next gates; this result does not establish either.

`attempt3-final-calibration-paired-profile-capacity.json` includes SHA-linked read-only workload snapshots `attempt3-capacity-host-{before,after}.json`. The non-project Python servicePID26076 remained active at approximately101–106% of one logical CPU; ordinary app/server processes were also active. Snapshot sampler processes ran/exited outside the measured block, and Windows' clearly labelled System Idle Process is idle capacity rather than an active workload. No process was paused, stopped or changed. The snapshot helper initially failed in the app environment because `psutil` was absent; rerunning through the existing Torch-export environment succeeded without installation. **Project-quiescent does not mean whole-host-exclusive.** GPU/CPU were then explicitly released to the coordinator; no further model fitting or inference is scheduled by this agent during its audits/runtime.

### C passes sustained throughput but still fails recall improvement

The coordinator's frozen original80 audit again finds **10/11 meaningful IoU≥0.1 views and10/11 any-pixel hits**, unchanged from v10h. Normal false alarms are1/69 versus5/69;≥50% GT coverage is9/11 versus7/11. Pixel recall improves to75.40%, precision falls to61.54%, and microIoU is.51249. Thus pixel aggregate/coverage/normal-alarm gains still **do not meet the requested recall increase**. No losing C checkpoint is selected from these exposed comparison outcomes.

The new secondary source-coordinate box audit finds9/11 actual emitted boxes matching mask-derived GT boxes atIoU.5, versus5/11 baseline. These GT boxes are derived from publisher-mask components, not independently supplied author box annotations. This improvement cannot replace the primary mask result: one-to-one mask-regionIoU.5 recall is4/11 versus5/11 baseline. All actual predicted masks/boxes, failures and source hashes remain in `output/optimization/factory-round3e/after-localization-audit.json`.

The paired project-quiescent16FPS/60-second replay then completed:

| Runtime model | Captured / processed | Dropped | Late>100ms | Errors | Median host ms | P95 ms | P99 ms | Maximum ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v10h | 960/960 | 0 | 1 | 0 | 35.5 | 48.0 | 52.4 | 100.2 |
| Frozen C200 | **960/960** | **0** | **0** | **0** | 51.6 | 63.4 | 70.1 | 84.7 |

C therefore sustains about**15.95 processed observations/s including60.2-second drain**, satisfying the15FPS throughput target in this replay. Its slower median/P95 and missed66.7ms periods remain visible; zero misses of the configured100ms deadline are not a hard-real-time guarantee. These are repeated public still observations, not a captured moving factory line or proof of physical-item coverage. Before/after JSONs and full traces remain in `factory-round3e/{before,after}-runtime-16fps-60s.json`. C remains rejected for the requested conjunction because localized-view recall is still10/11. The conditional fresh officialSDD2 test has **not opened**.

## Attempt4 / D: training orientation and full-width context coverage

Before this fit, `inspect_factory_train_orientation.py` audited only the original thirty training physical items:239 views,31 positive views and33 annotation components. The PCA major-axis bins are**32 horizontal,1 diagonal,0 vertical**, with31 components having elongation≥3. This does **not** support claiming that horizontal defects are absent. It shows a strong axis imbalance that flips alone cannot exchange. PCA is a coarse geometry summary, particularly for curved defects.

The more direct crop finding is that **20/33 components have GT boxes wider than256 pixels in the actual512-wide training input**, median294 pixels. All fit within512. In original source coordinates the conservative count is19 and median287; the reported20/294 explicitly accounts for the actual input resize. Prior640×256 patches therefore cannot show whole horizontal-defect context for many real training components. New square sources include width384/512 and may show that context while varying scale. This rationale comes entirely from TRAIN annotations, not a new examination of comparison failure pixels or filenames. `attempt4-train-orientation.json` contains all calculations and its SHA is pinned before fitting.

`scripts/train_factory_round3f.py` keeps the already-qualified C serving graph and starts from **C's immutable calibration-selected step200**. It uses seed20261024,**900 fixed steps/checkpoints300/600/900**, four320×320 patches/batch, one CPU thread, frozenBN, the same optimizer/loss/clip settings and20% frozen SDD2-fit rehearsal. SDD1 batches contain two positive-centered, one train-mined hard-location and one unbiased real patch; SDD2 batches contain two positive-centered plus two actual-normal patches. Source-square sides256/384/512 are resized isotropically to320, with matching image/annotation transforms. Matching uniform0/90/180/270-degree turns and horizontal flips add orientation coverage; train-only contrast.25–1.5,brightness±40 and15% mild blur alter training inputs only. There is no generated evaluation damage. The full square-patch mask loss remains real GT occupancy without validation-mask dilation.

The protected `spatial_round3f_orientation_multiscale-protocol-before-fit.json` pins source PT SHA **`ff3b5571295e80978b4be873b6cce24b7cb947ae25415959a4121337dfe22112`**, prior C freeze SHA and TRAIN orientation-audit SHA **`224f80252afa3749a9164ef21836deb5f58dbb239a217cff42b424e68b74dad5`**. Its unchanged calibration guard/rank and high/area/growth grids are fixed before fitting. Pipeline calibration may evaluate completed immutable checkpoints while the fixed900-step fit continues without feedback; all three results are required before the new global freeze.

D's **new-prefix-only** cache validation additionally pins ordered original calibration-image/maskSHA256, native image/mask geometry and class, alongside modelSHA/order. A changed source invalidates a cached map. Its freeze verifies the same80-view/10-defect source identity across **all three** checkpoints, including any rejected checkpoint, plus matching sidecar protocol/search/point fields. Old C caches, reports and model files stay unchanged.

Because serving operations/shapes are unchanged from the graph already measured19.66 full-profile calls/s and15.95 sustained observations/s, the coordinator predeclared an **accuracy-first original audit after all calibration/global freeze** if a construction fit is still active. This does not qualify D speed: new weights and data-dependent postprocessing must still pass final project-quiescent paired complete-profile capacity and sustained runtime≥15FPS before promotion. OfficialSDD2 test remains sealed until the requested conjunction is achieved. D starts with an agreed one-thread CPU fit alongside the construction experiment; training/calibration timings are explicitly contended.

### D progress, initial calibration and independently confirmed host sleep

D's300 checkpoint reached393.696 optimization seconds. Its actual GPU calibration passes both guards:10/10 meaningful views,1/70 normal alarms,worst-positiveIoU.31844,minimumGTcoverage35.34%,9/10≥50%coverage,pixelrecall84.39%/precision68.58%/microIoU.60859, atseed.999/area4/growth.9. Calibration walltime92.382 seconds is contended/non-qualifying timing. Its pinned ordered80-source identity SHA is **`7b09d8cbf39129dad144fcad6ecbad0ade0bb19e39de6e8fbdfc634eb6ed6d07`**. This is only the first planned checkpoint;600/900 and the global freeze remain mandatory. No original comparison has opened for D.

Independent static review found no material paired-transform, square-target, warmstart or cache/freeze issue. It identified an operational CLI risk: inherited default200/400/600 steps. Before D calibration, the CLI was hardened so orientation-balanced defaults to its protected300/600/900 and rejects unplanned steps, while old variants retain their prior defaults. The search/data/model logic is unchanged. Source identities remain fully stored in reports/cache; console output now shows only their digest/count to avoid printing80 records per checkpoint.

During the600 calibration launch, a requested10-second-yield tool call unexpectedly remained pending for **2,051.292 seconds**. Raw fit progress jumped from625 at811.536 seconds to650 at2,895.997 seconds, a2,084.461-second interval for25 steps; subsequent steps resumed. The coordinator then independently read Windows **Microsoft-Windows-Power-Troubleshooter event1**, confirming **SleepTime2026-10-20T04:55:16.7061573Z toWakeTime05:29:31.3114540Z**, a2,054.605-second low-power interval. The read-only event evidence is `host-low-power-interval.json`. No power settings/processes were changed.

All raw tool/training/calibration elapsed time and the interruption remain preserved. The fit continues the same fixed900 steps without restart or calibration feedback. **No sleep interval is subtracted to manufacture a speed claim**; these already-contended fit/calibration walltimes cannot qualify inference throughput. Final paired speed/runtime qualification must happen separately after fitting and after project work is quiet. The600 score-map cache completed after wake, and its remaining policy search is CPU-only.

The600 child process itself reports94.560 seconds of calibration processing after wake; the2,051-second value belongs to the stalled **tool-launch residence**, not an invented2,145-second child-processing duration. The raw clocks are kept distinct. The fixed fit eventually completed900 at3,224.991 seconds, total including export **3,227.078 seconds**, with the verified host low-power interval included and all optimization losses finite. No checkpoint, schedule or data was altered in response to calibration or sleep.

### All D checkpoints evaluated and step600 frozen

| D checkpoint | Meaningful IoU≥0.1 | Normal FP | Worst positive IoU | Worst GT coverage | ≥50% coverage | Pixel recall | Pixel precision | Micro IoU | Seed / area / growth | Cal child seconds, contended |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 300 | 10/10 | 1/70 | .31844 | 35.34% | 9/10 | 84.39% | 68.58% | .60859 | .999 /4 /.9 | 92.38 |
| **600, frozen** | **10/10** | **1/70** | **.35641** | **43.30%** | **9/10** | **87.27%** | **65.23%** | **.59563** | **.999 /1 /.7** | **94.56** |
| 900, rejected by fixed rank | 8/10 | 0/70 | 0 | 0% | 8/10 | 72.45% | 77.56% | .59894 | .995 /32 /.8 | 99.73 |

All three checkpoints satisfy the fixed final FP/precision eligibility guard, but900 loses two calibration defects. Thus more training is not automatically better. Step600 wins the declared meaningful-view-recall/worst-view-IoU rank over300; its lower aggregate precision/microIoU tradeoff remains explicit. The final choice is made from calibration only, before any new original80 audit. The combined D regime changes crop/context/orientation/scale/contrast, batch size and optimization budget together; it is **not** a single-factor causal test of rotation alone.

`winner-attempt4-orientation-freeze-before-comparison.json` verifies every planned checkpoint, exact model/sidecar/point/protocol/grid/rank identity and the unchanged ordered80-image/10-defect source identity across all three reports. Its frozen winner is:

- Model: `models/factory/spatial_round3f_orientation_multiscale-step600.onnx`.
- ONNX SHA-256: `93e836b11b2b2bfdb2696b557afa1e4a0e35cb31399101ebe3e396eaf5dfb916`.
- Metadata SHA-256: `2786262a40742f51a8f7ebfb206231fc03bb6b7512bb0121a699d99c99d9d59e`.
- Policy: seed≥.999, minimum1 native256×640 cell, connected growth≥.7; raw learned scores unchanged.
- Source/calibration identity SHA: `7b09d8cbf39129dad144fcad6ecbad0ade0bb19e39de6e8fbdfc634eb6ed6d07`.

The coordinator now owns the **accuracy-first** independent original-mask and actual-runtime-box audit under the predeclared schedule. Final new-weight/policy capacity and sustained≥15FPS still require project-quiescent measurement after construction fitting ends. No officialSDD2 test has opened, no existing model/report was replaced, and no more NN runs occur from this agent during the coordinator's audit.

Supplementary independent TRAIN-only boundary evidence also prevents an unsupported explanation:30/33 GT components touch the full source image's **right** edge and31/33 are within32pixels of it; none lies within32pixels of the other edges before augmentation. This is directional concentration, **not generic edge scarcity**. Only2/33 have32real observed pixels of context all around; larger D crops cannot invent source context outside the camera image. The independent report `docs/factory-train-boundary-context-audit.md` and raw `attempt4-train-boundary-context-audit.json` do not select padding, alter D or establish the cause of a comparison miss. No boundary-model exception or filename/ROI rule was introduced.

### Frozen D fails the original development comparison

The independent frozen D600 audit regresses to **9/11 any-pixel and meaningful IoU≥.1 views**, versus10/11 for the fresh v10h baseline. Normal false alarms fall from5/69 to2/69, but this does not meet the requested conjunction. D covers≥50% of GT in7/11 views; pixel recall is69.38%, precision57.72%, and microIoU.460005. One-to-one mask-regionIoU.5 recall is3/11, and actual emitted source-boxIoU.5 recall is6/11. No view abstains. These results cannot support a recall improvement or compensate for weaker primary masks through box counts. Full before/after reports and all defect/normal-FP panels are retained in `output/optimization/factory-round3f/`.

No losing D300/900 checkpoint is scored on the original comparison, no D model is promoted, and officialSDD2 test remains sealed. D's new-weight/policy speed record remains separate from its failed accuracy result.

## Attempt5 / E: native full-width context without patch downsampling

Before fitting, the existing TRAIN-only boundary/geometry audit establishes that all33 real components have target-grid box width≤452 and height≤120 pixels (width min/median/max32/294/452; height16/55/120). All fit a **640H×512W crop** of the512×1280 training input. Prior C's256-wide crop cannot contain20/33 boxes, while D's512→320 square resize reduces fine features. This motivates retaining full width **and native serving-input pixel scale**, without claiming that either factor caused any exposed comparison miss.

The approved E protocol starts from immutable **C200's calibration-selected PT**, never a losing D checkpoint. It keeps the identical phase-max serving graph, loss/optimizer/frozenBN and selection guards/rank, with seed20261025, one CPU thread, **900 fixed steps/checkpoints300/600/900**. Batch1×640×512 contains the same327,680 image pixels as C's batch2×640×256. Per five steps, original SDD1 TRAIN sampling is positive, positive, TRAIN-mined hard location, unbiased real patch, then SDD2 inner-FIT rehearsal alternating positive/actual-normal. Every crop retains its actual corresponding GT, including wider hard/unbiased crops that contain defects; those patches are not relabelled normal.

The source image/mask is first projected to the existing512×1280 input grid, then sliced vertically with **no patch resize and no90-degree transpose**. Matching0/180-degree turns and horizontal flips, contrast.4–1.4, brightness±30 and15% mild blur use C's train-only photometric range. Point-centered vertical jitter≤192 pixels plus≤120-pixel TRAIN component height leaves a selected SDD1 component within the640-pixel crop; full source width is retained. This containment statement does not cover arbitrary SDD2 components. No outside-camera context or synthetic evaluation defects are invented.

The new-prefix protocol pins warmstart/model-freeze/TRAIN-audit hashes before fitting. Calibration cache identities pin all ordered original image/mask hashes/native geometries/classes; all300/600/900 reports are mandatory before a new global freeze. The same final normal-FP≤1/70 and pixel-precision≥.65 guard applies, with unchanged weakest-positive rank and threshold grids. Pipelined calibration cannot feed back into the fixed fit. The original80 remain an **exposed development comparison**; the strategy follows prior comparison failures but original pixels/IDs never enter fitting, mining, threshold selection or inference rules. Accuracy-first frozen original audit is allowed for the unchanged graph; final project-quiescent paired capacity and sustained≥15FPS remain mandatory before promotion. The normal-feature density alternative is deferred, with no extra model/bank fitted.

E training is allowed to overlap the agreed construction CPU3 fit. Walltimes include that workload and ordinary host activity and are not capacity claims; the estimate14–20 minutes is a scheduling estimate only. A PowerShell quoting error while preparing the new freeze-script copy exited before writing a script or launching NN; copying the existing script and applying an explicit patch resolved it. All old model/protocol/report bytes remain unchanged.

The protected E protocol SHA-256 is **`648e1252c2cba0f3617e67f9561e2524e34ca1f7c34862383859af1cef766ba6`** and its pinned TRAIN geometry audit SHA is`0d9eaf54f9712789dcfbaba9297b9d52b4782ebea9767880c72ecee1efe9d69a`. Before fit, syntax checks passed and `check_factory_native_width_sampling.py` passed40 hardware-free coordinate/GT-pairing batches, including widened hard/unbiased crops retaining annotations away from the mined location and actual-normal rehearsal staying zero. The fixture's first assertion incorrectly assumed left/right widths could not swap under horizontal flipping; the assertion was corrected to compare unordered side counts before the passing run. This was a test-check correction, not a data/model correction. Synthetic pairing fixtures earn no defect-accuracy credit and perform no model forward. The fixed fit then launched with OMP/MKL/Torch1 and all raw progress redirected to `spatial_round3g_native_width-training-console.log`.

Independent read-only review in `docs/factory-attempt5-native-width-independent-review.md` verifies protocol/source/split/FIT hashes, paired label retention, native geometry, protected checkpoint defaults and all-three cache/freeze identity. It finds no material active inconsistency. The first25/50 optimization steps take26.0/52.2 seconds with finite losses; these contended fit observations do not qualify inference speed. As before, every25th printed step falls on SDD2 rehearsal; E alternates **positive at25/75/… and normal at50/100/…**, so printed losses are not comparable batch-type averages or validation metrics.

E300 exported at308.61 optimization seconds. It waited for construction's reserved final122-image calibration before taking its accuracy-only GPU window. Its fixed-rule selected policy isseed.9/area4/growth.5:10/10 meaningful views,0/70normalFP,pixelrecall85.03%/precision69.67%/microIoU.62063,worst-positiveIoU.17585/minimumGTcoverage18.10%,9/10halfcoverage. This worsens worst-view localization compared with D600 despite stronger aggregate/false-alarm values. The contended77.46-second calibration is not capacity evidence. Its ordered source identity remains`7b09d8cbf39129dad144fcad6ecbad0ade0bb19e39de6e8fbdfc634eb6ed6d07`. The600/900 checkpoints remain mandatory; no fitting schedule changes or comparison scoring follow this first result.

### All E checkpoints evaluated, with native-width E900 frozen

The fixed fit completes900 at706.179 seconds and total707.178 seconds including exports, with every optimization loss finite. Construction CPU fitting finishes during E, so contention varies across the raw fit walltime; it is not a controlled training-capacity comparison. No checkpoint feedback changes the900-step schedule. E600 exports at519.705 seconds and loses one meaningful calibration view despite still touching all ten defects. All three results remain in the report:

| E checkpoint | Meaningful IoU≥.1 / any-pixel | Normal FP | Worst positive IoU | Worst GT coverage | ≥50% coverage | Pixel recall | Pixel precision | Micro IoU | Seed / area / growth | Cal child seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 300 | 10/10 /10/10 | 0/70 | .17585 | 18.10% | 9/10 | 85.03% | 69.67% | .62063 | .9 /4 /.5 | 77.46 |
| 600 | 9/10 /10/10 | 1/70 | .00150 | .15% | 9/10 | 83.42% | 67.06% | .59172 | .5 /1 /none | 63.51 |
| **900, frozen** | **10/10 /10/10** | **1/70** | **.31275** | **38.04%** | **9/10** | **86.97%** | **65.59%** | **.59720** | **.995 /32 /.8** | **55.52** |

All meet the final FP/precision guard. E900 wins the declared meaningful-recall/worst-IoU rank; it has worse normal false alarms, precision and microIoU than E300. Thus the predeclared ranking chooses stronger weakest-view localization under the allowed false-alarm budget rather than optimizing every metric. Neither losing checkpoint is evaluated on original80. The exact same80-source identity is checked across all three before freeze.

E300/600 calibration overlaps the fixed E CPU1 fit. E900 has no concurrent fits or other NN, but **project dataset downloads/hash/contact-sheet QA and ordinary host services are still active**. None of these calibration timings earns complete-profile/runtime capacity credit. The inherited generic calibration-plan sentence saying the complete-profile gate comes before comparison is superseded for execution order by the explicit protected E training-protocol `capacity_gate`: all calibration/freeze first, then accuracy-first original audit, then mandatory new-weight/policy project-quiescent speed before promotion. Historical plan bytes remain preserved; no rewritten plan is presented as having been known before scoring.

Before original scoring, the coordinator clarified that observed workload must be additive to historical plans. An initial descriptive correction had edited E900's nested scope description and created an initial freeze, with original metadata/report copies preserved. This was corrected by restoring the exact historical plan objects and adding separate `observed_execution_scope` / `observed_calibration_execution_scope` fields. **The initial freeze remains byte-for-byte preserved and is superseded by v2**, solely for descriptive provenance. `attempt5-additive-workload-provenance-correction.json` verifies unchanged ONNX, score-cache SHA, selected point, raw timings and protected pre-fit/pre-scoring protocol file hashes; no NN, policy search, checkpoint selection or original scoring occurred between the two freezes. Its preserved score-cache SHA is`e22febdbfda5629a1137829d43a62b597efb2a301b135e50af7e8033f96052db`.

The **current authoritative pre-comparison freeze** is `winner-attempt5-native-width-freeze-v2-before-comparison.json`, SHA **`f8ffd7ec9774ad143ec414afa6f18fe927a17385f271c0a05463dfcbb427e4d7`**. It pins:

- Model `models/factory/spatial_round3g_native_width-step900.onnx`, SHA **`a49106fce0573554b1a0794f541d14501f3db60b0fdb4147deccf011a20c8c01`**.
- Final sidecar SHA **`6e6334c835b684ccb16a30ed9e2cb679c268dfd9d7c82c40475ca9edcd1b5299`**.
- Policyseed≥.995, area32 native256×640 cells, connected growth≥.8; raw scores unchanged.
- All planned checkpoint/report/cache identities, prior attempted freezes and the superseded initial E freeze SHA`20302d1d84c798de8bcef2832b52a4f006283ab098dcd613fd1a8256ee489fb6`.

All E fit/calibration processes have exited0 and CPU/GPU are released to the coordinator for independent original mask/source-box audits and final paired capacity/runtime. No officialSDD2 test has opened. This calibration result does **not** establish the requested original recall or speed conjunction, and E is not promoted pending those checks.

### E original accuracy and final D/E runtime: no promotion

The coordinator's independent E900 original80 audit still finds **10/11 meaningful views**, with2/69 normal false alarms,9/11half-GT coverage,pixelrecall76.7427%/precision57.1509%/microIoU.487137. Mask-regionIoU.5 recall is3/11 and actual source-boxIoU.5 recall9/11; no abstentions. This again fails to exceed90.91% view recall. The box result cannot substitute for the weaker mask-region precision/localization. Losing E300/600 are not scored on original80, and officialSDD2 test stays sealed.

After project fitting, competing inference, data-copy/QA and tests quieted, the paired calibration capacity gate measured E21.069 complete-profile calls/s against baseline27.445. Final controlled16FPS/60-second public-still replay then reports:

| Model/run | Captured / processed | Dropped | Late>100ms | Median host ms | P95 ms | P99 ms | Maximum ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| v10h paired baseline | 960/960 | 0 | 0 | 30.9 | 38.6 | 59.8 | 93.7 |
| D600 final record | 960/958 | 2 | 25 | 46.0 | 63.6 | 132.5 | 168.5 |
| E900 final record | 960/960 | 0 | 14 | 41.7 | 51.1 | 114.6 | 151.5 |

All have0 source skips, inspection errors, missed-unit and coverage-gap counters, and60.2-second uptime. E's960/60.2≈15.95 observations/s satisfies throughput in this replay but retains14 configured-deadline misses; D loses two observations and has25 official late outcomes. These software counters do not prove physical product coverage or accuracy on moving factory video. All raw JSONs/traces remain in `factory-round3{f,g}`.

The read-only `factory-round3g/runtime-trace-analysis-v1.json` finds that E's14 late end-to-end observations each had an individual profile call≤100ms; maximum queued time reaches93.39ms and12/960 profile calls exceed the62.5ms input period. D's23 traced processed-late observations plus2 stale drops explain25 official late outcomes. This is observed queue accumulation; it does not establish that any named host process caused it. E/D remain rejected for recall and are not promoted.

The next materially different normal-feature density attempt, its failed precision guard and the separately protected F2 numerical correction are documented in [factory-normal-density-round3.md](factory-normal-density-round3.md). No more supervised augmentation fits or comparison-specific filename/ROI exceptions are introduced.


F2 normal-density fusion completed all nine actual GPU calibrations and was rejected by its unchanged strict control-ranking gate before original scoring. Control7/10 meaningful views with7/10 half-coverage outranked every eligible fusion; no original test, promotion or fitted performance qualification followed. Numeric correction passed, but accuracy did not improve. See [the complete F/F2 study](factory-normal-density-round3.md). Executed source/evidence was archived before later shared-source changes.
