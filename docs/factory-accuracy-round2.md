# Factory defect accuracy: spatial segmentation experiment

The previous factory comparison found real defect locations in only 8 of 11 defective views (72.7%). A box elsewhere in a defective image is not a recovered defect. This round changes the learned representation and spatial model, while preserving a 15 processed observations/s laptop target. Results below must distinguish model-only capacity, complete-profile latency and sustained runtime throughput.

**Measured outcome:** the frozen spatial model localizes **10/11 defective views (90.91%)**, with per-view pixel IoU at least 0.1 in all ten, versus 8/11 any-pixel hits and only 3/11 IoU-qualified views before. Micro pixel IoU improves **0.1133 → 0.4363**. The final complete-runtime trial processes all **960/960 observations at approximately 15.95/s**, with zero drops but **three decisions later than 100 ms**. Normal false alarms increase **1/69 → 5/69**. This is a higher-recall human-review prototype, not a production-qualified or hard-real-time system.

## Research and candidate choice

The authors' [official SegDecNet implementation](https://github.com/vicoslab/mixed-segdec-net-comind2021) trains spatial segmentation jointly with classification. Its published settings include annotation dilation, balanced losses and frequent positive sampling. Its pretrained experimental splits would not match this project's 30/10/10 physical-item partition. No author result is substituted for a measurement on this laptop, and no downloaded experimental checkpoint is treated as independent of the exposed comparison items.

The [official KolektorSDD2 dataset](https://www.vicos.si/resources/kolektorsdd2/) adds a different real industrial surface domain: 246 defective and 2,085 normal training images, plus 110 defective and 894 normal official test images, approximately 230×630 pixels. Its CC BY-NC-SA 4.0 license requires a separate commercial agreement for commercial use. Its domain and test partition must remain separate from the original KolektorSDD headline; adding images does not itself establish transfer to the original commutator defects.

The implemented candidate is an end-to-end spatial MobileNet U-Net. It initializes the first seven MobileNetV3 Small blocks from pinned official torchvision ImageNet weights, then trains the encoder and a new spatial decoder. Stride-two, stride-eight and stride-sixteen skip features preserve detail and provide context; depthwise 3×3 residual layers learn spatial neighborhoods. This materially differs from the previous frozen encoder with a pointwise head, which could not learn new spatial filters or adapt the encoder to the defect domain.

## Frozen training protocol

Training opens only the original 30 training items. The ten calibration items select checkpoints and operating points; the ten comparison items remain untouched by the trainer and calibrator. The comparison has been exposed in previous rounds and is a reused development comparison, not a blind test.

The native-detail training patches are 320 pixels tall and 128 wide, batch size eight. Half are positive-centered with jitter; one quarter come from normal views and one quarter are unbiased training patches. Matching image/mask flips, contrast and brightness changes apply only in training. The loss combines positive-weighted binary cross entropy (weight ten) and soft Dice on positive patches. All convolution and affine normalization parameters train; pretrained batch-normalization population statistics remain fixed to avoid corrupting them with biased patches. AdamW uses encoder learning rate 0.0002, decoder 0.001 and weight decay 0.0001. The seed is 20261025; checkpoints are fixed at 300 and 600 steps.

Training masks are nearest-resized to the input field and max-pooled over 2×2 cells for the stride-two output, preserving thin annotated defects. This is target projection for training; all reported localization uses untouched original-resolution annotations.

Full inference input is 512×1280 BGR; output logits are 256×640. The actual graph contains BGR-to-RGB conversion and ImageNet normalization. ONNX files and metadata use explicit SHA-256 hashes. Training uses the isolated Torch export environment; application inference uses local OpenVINO on Intel GPU, without Torch or network inference.

Calibration chooses from fixed probability thresholds 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995 and 0.999, and minimum connected-component areas 1, 2, 4, 8, 16 and 32. It permits at most three false alarm views among 70 normal calibration views. Selection first maximizes defective views whose predicted foreground has pixel IoU at least 0.1 with the true mask; then any-overlap localized recall, micro pixel IoU, foreground recall and fewer false alarms. The calibration protocol is written before scoring. Predictions use the runtime's largest-20 component cap and nearest projection to original annotation resolution; validation annotations are never dilated.

## Capacity preflight, before training

Hardware is the same Windows laptop used for the earlier study: Intel Core i7-8550U (four cores/eight threads), 24 GB RAM and Intel UHD 620. Training/export uses the isolated CPU Torch 2.14.1 environment with four training threads. Application inference uses OpenVINO 2026.1.0 on GPU.0, one stream, latency mode and an explicit FP16 precision hint. The paired profile scripts use two OpenCV threads. Historical hardware/runtime details remain in [the performance study](performance-study.md).

The model has 140,913 parameters and exports to a 579,562-byte ONNX file. An untrained capacity probe on the actual 512×1280 laptop input measured preprocessing, GPU inference, float32 sigmoid and connected components at **29.95 serial observations/s**, warm median **32.72 ms**, P95 **37.71 ms**. Twenty-one warm observations followed three startup observations. GPU initialization took 7.95 seconds; execution device was GPU.0. These are capacity measurements, not accuracy or complete-runtime acceptance.

CPU training preflight measured batch-eight 320×128 forward/backward steps at a warm median 0.271 seconds. The raw reports are `output/optimization/factory-round2/spatial_unet_v9-training-capacity.json` and `spatial_unet_v9-gpu-capacity.json`. The trainer writes its protocol before fitting, progress every 25 steps, loss history, checkpoint artifacts and elapsed time.

## Localization and deployment acceptance

The independent `scripts/factory_localization_audit.py` reports any-pixel overlap only for comparability. It also reports views covering at least 10% and 50% of actual defect pixels, foreground pixel recall and precision, micro IoU, macro defective-view IoU and one-to-one component matches at IoU 0.1, 0.25 and 0.5. False normal alarms, false locations in defective views, abstentions and every missed item remain visible.

No model becomes the factory default merely because calibration improves. Frozen comparison, meaningful localization, full-profile latency and a sustained at-least-15 FPS runtime benchmark are separate gates. Real public-image replay measures compute capacity; it does not validate moving production-line video, exposure, object triggers or customer material coverage.

## Reproduction

```powershell
.venv-roads-export/Scripts/python.exe -X utf8 scripts/train_factory_spatial_unet.py --probe
.venv-roads-export/Scripts/python.exe -X utf8 scripts/train_factory_spatial_unet.py --steps 600 --save-at 300 600
.venv/Scripts/python.exe -X utf8 scripts/calibrate_factory_spatial.py
```

Final calibration, comparison and complete-runtime results are appended after the artifacts are frozen. Failure is retained explicitly rather than replaced with image-level flags or boxes away from annotations.

## First spatial experiment: useful segmentation, rejected false alarms

The 600-step SDD1-only fit completed in 234.92 seconds using 239 real training views, including 31 defective views. It opened zero calibration or comparison images while fitting. Both fixed checkpoints found all ten actual calibration defect masks with per-view pixel IoU at least 0.1 at their strictest grid point (probability 0.999, minimum 32 grid pixels). Neither met the predeclared normal false-alarm budget.

| Calibration-only checkpoint | Meaningful localized defective views | Normal false alarms | Foreground recall | Foreground precision | Micro pixel IoU |
|---|---:|---:|---:|---:|---:|
| Step 300 | 10/10 | 8/70 | 46.31% | 84.37% | 0.4265 |
| Step 600 | 10/10 | 9/70 | 64.24% | 80.59% | 0.5563 |

Step 600 covered at least half the true annotated pixels in eight of the ten calibration defects. Its preprocessing/NN/sigmoid/components path measured mean 29.88 ms and P95 36.55 ms on these calibration views. These are not comparison results, not a complete-runtime benchmark and not a reason to deploy a model with excessive false alarms. Both checkpoints are explicitly rejected under the original calibration grid, and the original 80 comparison images were not opened.

The raw training and calibration reports are `spatial_unet_v9-training.json` and `spatial_unet_v9-calibration.json` in `output/optimization/factory-round2/`. No calibration failure is erased by relabeling a wrong location as a true detection.

## Predeclared second experiment: more real data and hard normal patches

Before any comparison scoring, the next bounded experiment uses fresh official ImageNet initialization, 300 supervised steps on the reserved-fit portion of official SDD2 training data, followed by 300/600-step checkpoints on the original SDD1 training items. The architecture remains identical, so its capacity probe still describes the graph shape; actual final latency is measured again.

The SDD2 inner split was written before training: 1,865 fit views (197 defective, 1,668 normal), and 466 reserved calibration views (49 defective, 417 normal). Canonical numeric filenames exclude byte-identical author-archive copy files. Identical source-image hashes cannot cross the split. The author release supplies no verified physical-item/batch identifiers, so source-image separation is not claimed to be physical-item independence. The 1,004-image official test remains sealed during training and model selection. The split manifest is `datasets/factory/KolektorSDD2/train-inner-split-before-fit.json`, SHA-256 `ecee83db4becdac3133a2d4c0950ef383c511d143b5509aa8c1fdf61d445a47c`.

SDD2's roughly 230×630 images are resized to the same 512×1280 training field before cropping, introducing a small aspect change as well as approximately twofold upscaling. This is a transfer-learning experiment across different industrial surfaces, not extra SDD1 images and not evidence of universal factory coverage.

The second fit also mines high-score false locations from original SDD1 **training** views using the rejected step-600 model. Real annotated defect pixels and a surrounding margin are excluded from mining. At most sixteen spaced hard locations are retained per view. One quarter of SDD1 fine-tuning patches are centered on these locations with jitter, while the positive half and unbiased quarter remain; matching true masks are still used for every patch. Neither calibration nor comparison pixels enter hard-negative mining.

The second candidate's threshold grid additionally includes 0.9999, 0.99999 and 0.999999 to handle the first model's overconfident logits. This expansion is declared before the second candidate is fitted or calibrated; it is a documented response to a calibration failure, not a comparison-set adjustment. The original grid's failed result remains above. Improvements, if any, arise from this combined experiment; they cannot be attributed solely to the extra dataset without an ablation study.

Training-only mining completed in **25.02 seconds**, yielding **3,823** hard locations across 239 original training views. Its report is `output/optimization/factory-round2/spatial-train-hardneg.json`. The mining model was the rejected v9 step-600 ONNX, SHA-256 `5cda3fc0bcf8f8f0acb269e7166564d05cd0f4bfd14cf8f459a08bd5bc4fcf9d`.

The SDD2 pretraining stage completed 300 steps in **125.89 seconds of optimization/export**, after loading native images. It used 1,865 real fit views, including 197 defective ones. Its trainer opened zero inner-calibration and official-test images. Raw history and the complete protocol are `spatial_unet_sdd2_pretrain_v10-training.json` in the round-two output directory.

The subsequent 600-step SDD1 fine-tune completed in **192.17 seconds**, using the original 239 training views (31 defective) with the hard-negative patch policy. The trainer opened no SDD1 calibration/comparison images. Both fixed checkpoints were exported for calibration; optimization loss is not treated as accuracy.

```powershell
.venv/Scripts/python.exe -X utf8 scripts/mine_factory_spatial_hardneg.py
.venv-roads-export/Scripts/python.exe -X utf8 scripts/train_factory_spatial_unet.py --dataset-family sdd2 --split-json datasets/factory/KolektorSDD2/train-inner-split-before-fit.json --name spatial_unet_sdd2_pretrain_v10 --steps 300 --save-at 300 --extended-score-grid
.venv-roads-export/Scripts/python.exe -X utf8 scripts/train_factory_spatial_unet.py --name spatial_unet_v10_sdd2 --resume models/factory/spatial_unet_sdd2_pretrain_v10-step300.pt --hardneg-json output/optimization/factory-round2/spatial-train-hardneg.json --steps 600 --save-at 300 600 --extended-score-grid
.venv/Scripts/python.exe -X utf8 scripts/calibrate_factory_spatial.py --name spatial_unet_v10_sdd2
```

## Frozen calibration winner and full-profile capacity

The declared IoU-first rule selected **v10 SDD2-transfer step 300**, probability threshold **0.999**, minimum component area **16 grid pixels**. Its calibration results were 10/10 true-mask touches, **9/10 views with pixel IoU at least 0.1**, and 3/70 normal false alarms. Foreground precision was **94.05%**, foreground recall **34.63%**, micro IoU **0.3388**. These are calibration results on only ten defective views; they are not independent accuracy estimates. The remaining calibration defect had only weak foreground coverage and does not count toward the IoU-based recall.

Step 600 was not selected. Although it produced fewer normal false alarms (1/70 at threshold 0.9999 / area 4), only 7/10 calibration views met pixel IoU 0.1, and micro IoU fell to 0.2410. More training did not automatically mean better localized predictions.

| Complete-profile calibration capacity, 80 views | Baseline | Frozen spatial model |
|---|---:|---:|
| Mean process latency | 35.68 ms | 36.58 ms |
| P95 process latency | 54.73 ms | 44.44 ms |
| Maximum process latency | 67.91 ms | 49.37 ms |
| Serial profile capacity | 28.03/s | **27.33/s** |
| Quality abstentions | 0 | 0 |

This paired, warmed pass includes actual quality checks, neural preprocessing/inference, components, findings and profile annotation. It excludes capture, queueing, JPEG/evidence work and the browser, so it is a capacity screen, not the final sustained-runtime FPS result. The raw report is `output/optimization/factory-round2/spatial-winner-calibration-capacity.json`.

The frozen model is `models/factory/spatial_unet_v10_sdd2-step300.onnx` (579,607 bytes), SHA-256 `dd5019cbcbc859c4eeb994a7639fb0c520ee3e3175e47a3f1544ec4c5dd60753`. Its `.onnx.json` metadata SHA-256 is `8d79ccc907ebcd6e754f57db0419cd7d1a4de49a34afb261dfaad6e44d585625`. Metadata records the architecture, original item split, complete pretraining lineage, training-only hard-negative source and frozen operating point. It explicitly prohibits automatic unit acceptance/rejection. The application verifies the ONNX hash and explicitly requires the installed Intel GPU.

`round2-calibration-winner-before-comparison.json` records this winner and capacity gate before original-comparison scoring. The original baseline remains preserved. The next gates are the independent original-mask audit, paired comparison latency and sustained runtime measurement at at least 15 processed observations/s; any comparison failures must remain visible.

## Frozen seeded-growth refinement: more actual foreground coverage

Before opening the original comparison, a separate calibration-only experiment retained the frozen step-300 high-score seeds (0.999 threshold, minimum area 16, largest twenty components) and grew only lower-score connected foreground that touched one of those seeds. The graph, raw logits and sigmoid scores remain unchanged. An observation with no retained high-score seed can never become flagged through this policy. Disconnected weak foreground is not retained, and no bounding-box or ground-truth expansion is used.

The growth grid, written before scoring, was no growth / 0.1 / 0.25 / 0.5 / 0.75 / 0.9 / 0.95 / 0.98 / 0.99 / 0.995. Selection maximized per-view pixel IoU-at-least-0.1 recall, then micro pixel IoU and foreground recall, while asserting identical flagged-view decisions to no growth for every calibration image. No comparison pixels or masks were opened.

Calibration selected growth threshold **0.95** in **29.59 seconds**:

| Calibration original-mask metric | Frozen no growth | Seeded growth |
|---|---:|---:|
| Defective views with any actual overlap | 10/10 | 10/10 |
| Defective views with pixel IoU ≥0.1 | 9/10 | **10/10** |
| Defective views covering ≥50% true pixels | 1/10 | **9/10** |
| Foreground pixel recall | 34.63% | **74.00%** |
| Foreground pixel precision | 94.05% | 75.28% |
| Micro pixel IoU | 0.3388 | **0.5954** |
| Normal false alarm views | 3/70 | 3/70 |

The precision reduction is explicit: growth includes more wrong foreground pixels as it recovers more true defect pixels. The increased IoU and coverage are measured against untouched publisher annotations, not inferred from the number or size of displayed boxes. Growth cannot increase flagged-image recall by itself; its purpose is meaningful segmentation of already-seeded candidates.

The separate final artifact is **`models/factory/spatial_unet_v10h_sdd2-step300.onnx`**, with unchanged ONNX SHA-256 `dd5019cbcbc859c4eeb994a7639fb0c520ee3e3175e47a3f1544ec4c5dd60753`. Its new policy metadata SHA-256 is **`25d8d250c3abc0deab82994c3cc48f6f5d356058bbd74ab4903b9cd497f1ba78`**. The no-growth ONNX and sidecar remain unchanged. `seeded-growth-protocol-before-scoring.json` and `seeded-growth-calibration.json` preserve the experiment.

The grown model's full-profile calibration capacity remained **26.91 observations/s**: mean **37.17 ms**, P95 **45.15 ms**, maximum **54.60 ms**, with zero quality abstentions. The paired baseline in that pass measured 31.92/s, mean 31.32 ms and P95 46.90 ms. These are warmed profile capacity measurements, not sustained source/runtime throughput. The raw report is `spatial-grown-calibration-capacity.json`.

Five focused contract tests passed: raw-score immutability, no new flags without a high seed, disconnected weak-region exclusion, independent audit geometry agreement and the existing neural runtime contract. Fixtures establish implementation behavior; real annotations establish accuracy. The independent auditor and original-mask comparison script reconstruct the seeded mask from metadata, rather than misreporting a raw single-threshold mask as the runtime prediction.

```powershell
.venv/Scripts/python.exe -X utf8 scripts/calibrate_factory_seeded_growth.py
.venv/Scripts/python.exe -X utf8 scripts/factory_localization_audit.py --model models/factory/spatial_unet_v10h_sdd2-step300.onnx --output output/optimization/factory-round2/spatial-final-localization-audit.json --visuals-dir output/optimization/factory-round2/spatial-final-localization
```

The final policy was frozen and passed the complete-profile calibration screen before comparison scoring. Original-comparison and sustained-runtime results are reported separately below; calibration percentages must not be substituted for them.

## Frozen original-mask comparison: actual localized recall improves

The independent audit evaluated the frozen original baseline and final v10h policy on the same **80 real views from ten held-out physical items**, including **eleven defective and 69 normal views**. These comparison items have been exposed in earlier development; this is a reused development comparison, not a new blind test. No training, threshold selection or policy adjustment used these results.

| Original-resolution comparison metric | Original baseline | Frozen spatial model |
|---|---:|---:|
| Defective views with any actual predicted-pixel overlap | 8/11 (**72.73%**) | 10/11 (**90.91%**) |
| Defective views with pixel IoU ≥0.1 | 3/11 (27.27%) | **10/11 (90.91%)** |
| Defective views covering ≥10% true defect pixels | 3/11 | **10/11** |
| Defective views covering ≥50% true defect pixels | 0/11 | **7/11** |
| Foreground pixel recall | 11.37% | **54.44%** |
| Foreground pixel precision | **96.88%** | 68.73% |
| Micro pixel IoU | 0.1133 | **0.4363** |
| Macro defective-view IoU | 0.0886 | **0.4388** |
| One-to-one defect-region recall at IoU 0.1 | 2/11 | **10/11** |
| One-to-one defect-region recall at IoU 0.25 | 0/11 | **9/11** |
| One-to-one defect-region recall at IoU 0.5 | 0/11 | **5/11** |
| Normal false alarm views | **1/69 (1.45%)** | 5/69 (7.25%) |
| Flagged defective views with zero actual overlap | 0 | 0 |
| Quality abstentions | 0 | 0 |

This is a real localization improvement, not an extra flag or a box elsewhere in a defective image. The new model recovers the annotated defects in `kos02/Part6.jpg` and `kos38/Part1.jpg`. **`kos12/Part5.jpg` remains missed.** The five new-model normal alarms are `kos07/Part2.jpg`, `kos26/Part1.jpg`, `kos32/Part6.jpg`, `kos45/Part0.jpg` and `kos45/Part6.jpg`. The original baseline flagged only the first of those normal views.

The tradeoff matters: flagged-view precision on this particular public comparison falls from 8/9 (88.89%) to 10/15 (**66.67%**) as four additional normal views enter review. Foreground precision also falls while recall and IoU improve. These are not across-the-board accuracy gains and do not justify automatic line rejection. Review workload depends on the real line's defect prevalence and appearance; the prototype remains manual review only.

Raw audits are `output/optimization/factory-round2/baseline-localization-audit.json` and `spatial-localization-audit.json`. The audit saves **every defective view and every normal false alarm**, including the remaining miss, rather than choosing favorable examples. Each evidence panel shows raw input, untouched publisher ground truth, independently reconstructed model foreground and true-positive/false-negative/false-positive pixels. Separate original-resolution ground-truth and predicted-mask PNGs support inspection without display scaling.

The eleven defective views are too few, and the comparison too exposed, to establish production reliability. This score is restricted to this surface family. SDD2 served as separately documented training transfer; its official test was not used to select the SDD1 winner and is not substituted for this headline.

## Paired frozen-comparison profile timing

The same original 80 comparison images were predecoded, warmed and processed twice by each full profile, with the before/after execution order alternating. Repetition adds latency observations, not independent accuracy examples. Both frozen policies were unchanged after the audit.

| Full-profile processing, 160 timing observations per model | Baseline | Spatial v10h |
|---|---:|---:|
| Mean | 30.86 ms | 38.58 ms |
| P95 | 45.95 ms | 48.69 ms |
| Maximum | 55.76 ms | 59.71 ms |
| Serial profile capacity | 32.40/s | **25.92/s** |

The richer segmentation costs approximately 7.72 ms mean profile time while retaining capacity above 15/s. This includes actual quality checks, inference, mask policy, findings and annotation; capture, queueing, preview/evidence work and browser interaction are excluded. Sustained complete-runtime results remain a separate gate. The raw paired report is `output/optimization/factory-round2/paired-profile-comparison.json`.

The report's image F1 decreases from **0.8000 to 0.7692**, and normal-dominated flagged-image accuracy from **95.0% to 92.5%**, because the four extra normal alarms outweigh the two recovered defective views under those metrics. This round achieves higher genuine defect recall and substantially better segmentation IoU; it does **not** establish higher overall classifier accuracy or lower review workload.

## Sustained complete-runtime benchmark

The before/after protocol configures 16 input observations/s for 60 seconds, preview compression at 6 FPS and a 100 ms capture-to-decision deadline. It cycles the same eighty **real, predecoded public images** at a software clock. This exercises capture scheduling, bounded queues, profile processing, preview/evidence work and event persistence; it is not real moving-factory-video accuracy or physical line-coverage validation. Repetition creates timing load, not new independent examples.

The preserved baseline completed **960 captured / 960 processed observations** in 60.2 seconds of runtime uptime, with zero drops, late decisions, source skips, inspection errors, missed units or reported coverage gaps. The snapshot reports 16.0 captured/processed FPS; dividing the count by full uptime gives approximately 15.95 observations/s including drain. Host capture-to-decision latency was P95 **47.8 ms**, P99 **52.8 ms**, maximum **58.5 ms**. The raw report is `output/optimization/factory-round2/baseline-runtime-16fps-60s.json`.

The first cold-start spatial runtime run processed **956 of 960 captured observations**, reporting **15.9 FPS**, **four drops** and **fourteen decisions beyond the 100 ms deadline**. Host P95 was only **45.9 ms**, but P99 reached **100.5 ms** and the maximum **208.3 ms**. Model initialization took **8.59 seconds**. There were zero inference errors. The failed run remains `spatial-runtime-16fps-60s.json`.

That average FPS meets the throughput target, but the drops and latency tail prevent treating this first run as reliable timely coverage. Paired profile tests warmed their model, whereas the initial runtime compiled a request and immediately enabled capture. Startup/lazy-execution behavior is being traced explicitly; any warmup fix must report its initialization cost and cannot remove, relabel or omit the captured slow observations from this failed report.

### Trace and explicit model initialization

A separate cold 15 FPS / 30-second trace used the stricter **66.667 ms** frame-period deadline. It processed all **450/450 observations**, with zero drops but **two late decisions**; P95 was **54.7 ms**, maximum **71.1 ms**. The first profile call was 58.46 ms (58.79 ms host latency), followed by calls near 33 ms. The slowest observation was sequence 43 at 2.808 seconds (70.75 ms profile time); another tail was sequence 88 at 5.814 seconds (63.52 ms profile time plus 3.17 ms queue age). The report is `spatial-cold-diagnostic-15fps-30s.json` and includes the per-observation trace.

This trace **does not support attributing all stalls to startup**. Midrun scheduling/processing tails remain possible, and the earlier 208 ms spike cannot be localized retrospectively because that first report did not record a trace.

The implemented initialization improvement performs **three full neural prediction-path warmups before capture starts**, including input preparation, inference, output readback, sigmoid and the frozen seeded-component policy. Fixed-shape zero tensors are initialization inputs only; they are never inspected observations, validation examples, events, triggers or accepted/rejected units. The real inspection quality gates and all captured observations are unchanged.

`description.model_initialization` reports graph/request initialization time, each warmup duration, total warmup cost, execution thread and scope. Those costs remain visible in station startup. A sixth focused contract test verifies the three calls and telemetry without creating inspection state. Warmup leaves all weights, thresholds, raw-score semantics and output masks unchanged, and explicitly does not promise to remove later GPU/OS scheduling delays. A new traced sustained run is required before judging its effect.

The constructor-warmed retest processed **960/960 observations** at reported **15.9 FPS**, with zero drops but **two decisions beyond 100 ms**. P95 was **47.0 ms**, P99 **53.0 ms**, maximum **106.3 ms**. Main-thread warmups took 45.7 / 39.2 / 36.7 ms, but the first real processing-thread profile call still took 102.4 ms. A separate midrun stall occurred at sequence 538 (33.564 seconds, profile time 106.1 ms). Total station startup was 9.687 seconds, including approximately 9.531 seconds graph/request setup and 122 ms warmup. These observations remain in `spatial-warmed-runtime-16fps-60s.json`; constructor warmup did not eliminate all tail latency.

The final startup revision moves preparation to the **actual `edge-inference` worker** and gates source creation/capture on successful completion. It directly initializes the native OpenCV fast-quality routines without calling the inspection method, then runs the three neural-path warmups using zero images 500 pixels wide and heights 1280 / 1281 / 1267. These representative source shapes exercise resizing instead of bypassing it with an already-512×1280 tensor. Quality and neural warmup timings, shapes and worker names remain explicit.

The worker-ready barrier times out after at most fifteen seconds of preparation waiting. Failure/timeout prevents source creation, closes the pending buffer/media worker, records `startup_error` and retains ownership of any still-exiting native worker until shutdown. Acquisition uptime starts only after successful preparation; external station-start time includes all initialization cost. No captured observation is removed or reclassified to improve reported percentiles.

Per-prediction `factory_neural_stage_ms` telemetry separates input resize/layout, inference plus output readback, and sigmoid/components/postprocessing, so remaining stalls can be investigated rather than assumed to be startup effects. Model initialization and the growth threshold are also retained in event provenance. Twenty-one focused factory/runtime tests passed, including worker-before-capture ordering, immediate failure cleanup and timeout cleanup. The remaining acceptance step is a final traced baseline/candidate runtime comparison on this common startup code.

The worker-warmed 16 FPS / 60-second trial processed **960/960 observations**, with zero drops and reported **15.9 FPS**, but still had **four decisions beyond 100 ms**. Host P95 was **51.4 ms**, P99 **85.6 ms**, maximum **127.3 ms**. Worker neural warmup took 97.8 ms and quality warmup 3.3 ms; graph/request setup took 8.371 seconds, total station start 8.484 seconds. The first real call was no longer the worst. Remaining queue/timing tails appeared around 32.4 and 56.7 seconds.

The stage trace showed relatively stable GPU inference/readback at approximately 21–25 ms, with slow-window CPU preprocessing at 8–20 ms and postprocessing at 11–17 ms. The 127 ms host tail included roughly 60 ms of queue age plus a 66 ms profile call. Worker preparation improved initialization ownership but did not establish a hard bound on later CPU scheduling or queue latency; this trial's misses remain in the report.

One final exact optimization removes provably unnecessary postprocessing. A frame without any retained high-score seed cannot produce a grown mask: after validating the growth threshold, it now returns an empty mask immediately. The low-score labeling pass no longer calculates unused statistics. Empty predictions skip final component statistics and score accumulation. **Every inspected frame still executes the full neural network forward pass and high-score seed decision**; this is neither frame skipping nor a cached/precomputed prediction path. No pixel scores, seed thresholds, component selection, inspected frames or nonempty-mask results change. Ten focused factory/worker tests passed, including a mocked empty observation that forbids the skipped work and still requires the high-seed decision pass. A final sustained retest and independent-mask identity check are required after this optimization.

On the common worker-barrier code, the preserved baseline completed **960/960 observations**, zero drops/late decisions, reported **16.0 FPS**, P95 **48.5 ms**, P99 **54.3 ms**, maximum **94.4 ms**. Its worker quality warmup took 4.967 ms and station start 2.601 seconds. That report is `baseline-final-runtime-16fps-60s.json`; the pre-fast-path spatial worker report is `spatial-final-runtime-16fps-60s.json`. The final exact-fast-path spatial trial is recorded separately, not substituted for the earlier failing trials.

## Final complete-runtime result and all retained variants

The final exact-fast-path model processed **960/960 captured observations**, zero drops, source skips, inspection errors or reported missed units, at reported **16.0 FPS** over 60.2 seconds uptime (approximately **15.95 processed observations/s including drain**). Every one of the 960 trace entries includes a real neural forward pass. Median host latency was **33.0 ms**, P95 **47.3 ms**, P99 **70.4 ms**, maximum **130.4 ms**. **Three of 960 decisions exceeded the 100 ms deadline.** The throughput target is met; reliable hard-deadline completion is not established.

| Sustained 16 FPS input, 60 seconds | Processed / captured | Dropped | >100 ms decisions | Host P95 | Host P99 | Host maximum |
|---|---:|---:|---:|---:|---:|---:|
| Initial baseline | 960/960 | 0 | 0 | 47.8 ms | 52.8 ms | 58.5 ms |
| Initial cold spatial model | 956/960 | 4 | 14 | 45.9 ms | 100.5 ms | 208.3 ms |
| Constructor-warmed spatial model | 960/960 | 0 | 2 | 47.0 ms | 53.0 ms | 106.3 ms |
| Baseline on final worker-barrier code | 960/960 | 0 | 0 | 48.5 ms | 54.3 ms | 94.4 ms |
| Worker-warmed spatial model | 960/960 | 0 | 4 | 51.4 ms | 85.6 ms | 127.3 ms |
| **Final exact fast path** | **960/960** | **0** | **3** | **47.3 ms** | **70.4 ms** | **130.4 ms** |

These are sequential measured trials, not guaranteed best/worst-case bounds. Warmup and exact postprocessing optimization improve initialization behavior and typical latency, but the tail results are not uniformly better and do not justify a hard real-time claim. All failed and intermediate reports remain available. The separate 15 FPS / 66.667 ms diagnostic above is not merged into this 100 ms table.

The final result satisfies the requested at-least-15-FPS laptop processing target while preserving the frozen model's improved actual defect localization. It still incurs a higher normal-review rate, one missed defect in eleven reused defective views and occasional late decisions. The device remains a research prototype requiring human review; these measurements do not qualify a physical production line or moving-camera acquisition.

The final traced report is `output/optimization/factory-round2/spatial-fastpath-runtime-16fps-60s.json`. It is kept alongside the baseline, cold, constructor-warmed and worker-warmed reports named above. No unfavorable run is overwritten.

## Final accuracy identity and reproduction

After the startup/barrier and exact empty-mask changes, the independent auditor reran all eighty comparison views. **All eighty original-resolution binary prediction masks were hash-identical**, with **zero changed views**, and all aggregate localization metrics remained identical. Source-image, publisher-mask, ONNX and frozen policy hashes also matched. The comparison used the previously saved exact PNG masks for every defective view and all five normal false alarms; the other 64 reference masks were known-empty predictions. No inference was skipped in the new run.

The evidence is `output/optimization/factory-round2/fastpath-accuracy-identity.json` and `spatial-final-localization-audit.json`. The final audit includes per-view prediction-mask SHA-256 and shape. Thus the speed change does not purchase throughput by removing defects, changing operating points or suppressing normal alarms.

Timing jobs ran sequentially on the shared laptop; dataset/source research, documentation and UI work ran in parallel through subagents. Reproduction commands below use separate filenames to preserve the measured reports:

```powershell
.venv/Scripts/python.exe -X utf8 scripts/benchmark_runtime.py --varied --model models/factory/spatial_unet_v10h_sdd2-step300.onnx --fps 16 --seconds 60 --deadline 100 --preview-fps 6 --trace --output output/optimization/factory-round2/spatial-runtime-reproduction.json
.venv/Scripts/python.exe -X utf8 scripts/factory_localization_audit.py --model models/factory/spatial_unet_v10h_sdd2-step300.onnx --output output/optimization/factory-round2/spatial-localization-reproduction.json --visuals-dir output/optimization/factory-round2/spatial-localization-reproduction
.venv/Scripts/python.exe -X utf8 -m pytest tests/test_factory_seeded_growth.py tests/test_factory_neural.py tests/test_worker_preparation.py -q
```

Repository-wide verification and browser evidence are recorded by the parent task's final report. This factory study records every experimental result and the measured limits rather than declaring production readiness from a small reused public comparison.
