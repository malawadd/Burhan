# Round5: fixed thin-structure supervision

Status: the fixed fit, all numerical/calibration outcomes and root's original/speed/runtime measurements are complete. The joint target was not achieved and no model was promoted. See [actual measured results](C:/Users/IAM/Documents/hackathon/opencv/project1/docs/factory-round5-measured-results.md) and [all11 defect views plus the normal false alarm](C:/Users/IAM/Documents/hackathon/opencv/project1/docs/factory-round5-localization-evidence.md).

Round4 added real training coverage but its three frozen comparison candidates each still localized10/11 defect views. Their normal-view alarm counts were2/69,2/69 and1/69; none exceeded C200's10/11 with one alarm. More training sources therefore did not establish the requested joint improvement. Round5 tests a different training objective while retaining the same serving graph.

## Hypothesis and primary sources

[Shit et al., CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html) introduce a differentiable loss based on agreement between foreground and skeletons. Their experiments concern structures such as vessels and roads, rather than this industrial defect domain. The [authors' implementation](https://github.com/jocpae/clDice) uses iterative soft erosion/opening. This provides a connectivity-oriented training hypothesis, not evidence that factory recall will improve.

The new auxiliary term uses five fixed skeleton iterations and smoothing1. Skeleton precision penalizes predicted traces outside observed foreground; skeleton sensitivity penalizes missing true traces. It is applied only during training. Inspection receives only the original camera image and executes the ordinary PackedPixelRefiner followed by MaxPool2d(2). No skeleton, annotation, morphology operation or GT-derived feature enters inspection.

## One fixed fit

- Warmstart: the exact latest Round4 fine900 PT, SHA256 `4b39937f9719ba39c2fb256f29c5cec4e45be2474e312c1b577aec6a58a818d8`. It is chosen as the fixed latest checkpoint, rather than because it performed best on the exposed comparison.
- CPU1, batch1, fixed600 updates, with checkpoints200/400/600. All600 finish before any checkpoint numerical or calibration feedback. No early stopping or extension.
- Actual native320×512 full-width patches. Positive sampling cycles all projected SDD1 TRAIN components; each chosen component must fit fully under vertical jitter±64. The continuous bounding-box center uses `y+h/2` to preserve half-open box containment at the maximum declared height.
- Fixed roles:300 SDD1 positive,120 existing TRAIN hard-location,120 SDD1 normal and60 existing real-domain rehearsal patches,30 SDD2 and30 Magnetic Tile. This rehearsal does not repeat Round4's complete-source-coverage claim.
- Fresh AdamW: encoder/context learning rate1e-5, decoder/refiner5e-5, weight decay1e-4, gradient clipping5. Normalization buffers remain fixed, with full-state finite and buffer-equality checks after every update.
- Existing matched flips, contrast, brightness and mild blur remain fixed. There is no new augmentation/configuration grid or original-image-specific rule.

The dense objective is2×positive BCE +3×hard-negative BCE +FN-weighted Tversky +1×soft-clDice. Background supervision remains because topology alone can connect normal texture and increase alarms. All roles retain their complete actual crop masks, including defects away from a sampled point.

Unobserved padding has zero loss validity. Topology additionally excludes an eight-pixel crop/validity boundary: five erosions plus an opening have a maximum seven-pixel dependency radius. If no target skeleton survives in that observed interior, topology contributes exactly zero; ordinary foreground/background supervision still applies to observed boundary pixels. Annotation files and their foreground support are unchanged. Skeletonization is a loss calculation, not new defect data or a modification of evaluation GT.

## Fixed evaluation

Each exported candidate must pass its own three-case Torch versus actual GPU numerical check at maximum score error0.01, finite/bounded outputs and the retained deployment recipe. Failed numerical candidates open no calibration inputs; all three terminal outcomes remain mandatory.

There is one prospective zero-alarm calibration arm. It uses the existing raw thresholds, areas, growth options, seed shortlist and rank. A candidate qualifies only with10/10 meaningful calibration views,0/70 normal-view alarms and pixel precision at least0.65. The original80 calibration identities use an explicit local historical JSON serializer, plus exact ordered paths, image/mask hashes and native-geometry metadata. Their preparation performs no calibration-pixel decode for fitting.

All three actual calibration outcomes freeze before any original comparison. Root must then measure fresh fitted complete-profile capacity of at least15FPS, followed by exposed-comparison localization and the sustained runtime/capture/queue/evidence checks. The source graph's previous speed is not an acceptance result for these weights. No losing checkpoint or threshold can be substituted after original scoring.

The rough4–8-minute CPU fit estimate comes from prior320-patch optimizer observations plus an unmeasured auxiliary-loss cost. It is not a measured speed guarantee. Actual walltime, contention, every numerical failure and all accuracy/runtime outcomes must be retained.

## Actual preparation and launch

All eight tiny TRAIN-only tests passed in3.233 seconds with zero skips. They check whole-component containment at jitter/border limits, the exact historical calibration serializer, missing thin traces, false bridges, empty targets, invalid annotations, padding invariance/zero gradients and observed border supervision. They construct no neural model and load no weights or dataset pixels.

The actual TRAIN preparation found33 projected SDD1 components, all represented and fully contained in the fixed positive ledger. The600-update role counts are exactly300/120/120/60. Independent metadata/source review verified the3444 accepted source IDs, genuine normal-role masks, existing partitions and grids, and protected calibration identity without decoding calibration pixels for fitting.

The exact prepared proposal and recorded protocol SHA256 is `7153f2e58df0b37e6230438faaee23e2eb7257e5b2b8b562e6f5e724e58431b7`. Root approved it, protocol recording passed, and the fixed fit started as session45842 with OMP/MKL/OpenBLAS/Torch threads1. Construction CPUORT1 validation may run concurrently; optimizer walltimes are therefore contended observations and receive no inference-speed credit. No schedule, policy guard or checkpoint was changed after launch.

The first50 optimizer updates completed in41.169 seconds with finite batch loss1.48509; update100 completed at83.740 seconds with loss1.22181. Full-state/gradient finite checks and frozen-buffer equality passed through these observations. These are individual TRAIN batches, not validation accuracy or inference timing. The fixed600-update schedule remains unchanged.

All600 updates and three exports completed in475.484 seconds, approximately7.925 minutes, with no early stopping or extension. Each200/400/600 state, optimizer, ONNX, sidecar and three-case Torch reference is retained. All updated states and gradients were finite, frozen normalization buffers remained identical, and all3444 TRAIN source hashes were unchanged. Completion receipt SHA256 is `e6ec915764ef9fecf24d992ba17187e0f89e78a34b493eef168aec61d0f135c5`. This observed walltime includes contending construction CPUORT1 validation and has no inference-FPS credit. No checkpoint had numerical/calibration feedback before the complete fixed fit finished.

All three actual GPU numerical checks passed the fixed maximum-error0.01 guard. Errors for TRAIN/zero/255 were200:[0.000167788,0.000602171,0.000133559],400:[0.000117611,0.002215677,0.000098001],600:[0.000218627,0.000334272,0.000353981]. The actual GPU/f16/one-stream/LATENCY/default-Winograd recipe, finite bounded maps and nonconstant TRAIN output passed. All600 optimizer updates finished before these checks. Passing-only calibration then started; concurrent CPU media/private diagnostics give this stage no FPS or latency-qualification credit.

## Actual calibration and global freeze

Each of the three numerical-passing models processed the exact same80 protected calibration views. The existing search/rank was retained; only the predeclared zero-alarm/all10/precision guard determines qualification.

| Checkpoint | High / minimum cells / growth | Meaningful localization | Normal-view alarms | Pixel precision | Pixel recall | Micro IoU | Worst positive IoU |
|---|---|---:|---:|---:|---:|---:|---:|
| 200 | .99 / 32 / .9 | 10/10 | 0/70 | 70.907% | 84.568% | .627836 | .220888 |
| 400 | .99 / 16 / .7 | 10/10 | 0/70 | 70.422% | 84.682% | .624650 | .242190 |
| 600 | .99 / 32 / .5 | 10/10 | 0/70 | 65.094% | 87.712% | .596543 | .255695 |

The fixed weakest-positive rank selects600. Its greater pixel recall comes with lower precision and micro IoU than200/400; this is not an across-the-board gain. All three outcomes are retained, with no later checkpoint substitution from original results.

The global freeze SHA256 is `a3e4cde31918dd21681502c243201852787a31ce4ccc1d8680e7fb983883fba3`. The selected file is `models/factory/supervised_round5_thin_cldice-step600-zero-alarm.onnx`, model SHA256 `cd4c54ee35fb1491ffb63a6b2b86bee8be8021153712306a81e650e0f80a2902` and metadata SHA256 `77cc51a7103a5e2ef8003329af2d86082c8f024dc296620ce8d4882c4ab3a2cd`.

No original-comparison or speed benchmark had run at this freeze. Root must run the fresh capacity/original/paired/sustained gates; no default model was promoted. In the exposed80 comparison, the69 unaffected views belong to ten defective physical items, with no completely normal products. Normal-view alarms therefore do not measure good-product false-rejection rate. Calibration and this repeatedly exposed comparison are development evidence, not independent production validation.
