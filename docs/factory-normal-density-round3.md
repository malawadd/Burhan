# Factory round3: shared-feature normal-density complement

This work follows the unchanged10/11 recall ceiling and failed D/E candidates documented in [factory-accuracy-round3.md](factory-accuracy-round3.md). No model here is promoted without independently improving localized recall/normal false alarms and retaining≥15FPS. OfficialSDD2 test remains sealed. The original80 are repeatedly exposed development evidence, not a blind test; no original comparison pixels, labels, IDs or ROIs enter this fitting/calibration/inference pipeline.

**Normal-view false flags are not good-product false rejects.** The original80 contain ten defective physical items and69 unaffected views of those items, with zero fully normal products. Product false-reject rate is therefore unmeasured. Normal-bank labels likewise refer to GT-empty views/regions, not a guarantee that the entire physical product is normal. The coordinator's offline item-grouping receipt leaves primary view/mask metrics unchanged.

## F: a different learned signal from actual TRAIN normals

F freezes the faster original **v10h** segmentation weights: PT SHA`eb2652b23e3e0cf63a7396902bc99cdb068780d02347df16c8e9cd1b545c9ba1`, sourceONNX SHA`dd5019cbcbc859c4eeb994a7639fb0c520ee3e3175e47a3f1544ec4c5dd60753`. The faster backbone leaves more measured laptop headroom than C's added context. One graph reuses its existing stride2/16-channel first-block features, keeping input1×3×1280×512 and output1×1×640×256. There is no second runtime model, network upload or frame-level fallback.

Only the208 GT-empty views among239 original TRAIN views from30 physical items fit the new statistical head;31 actual-defect TRAIN views and all SDD2/calibration/comparison images are excluded from the bank. This is a normal-only statistical component attached to a supervised source segmenter, not an entirely unsupervised system. Streaming float64 moments fit a16×256 column-conditioned mean and global16×16 residual covariance. Fixed10% diagonal shrinkage and eigenvalue floor`max(1e-6,.001*median nonnegative diagonal)` regularize whitening. Global rescaling bounds maximum whitening row-L1 at8. Centered features/whitened values clip±32; density is channel mean of squared whitened values. Guarded log-distance uses`log(max(D,1e-6))`.

Normal q50/q95 come from a second fixed normal-only pass with2048 pixels/view without replacement, seed20261026,425984 total sampled pixels. Signed evidence is`A=clip((logD-q95)/max(.001,q95-q50),-8,8)`. A single raw score map is`alpha*source_pre_sigmoid_logit+beta*A`, followed by the existing sigmoid/seed/component/growth policy. Common normal evidence can suppress a semantic false alarm while unusual evidence can strengthen a weak signal; **there is no OR union or full-frame flagging**. The resulting sigmoid is not a calibrated defect probability.

The nine candidates are fixed before scoring: control`(alpha,beta)=(1,0)`, then alpha∈{.5,1}×beta∈{1,2,4,8}. All actual exported single-output GPU graphs must calibrate independently; CPU-fused scores cannot replace served outputs. The existing final normalFP≤1/70 and pixelprecision≥.65 guard, meaningful-view/worst-positive-IoU rank, high/area/growth grids and exact80-source hash/geometry caches remain unchanged. Every candidate, including control, must finish before freeze. If control wins/ties or a noncontrol has no strict eligible rank gain, reject without a new original audit. Candidate IDs0–8 are **not SGD steps**; source weights/BN never update.

`factory_density_identity.py` binds actual source PT/ONNX/metadata, initial export manifest, bank/fit JSON, exact nine ordered IDs/coefficients/standard paths/actual modelSHAs, diagnostics and calibration rows. Source image/mask hashes are checked before/after normal fitting. Fit, export, parity, calibration and freeze cannot silently mix stale banks/graphs. Independent static review and four independent NumPy numerical-contract fixtures cover whitening orientation, singular floors, bounded arithmetic and invalid inputs.

## F architecture preflight and failed fitted precision guard

Protected protocol SHA:`fa8adad1f5b03d2f840d76c04b52d1b33ad975cc07d9c7594712e44cb86669e9`.

Before bank fitting, a nonconstant dense16×16 whitening placeholder prevents the compiler from removing head cost. In a coordinator-authorized project-quiescent window,16 actual TRAIN views/64 complete-profile calls all execute NN and measure mean42.589ms/P9548.248/P9953.517/max56.671, **23.480 calls/s**, startup7.984s. This is an architecture gate, not fitted accuracy, moving-line coverage or a final sustained-runtime result. Probe SHA:`86cd51c9f647fbd7ff230aecf2253bd06e1c57b56c764afb81e0dc13873936ea`. Ordinary host services remain active; project-quiescent is not host-exclusive.

The copied source path reproduces raw logits exactly in Torch and under frozen-BN train/eval modes. Actual GPU source/control raw logits also match exactly on the fixed TRAIN frame, source-policy masks are identical, and TRAIN/zero-input outputs are finite. Dummy GPU-vs-NumPy evidence maxerror.006278 passes the fixed≤.1 numerical guard. Standalone-versus-full-graph fine-feature maxerror.088867 is explicitly recorded, not treated as bit identity.

The208-normal fit then completes in53.196s during construction CPU4 fitting:34,078,720 feature pixels,425984 quantile samples. Covariance min shrunk eigenvalue1.76434e-5,0floored eigenvalues,global whitening rescale55.7633,rowL1=8; q50−9.3820658/q95−8.0494806/span1.3325852. No centered/whitened normal values clip. Bank SHA:`4eb6b26dcb2a243708883fc4b856bd44da59078a9cd4ff4f692a2f0d691382d6`. Fit/extractor/source hashes and all208 normal source records remain in the fit report. All nine graphs export before fitted parity, without any calibration access.

**F fails fitted numerical parity and stops before calibration.** Source/control logits and actual single-output candidate7 versus three-output diagnostic fused logits still match exactly, with all outputs finite. But actual GPU evidence versus the intended NumPy reference differs by.796028, exceeding the unchanged.1 guard. No CAL/original image is opened, no threshold/weight grid is changed, and the failed bank, nine graphs, manifests, error report and console logs remain unchanged.

## Bounded TRAIN-only numerical diagnosis

The coordinator authorizes only the same fixed TRAIN frame and zero initialization input, no bank refit, CAL or original scoring. All actual GPU graph SHAs, compiler properties and input SHA are recorded in `spatial_round3h_normal_density-numeric-diagnosis.json`. The measured decomposition is:

| Reference to actual fitted GPU evidence | TRAIN max error | TRAIN P95 | TRAIN P99 | Zero max error |
|---|---:|---:|---:|---:|
| NumPy using standalone fine features | .796028 | .145701 | .240142 | .019719 |
| NumPy using actual full fitted-graph fine | .495292 | .114537 | .184886 | .014502 |
| Same full fine, constants rounded to actual f16 representation | **.008186** | **.003207** | **.003844** | **.003423** |

Pre-fit full-graph versus fitted full-graph fine features match **exactly**, while standalone/full features differ by max.088867 on TRAIN (P99.012085) and.008545 on zero. Every2,621,440 feature value and163,840 evidence value per case is finite. Thus changing feature context alone is insufficient; constant representation also matters. The rounded-reference diagnostic explains the discrepancy but **does not retroactively pass F or relax its guard**.

## F2: separately protected numerical correction

New prefix:`spatial_round3i_normal_density_f16`; protected protocol SHA **`1b1a4761e697f5deec693d24b52d0be4ac431a32b2697dc3e432a71cca77965e`**. Shared scripts use an explicit precision-corrected flag; all F artifacts remain preserved.

F2 fits the same208 normals using the byte-identical pre-fit **full** diagnostic graph with logits, evidence and fine outputs retained, selecting its unique16×640×256 fine output. Exporting only fine would allow pruning and is expressly avoided. This graph's full fine is known to match the fitted full context in the bounded diagnosis. Source/manifests/hashes pin its identity before fitting.

The fixed computation order is:

1. Collect actual full-context fine-feature moments mu64 and E[ffᵀ].
2. Round deployed mean muq to exactly f16-representable float32 values.
3. Center residual covariance correctly around muq: `E[ffᵀ]−mean_x(mu64·muqᵀ+muq·mu64ᵀ−muq·muqᵀ)`. Simply subtracting muq·muqᵀ would be wrong.
4. Apply the same shrinkage, eigen floor and global row-L1 bound. Round whitening magnitudes towardzero to f16 so rounding cannot increase the bound.
5. Run the same fixed normal-quantile sample using the exact deployed muq/Wq. Preserve raw quantiles; round q95 and the **independent** span`max(.001,rawq95−rawq50)` to f16-representable values.
6. Store/read explicit qspan in both graph/reference. Never reconstruct it from rounded q50/q95.

All nine source/fusion declarations, search/rank/false-alarm guards, fixed≤.1 numerical guard and no-original-if-control rule remain unchanged. F2 explicitly reuses immutable F's23.48/s identical-topology receipt **only as an architecture gate**, not as new F2 timing. Final fitted capacity and sustained runtime still need project-quiescent paired measurements. Code/static review passes before F2 bank fitting; source/control/single-output and actual numerical parity remain mandatory before any calibration. Current walltimes overlap construction CPU4 and carry no speed qualification.

F2 normal fitting completes81.830s with the same208 source-identitySHA`f7ec8222431065047de889ff2b03f24172bc59257d6962c51103377566f733b4`. Mean quantization maxerror.00192874; quantized whitening row-L1max7.996465. Raw normal q50/q95 are−9.44397211/−8.11582661; stored q95 is−8.1171875 and explicit qspan1.328125. Bank SHA:`211e3176df214c71e5c2450876a55657d614420bf8c28b1d293e19804a8c5038`.

The fitted F2 actual-GPU guard **passes**: source/control raw logits and masks identical, full-context extractor/fitted fine maxerror0, actual single-output candidate7/diagnostic fused logits maxerror0, and GPU-vs-NumPy evidence maxerror **.007570≤.1**. All fixed-TRAIN/zero outputs are finite. The nine-candidate manifest SHA is`56ede7ac83b0b7d84ee2a36f2d98766236b478021b066e0743fbddc18da53b0d`.

Only after that pass, fixed-guard diagnostics examine all208 TRAIN-normal,70 CAL-normal and10 CAL-positive views with the frozen actual diagnostic head. All three groups have **zero centered/whitened clipping and zero actual evidence saturation at±8**. The64.093-second diagnostic is accuracy-only under construction CPU4 contention; it does not change guards/bank/selection. Its CAL source identity is exactly`7b09d8cbf39129dad144fcad6ecbad0ade0bb19e39de6e8fbdfc634eb6ed6d07`, which every following candidate must match.

All nine actual single-output GPU calibrations then start without early exit. The initial control0 under the strict1/70 normal-view false-alarm guard selectsseed.999/area32/growth.8:7/10 meaningful/any hits,pixelrecall69.76%/precision73.45%/microIoU.55709,worst-positiveIoU0. This is a retuned source control under the new guard, not the deployed baseline's more permissive calibration point. The remaining eight fusions are mandatory before unbiased freeze; no original comparison follows an intermediate result.

## F2 completed all-nine result: rejected before original scoring

All nine actual single-output GPU models complete calibration. The frozen ranking rejects F2: the control retains seven meaningful defect views and seven views with at least50% GT coverage, while every fusion with seven meaningful hits has only six views at50% coverage. No fusion strictly outranks the eligible control. This is a failed accuracy experiment despite successful numerical correction. No original-comparison inference, official SDD2 test opening, fitted-capacity claim or promotion follows.

| ID | alpha/beta | Meaningful views /10 | Normal-view flags /70 | ≥50% coverage /10 | Pixel recall | Pixel precision | Micro IoU |
|---|---|---:|---:|---:|---:|---:|---:|
|0|1/0|7|1|7|69.76%|73.45%|0.55709|
|1|0.5/1|7|1|6|72.16%|74.35%|0.57778|
|2|0.5/2|7|1|6|69.39%|76.70%|0.57308|
|3|0.5/4|4|1|4|53.66%|74.61%|0.45375|
|4|0.5/8|ineligible|—|—|—|—|—|
|5|1/1|6|1|5|61.66%|75.16%|0.51225|
|6|1/2|7|1|6|72.16%|74.35%|0.57778|
|7|1/4|ineligible|—|—|—|—|—|
|8|1/8|ineligible|—|—|—|—|—|

All eligible points have worst-positive IoU0. Candidate IDs4,7,8 have no point satisfying the fixed final guards; they are retained as rejected trials. Raw reports preserve every selected threshold, all sweep metrics, actual model/metadata SHA, ordered image/mask identities and contended per-candidate timing. The strict control rule and ranking were not amended after these results.

Freeze: `output/optimization/factory-round3/winner-attempt7-precision-density-freeze-before-comparison.json`, SHA **`1454427f6eac17891ce59cc527423a1226a056f48f46ded02306d3ca3762666b`**; `accepted_for_independent_audit=false`.

Before subsequent shared-source extensions, the actual executed scripts, all F2 protocol/fit/export/parity/clipping/calibration/freeze JSON and calibrated metadata are copied and hash-verified into `output/optimization/factory-round3/f2-executed-source-snapshot-v1/`. Its32-file manifest SHA is **`e3e94194b1ec0abb08778742585db988e33a462503d34d5e0c0b9387c6bd2746`**. Model/bank binaries remain at the immutable referenced paths, bound by the copied manifests.

All fit/calibration timings overlap the declared construction fit and ordinary services and carry no speed qualification. The new concurrent curvature fit is a separate protected experiment; it does not alter this F2 outcome.

## Prepared post-fit F2 capacity characterization (not executed here)

F2 remains rejected. ID7 has no eligible calibration point or sidecar; no metadata or policy is invented. The coordinator fixes **ID1**, the lowest-ID eligible noncontrol, before any speed measurement. Its actual frozen policy is seed.999/area1/growth.5. The baseline retains its existing policy. Both replay the identical80 calibration views, never the original exposed comparison. This is throughput characterization of a failed candidate, not another selection, adoption or independent accuracy test.

The exact commands, model/metadata/script SHAs and readiness conditions are frozen in `output/optimization/factory-round3/f2-rejected-id1-calibration-capacity-plan-before-timing.json` (SHA `77b7822c95a9af2195aeb3baa49c9092bdb1f79806735bc5d2c10d837525a221`). Execute sequentially only after all fits/export/inference/data-I/O/tests are quiet; record ordinary background services. No NN or performance command below has been run for this tooling change.

```powershell
.venv/Scripts/python.exe -X utf8 scripts/benchmark_factory_accuracy.py --before models/factory/spatial_unet_v10h_sdd2-step300.onnx --after models/factory/spatial_round3i_normal_density_f16-step1.onnx --split calibration --repeat 3 --output output/optimization/factory-round3i-density-capacity/paired-calibration-capacity.json

.venv/Scripts/python.exe -X utf8 scripts/benchmark_runtime.py --model models/factory/spatial_unet_v10h_sdd2-step300.onnx --backend auto --varied --partition calibration --fps 16 --seconds 60 --deadline 100 --preview-fps 6 --trace --output output/optimization/factory-round3i-density-capacity/baseline-calibration-runtime-16fps-60s.json

.venv/Scripts/python.exe -X utf8 scripts/benchmark_runtime.py --model models/factory/spatial_round3i_normal_density_f16-step1.onnx --backend auto --varied --partition calibration --fps 16 --seconds 60 --deadline 100 --preview-fps 6 --trace --output output/optimization/factory-round3i-density-capacity/failed-density-id1-calibration-runtime-16fps-60s.json
```

`benchmark_runtime.py --partition calibration --varied` selects the selected-model metadata calibration items **before any decode**, avoiding the historical kos04/Part3 sample entirely. Default `--partition test` preserves the previous80-view lexicographic order and identifies it as exposed development data. Single-array/LocalSource decode modes preserve their historical sample and reject a nondefault partition. Reports record actual partition, scope, selection metadata path and SHA. Six hardware-free tests pass, covering both80-path selections, historical default order, no image payload/model loading, singleton behavior, invalid mode combinations, split overlap and missing views. The old runner is archived with SHA `165f4fa9c95e32783547a87fb8423c30b238a485023be755f92ba3f265951437` in `output/optimization/factory-round3/benchmark-runtime-before-partition-v1/`.
