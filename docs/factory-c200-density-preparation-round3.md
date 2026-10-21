# New C200 shared-feature density experiment

This is a distinct C200 experiment, prefix `spatial_round3k_c200_density_f16`. It preserves the original learned context/packed-phase semantic model and fits a new normal-density head. It does not reuse F2's bank, compiler-context graph, thresholds, or speed receipts. All original, F/F2 and curvature artifacts remain separate.

## Source and protected protocol

The [protocol](C:/Users/IAM/Documents/hackathon/opencv/project1/output/optimization/factory-round3/spatial_round3k_c200_density_f16-protocol-before-fit.json), SHA-256 `69433138b7a8a8c8d18ac68ae901487647d5e103db8330c2a7f8ece16bf0662a`, was recorded before any new NN. It pins every implementation file and the immutable C source:

| Source | SHA-256 |
|---|---|
| C200 PT | `ff3b5571295e80978b4be873b6cce24b7cb947ae25415959a4121337dfe22112` |
| C200 ONNX | `ca9b90e33f1cacc8cd83ccc8e406b32faa8f45947fd2d672dc48e678b20d7318` |
| C200 metadata | `6d727b95adef756399dd72a9d0a6d814c4ce2a4b72c7cf4ee15abd4a3b702bdb` |
| C200 global freeze | `c3c2a6d86f86d625f9f6499c4d7477a25c2b9610b26a870be994a29ac969db40` |

The208 original GT-empty TRAIN identities are crosschecked against the independent239-view TRAIN source inventory, not merely a self-consistent new list. Only the established source-identity list is read from the old F2 fit receipt; no feature moments, bank tensors, quantiles or fitted graph are copied. Bank fitting revalidates all239 actual TRAIN pairs, excludes31 positives, and requires the exact protected208 image/mask hashes before/after its passes. No CAL/comparison/official-SDD2-test input is used for bank statistics.

## Geometry and numerical contract

[The new graph](C:/Users/IAM/Documents/hackathon/opencv/project1/scripts/factory_c_density_graph.py) loads C's entire state strictly, freezes every parameter and runs in eval mode. It executes the encoder once, retaining first-block16-channel fine features. It preserves the additional context branch, bilinear semantic upsampling, actual gray/local-contrast pixel-unshuffle, learned four-phase residual, and full-resolution addition. **The original2×2 max-logit reduction happens before density fusion.** Fine and pooled semantic grids are both640H×256W.

The single serving output is `alpha*pooled_C_raw_logit + beta*signed_density(fine)`, with sigmoid afterward in the unchanged serving path. There is no mask OR, second encoder, ROI, part-ID rule or alternate inference fallback. Fixed candidates are `(1,0)` control and `alpha∈{.5,1} × beta∈{1,2,4,8}`.

Normal statistics come only from a new **full three-output C semantic+density diagnostic graph**. The offline feature extractor is a byte-identical copy retaining all outputs; a fine-only exported wrapper would prune the context and risk the compiler mismatch found in F. The new bank uses F2's corrected numeric method: float64 conditional mean/moments; actual residual covariance around deployed nearest-half mean; fixed shrinkage/floor; bounded whitening rounded toward zero before quantiles; quantiles from the same fixed TRAIN-only samples; independently stored half-representable q95/qspan. Both graph/reference read those exact constants. Clipping guards remain fixed and cannot be adjusted from CAL.

Six pure CPU fixtures passed in .059s before protocol: AST equality with original C operations plus explicit maxpool placement, density-after-pool expression, exact-nine/geometry guards, rounded-mean covariance against empirical residuals, half-whitening bounds, and independent qspan despite rounded q50/q95 collision. [Fixture receipt](C:/Users/IAM/Documents/hackathon/opencv/project1/output/optimization/factory-round3/spatial_round3k_c200_density_f16-preparation-fixtures.json) SHA-256 `f2004e4fd35e301ce8d6c6a491faf5f5f4ce04b343faeaf002221f0ef541c54d`. These are numeric/software fixtures, not artificial inspection defects. No dataset pixels or NN were used by them. The initial pre-review fixture receipt is separately preserved.

Independent source review found a runtime-environment import issue before protocol: the dummy-bank helper would have imported Torch in the app environment. It was moved to [NumPy-only math](C:/Users/IAM/Documents/hackathon/opencv/project1/scripts/factory_c_density_math.py) shared by export/checker before hashes were frozen.

## Authorized first nonconstant preflight

The coordinator explicitly released an exclusive project preflight slot after the previous timing window; other project NN/tests/fits/large IO stayed held. Ordinary host services were not controlled, so this is not whole-host-exclusive. Only new dummy export,64 complete profile calls on16 real TRAIN views, and fixed TRAIN/zero numerical parity were authorized. No bank was fitted during this stage. First results and tails were retained; no retry or guard relaxation occurred.

| New unfitted C-density preflight | Actual result |
|---|---:|
| Timed complete profile calls / real TRAIN views | 64 /16 |
| Serial full-profile capacity | **19.21635 calls/s** |
| Mean /P50 /P95 /P99 /max process latency | 52.0390 /50.8612 /59.4198 /61.4327 /63.2220ms |
| Compile/startup | 10.866s |
| Original versus copied C pooled Torch logits | exact |
| Actual GPU source versus control raw-logit max error | 0 |
| Original frozen C-policy mask parity | identical |
| Full-context extractor versus diagnostic fine-feature error | 0 |
| GPU versus exact-constant NumPy dummy evidence max error | .00341225 ≤.1 |
| Fixed TRAIN/zero outputs | all finite |

The GPU check uses one fixed original TRAIN image and zero initialization input. It supplies numerical agreement, not defect accuracy or population coverage. Fitted candidate7 versus diagnostic single-output parity is a later mandatory guard; its pre-fit report's inherited true/null placeholder is not a performed fitted-candidate measurement.

Receipts under `output/optimization/factory-round3/`:

| Receipt suffix after prefix | SHA-256 |
|---|---|
| `export-manifest.json` | `e573d48dab15099263816785b9e5df881b94004556c1e1f76a41e23c5b67a8a2` |
| `export-parity.json` | `6cc50161d26fb701fbd1a44daf02036e2ca0006210f481b6cfe099e9d47eed3a` |
| `fullprofile-preflight.json` | `1bd8c2be81da113d9de9a9228106291bc32c21a44909687e75ae21603f698440` |
| `pre-fit-gpu-parity.json` | `3dee4cea2a9efc27d486db33ec22bc26f4f3ea0b34d4efc2e0585928cd567da4` |

The actual new probe SHA is `a3f7f56d7325bc779ac177ddb09b5a9fbd5339f616f74966fb50f262e8ce53e5`; control `1315d907697a5f2bb635c3488aa6d273217ba1e97679efa720e035cfb5011207`; complete diagnostic/extractor `68ac7b76aba848b4b1eb6e3ff3141a51c7f2115df8a6ba4e63a028bb1fb46012`. This fresh capacity gate does not establish final fitted-model/video speed or detection accuracy.

## Separately authorized bank and later gates

After successful preflight and explicit release, the coordinator separately authorized the exact208-normal bank, fitted exports/parity and fixed clipping diagnostics. Pixel CPU1 fitting may overlap these accuracy jobs; their elapsed/latency values cannot qualify capacity. [Executed-source snapshot](C:/Users/IAM/Documents/hackathon/opencv/project1/output/optimization/factory-round3/c200-density-executed-source-snapshot-v1/snapshot-manifest.json) SHA-256 `b4389ec28f7d802925327639aee2217abc37e4c4c1df721a1aa53d3bb3e2e559` precedes bank fitting and any shared-source edit.

Only after all fitted numeric/context/source/finite guards pass may actual nine-candidate CAL run. The first GPU handoff goes to construction's fixed four-case video exports; candidate calibration must wait for their release. Every candidate uses its actual single-output GPU graph and exact same80 CAL source identities. Normal FP≤1/70, pixel precision≥.65, the original weakest-positive rank and fixed grid remain unchanged. All nine must finish before [global freezing](C:/Users/IAM/Documents/hackathon/opencv/project1/scripts/freeze_factory_c_density.py); a control winner or rank tie rejects the experiment. No original comparison is scored by this task. Final fitted-winner paired capacity and runtime≥15 remain mandatory after all fits/probes finish. No production, independent-test, physical fault-width or Saudi compliance claim follows automatically.

## Completed fresh-bank/fitted numerical gates

The authorized fresh bank completed in67.683s during accuracy-work contention, with208 normal views,31 positive TRAIN views excluded,34,078,720 feature pixels and425,984 fixed quantile samples. Source identity is `f7ec8222431065047de889ff2b03f24172bc59257d6962c51103377566f733b4`. The bank is new C data, SHA-256 `673b0cc15daab65d06c48b99388734685c10191ada69efc6d63af02d88c42476`; q95−8.28125, independent qspan1.349609375, quantized whitening row-L1 maximum7.9973202. Moments/parameters are finite and the normal-fit passes record zero centered/whitened clipping. The67.683s duration is not an inference-speed benchmark.

All nine fixed exports completed once. A transient chat-agent capacity interruption did not trigger any fit/export rerun: the existing process sessions were inspected and their successful completions retained before proceeding to only unfinished stages.

| Fitted guard | Actual result |
|---|---:|
| GPU original C versus control raw-logit max error | 0 |
| Frozen source-policy masks | identical |
| Full compiler-context extractor versus fitted diagnostic fine error | 0 |
| Actual single-output candidate7 versus diagnostic fused logits | exact, max error0 |
| GPU versus exact-stored-constant NumPy evidence max error | **.00701541 ≤.1** |
| Fixed TRAIN/zero outputs | all finite |
| Centered /whitened clipping /evidence-limit saturation | zero in all groups |
| Fixed clipping observations | 208 TRAIN normal /70 CAL normal /10 CAL positive |

The fixed clipping diagnostic took43.415s and used the already frozen bank/guards. It did not refit statistics or select/relax parameters. Exact original80 CAL source identity remained `7b09d8cbf39129dad144fcad6ecbad0ade0bb19e39de6e8fbdfc634eb6ed6d07`.

| Receipt suffix | SHA-256 |
|---|---|
| `normal-fit.json` | `e03b9f22f07b4612c8d561c98de054efc92ee4ff9a197b339461156b3e0c905a` |
| `fitted-export-manifest.json` | `e782275443e841612d4319f4d89067a6b1968559986893a0d869dcc1fcd0f6d1` |
| `fitted-gpu-parity.json` | `1e8fe2473235ea88f73968800e3cb1f7b1fce9c83ae9497d575c9c58c9e661d6` |
| `guard-clipping-audit.json` | `ad2c72c1f5b0a0c6f16b44b6390ad9e607694f4d6e311af47a6d73f69afee6b6` |

An inherited observational string in the immutable clipping script/report incorrectly said constructionCPU4 fitting was concurrent. That fit had completed and pixelCPU1 v1 had already stopped on its provenance exception by the clipping stage. Earlier bank/export/parity could overlap pixelCPU1; the later pixel v2 restart followed clipping/GPU handoff. Ordinary host/light metadata activity was not controlled. The original report is preserved, and a [workload-only amendment](C:/Users/IAM/Documents/hackathon/opencv/project1/output/optimization/factory-round3/spatial_round3k_c200_density_f16-guard-clipping-workload-amendment-v1.json), SHA-256 `5ab514e31db665d4b324375fa7db1a9b169cf8337a3e1c03a5075125bbfdc654`, records the correction without changing any metric, bank, guard, protocol or source code.

All own NN processes exited and GPU was explicitly handed to construction for its fixed4-case/two-model whole-video export. After all eight videos completed, construction explicitly returned the GPU for the authorized fixed nine-candidate calibration. Numeric success is not evidence of defect localization, and no new original/comparison/heldout inputs were opened by this task.

## All nine calibrations and protected global selection completed

The first actual calibration process completed successfully once, without cache reuse or candidate reruns. Every candidate used the same protected80 CAL views (10 defective/70 normal), actual single-output GPU graph and unchanged search/guards. Source identity remained `7b09d8cbf39129dad144fcad6ecbad0ade0bb19e39de6e8fbdfc634eb6ed6d07`. Pixel CPU1 work was coordinated separately and ordinary host activity was uncontrolled; these accuracy-job timings cannot qualify inference capacity.

| ID | Semantic α | Density β | High /area /growth | Meaningful localized views | Normal FP /70 | Pixel precision | Pixel recall | Micro IoU | Worst positive IoU |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 0, C control | 1 | 0 | .999 /1 /.8 | 10/10 | 0 | .669010 | .869830 | .608126 | .302386 |
| 1 | .5 | 1 | .99 /8 /.7 | 10/10 | 1 | .746989 | .806139 | .633237 | .317476 |
| 2 | .5 | 2 | .999 /16 /.7 | 10/10 | 1 | .755606 | .784163 | .625514 | .316738 |
| **3, global winner** | **.5** | **4** | **.99999 /16 /.25** | **10/10** | **1** | **.669141** | **.835667** | **.591330** | **.344167** |
| 4 | .5 | 8 | No eligible point | — | — | — | — | — | — |
| 5 | 1 | 1 | .9999 /1 /.9 | 10/10 | 0 | .760871 | .807630 | .644133 | .311733 |
| 6 | 1 | 2 | .99999 /1 /.9 | 10/10 | 0 | .774474 | .787729 | .640750 | .314985 |
| 7 | 1 | 4 | .999999 /16 /.25 | 10/10 | 1 | .653722 | .874660 | .597727 | .333915 |
| 8 | 1 | 8 | No eligible point | — | — | — | — | — | — |

All eligible points localized10/10 at mask IoU≥.1, covered at least50% of GT in9/10, and abstained on zero inputs. Candidate3 wins the **predeclared weakest-positive rank**, with worst IoU .344167 versus control .302386. It does not win every metric: CAL normal false alarms rise from0 to1, micro IoU falls from .608126 to .591330, and pixel recall falls from .869830 to .835667. No threshold, ranking, model or false-alarm guard was changed to obtain or excuse that outcome. IDs4 and8 have no point satisfying the fixed normal-FP≤1/70 and pixel-precision≥.65 guard; their complete reports remain retained.

[Global freeze](C:/Users/IAM/Documents/hackathon/opencv/project1/output/optimization/factory-round3/winner-c200-density-freeze-before-comparison.json) SHA-256 **`030281c7b8da5d98f2b84362c7a5059b13ffaeaf7daf9eaae3addee112cae070`** accepts only candidate3 for the coordinator's independent audit. [All-nine report](C:/Users/IAM/Documents/hackathon/opencv/project1/output/optimization/factory-round3/spatial_round3k_c200_density_f16-calibration.json) SHA-256 `77682123820a7533c3f752113cc6e4738552880dbe3cecf5af738e443a851cf2` binds all fixed outcomes.

The frozen model is [candidate3](C:/Users/IAM/Documents/hackathon/opencv/project1/models/factory/spatial_round3k_c200_density_f16-step3.onnx), SHA-256 `9532accab4f24b2824c7a03bcb84824ec088d92d452c876264910786969128a2`; its sidecar SHA-256 is `4366826008e001da6dc522c1e0be04f84979f843b89b3f7ee84b9338e40f03b4`. The serving point is high .99999 /minimum area16 /growth .25. No application default or registered production choice was changed.

All own NN and calibration processes exited before GPU release. This task performed **no original/comparison NN evaluation**, selected no losing candidate from comparison outcomes and opened no official SDD2 test. The coordinator must audit only this frozen winner. New fitted-model paired capacity and sustained runtime remain separate outstanding gates; the earlier dummy-head19.2163/s preflight cannot substitute for them. Factory recall improvement, fewer false alarms, production readiness and15+FPS final operation remain unproven at this stage.

## Coordinator's frozen-winner original audit: failed

After global freezing, the coordinator evaluated **only candidate3** on the original exposed development comparison. The [actual audit](C:/Users/IAM/Documents/hackathon/opencv/project1/output/optimization/factory-round3k/after-localization-audit.json) records meaningful mask localization in **6/11 defective views (54.5%)**, any-pixel overlap in7/11, normal false alarms **2/69**, at least50% GT coverage in6/11, pixel recall54.14%, pixel precision65.79%, micro IoU .42248, mask-IoU≥.5 recall5/11 and bounding-box-IoU≥.5 recall6/11. No inputs abstained. The factory target is **not met**; the new fusion is rejected and is worse than the existing C baseline's10/11 meaningful localizations and1/69 false alarms.

This failure remains part of the evidence. Calibration rank improvement did not generalize to the exposed comparison. No other fusion, point, ROI or losing checkpoint was subsequently scored to rescue it; no original pixels were used by this task for fitting, mining, new masks or thresholds. Official SDD2 test remains unopened, application defaults remain unchanged, and numeric parity/preflight speed do not override failed localization. Further factory work requires a materially different, separately protected experiment rather than relabeling this candidate as successful.

## Completed fitted-winner paired speed characterization

The coordinator subsequently characterizes the already rejected winner without another selection or original-comparison inference. The [pre-timing plan](../output/optimization/factory-round3/k-rejected-winner3-calibration-capacity-plan-before-timing.json), SHA256 `4298bbdf9ed33409d2ad2f3bbab75206ac4127e4516a1024363ecbf60273a210`, pins the actual model/metadata, current runtime sources, unchanged policies, identical80CAL views and exact sequential commands. All project fits, other inference, dataset I/O and tests are stopped for the timing window; the app station is stopped. Ordinary host services remain active. All pinned runtime source hashes are verified unchanged after the window. The first results are retained, with no better-tail repeat.

| Workload | Preserved v10h before | Frozen K3 after |
|---|---:|---:|
| Alternating full-profile calls,3×80 predecoded CAL views |26.425 calls/s|18.875 calls/s|
| Mean /P95 /max profile time |37.843 /45.927 /62.620ms|52.979 /62.566 /76.658ms|
| Clocked16-input-FPS,60seconds captured /processed |960 /960|960 /960|
| Drops /source-skips /inspection errors |0 /0 /0|0 /0 /0|
| Decisions later than100ms |0|10|
| Runtime receipt-to-decision P95 /P99 /max |42.2 /52.7 /86.9ms|77.5 /100.8 /116.2ms|

These measure **the actual fitted K3 artifact**, unlike its earlier dummy-head preflight. It sustains the input count in this replay but has10 late decisions and substantially worse original localization. It remains rejected. Profile call throughput excludes file decoding, capture/queue, UI, storage and actuation; runtime timing includes the documented software-clock replay path. Neither measures moving factory video, camera exposure/buffering, PLC rejection or physical every-item coverage.

All three source reports and before/after host snapshots are under [factory-round3k-capacity](../output/optimization/factory-round3k-capacity). The unrelated Stage project Python process PID26076 measures104.6%/105.4% of one logical core in the two snapshots; it was not controlled. The sampler's own CPU use is outside the timed runs. These are snapshots, not proof of the cause of any individual latency tail. No host service was paused or stopped.
