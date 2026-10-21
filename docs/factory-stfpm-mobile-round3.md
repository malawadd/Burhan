# Factory L: normal-only multiscale student–teacher

Status: **initial capacity passed; numerical parity failed; stopped before fitting/CAL**. The first authorized results are recorded at the end of this document. No defect accuracy has been measured for L. The [earlier metadata-only proposal](C:/Users/IAM/Documents/hackathon/opencv/project1/output/optimization/factory-round3/spatial_round3l_stfpm_mobile_preflight-metadata-proposal-only.json), SHA `f84b55a47d12dfe748903ea2e6310fec6f8ea06172b01b345375cad7d883ef3f`, is retained. The new concrete protected protocol below supersedes its provisional normalization details before any NN.

The prior C-density fusion failed the coordinator's frozen-winner original audit (6/11 meaningful localizations,2/69 normal false alarms); its failure, graph, source snapshot and all nine CAL results remain unchanged. L does not reuse that bank or combine its masks with another model.

## Graph and score

[STFPM's feature-pyramid learning method](https://arxiv.org/abs/2103.04257) motivates an independent MobileNet adaptation. This is our experiment, not a reproduction of the paper's ResNet benchmark or a claim based on its AUROC. See the [primary-source research and alternative analysis](C:/Users/IAM/Documents/hackathon/opencv/project1/docs/factory-normal-learning-next-proposal.md).

The [new graph](C:/Users/IAM/Documents/hackathon/opencv/project1/scripts/factory_stfpm_graph.py) contains two separate MobileNetV3-Small first-seven-block encoders. The frozen teacher strictly loads the existing official torchvision ImageNet checkpoint with `weights_only=True`. The student has independent seeded random convolution weights and Torch-default BatchNorm affine ones/zeros. Taps0/3/6 supply16/24/40 channels. Input remains BGR uint8 at1280×512, with RGB/255 and ImageNet normalization inside the graph; output is one640×256 score map. No256-square global squeeze, C model, density bank, ROI, GT input, raw-pixel bypass or mask OR enters the forward path.

Each feature vector first divides by its channel-wise maximum absolute value, floored at2^-8. Its scaled squared norm is floored at2^-12 before square root and division. Max scaling keeps the reduction bounded even for finite half-maximum feature values; both floors are exactly representable in FP16 and avoid zero division. Each level's disagreement is `.25 * sum((unit_teacher-unit_student)^2)`, clamped[0,1]. Middle/coarse disagreement maps are bilinearly projected to the fine grid. The final score is the equal geometric mean using log arithmetic, with map floor2^-14 to avoid FP16 product underflow. The explicit `output_logits=False` metadata prevents the runtime from applying a sigmoid.

This is a bounded **uncalibrated anomaly disagreement**, not defect probability, confidence in construction safety, physical width, or guaranteed localization. Several-scale disagreement can still suppress tiny defects or overreact to normal texture. Only measurements can establish whether it improves factory results.

## Normal data and bounded fit

Only208 established GT-empty original KolektorSDD TRAIN views/30 item groups are eligible. Exact image/mask identity is `f7ec8222431065047de889ff2b03f24172bc59257d6962c51103377566f733b4`. Source hashes and decoded-empty masks must pass before the optimizer; hashes are checked again after fitting. The31 positive TRAIN views, SDD2, CAL, comparison and construction assets supply no normal targets. No defect pixels are inserted, and no generated inspection evidence is created.

The fixed proposal is600 CPU1 batch1 updates, AdamW LR .001/weight decay .00001, saves200/400/600, no early stopping. Each208-view cycle is shuffled without replacement from seed20261030: every source appears twice and184 sources a third time. Each actual source is resized once to512×1280 with INTER_AREA, then teacher/student receive the **same observed640H×512W full-width crop**. Crop Y is0 with10% probability,640 with10%, otherwise uniform integer0..640. There is no warp/flip/rotation or invented border. Per-source native shape/std and per-step crop coordinates/std/visits are recorded.

Teacher weights and BN remain frozen/eval throughout; their complete state hash must match at every export and completion. Student BN learns normal-only running statistics (default momentum .1), and each exported checkpoint freezes its actual saved BN state. The finite-state guard includes BN buffers, not just parameters. Loss is the equal average of per-level normalized teacher/student squared feature error. Initial teacher and random-student hashes must equal the graph that actually passed speed/numeric preflight. Complete600 history and all three exports/reports are mandatory before global freezing.

## Protected gates and reproducible commands

The initial [protocol before any NN](C:/Users/IAM/Documents/hackathon/opencv/project1/output/optimization/factory-round3l/protocol-before-any-NN.json), SHA-256 `9980d288dbb2754735d8c77c0bcfb3b763ec9290942851973f614041ab916f2e`, remains unchanged. **The active protocol is the additive [v2 score-semantics protocol](C:/Users/IAM/Documents/hackathon/opencv/project1/output/optimization/factory-round3l/protocol-v2-score-semantics-before-any-NN.json), SHA-256 `84e3ab65009e40a5e13eb8f0e241ef2ea766aff1431841e1c75ae5e81a0f4720`.** It binds graph/trainer/math/export/probe/calibration/freezer, serving/profile/metric dependencies, teacher identity, exact normal rows, first16 preflight sources, fixed grids and all policies. Merely creating it authorizes no model execution. Root must separately release the GPU and project-quiescent window.

Before any L NN, root corrected a metadata-only runtime description: neural `SurfaceDefectModel.predict` now passes through an optional explicit `score_semantics`; old metadata retains the prior description and all maps/candidates/thresholds are identical. L now supplies `Bounded_multiscale_teacher_student_anomaly_disagreement_not_probability`, together with `output_logits=False`. Otherwise this non-sigmoid graph would have inherited the runtime's hardcoded sigmoid-score wording. V2 pins the changed runtime SHA `cb85bcfff2c8ac04d78b9c26165a798742b518dca569d2d295e7b2b0c7f7f794`; graph, numerical constants, fit recipe and grids are unchanged. The original9980 protocol, first fixture receipt and [eight-file pre-NN source snapshot](C:/Users/IAM/Documents/hackathon/opencv/project1/output/optimization/factory-round3l/pre-NN-v1-L-source-snapshot/snapshot-manifest.json) are preserved, rather than rewritten to disguise the amendment.

Authorized preflight sequence, once explicitly released:

```powershell
.venv-roads-export\Scripts\python.exe -X utf8 scripts/prepare_factory_stfpm.py --export-preflight --root-released
.venv\Scripts\python.exe -X utf8 scripts/probe_factory_stfpm.py --preflight --root-released
.venv\Scripts\python.exe -X utf8 scripts/probe_factory_stfpm.py --parity-step 0 --root-released
```

The export must be a nonconstant actual two-encoder graph. Speed uses the first16 pinned real normal TRAIN views,4 warm calls then64 complete FactoryProfile calls with fixed high .05/area4/no growth, all tails retained and all calls reaching the NN. Mean serial throughput must reach15/s. Source/geometry/finite/bounded checks and actual Torch-versus-GPU maximum map error≤.01 use one fixed normal TRAIN input plus zero/all255 **initialization tensors**. These tensors are numeric fixtures, never inspection evidence. A failure stops before fitting; first failed/partial artifacts are preserved without an automatic smaller-input fallback or rerun. Final fitted-winner paired capacity and sustained runtime remain separate mandatory gates even if preflight passes.

Only after fresh capacity/numeric/source gates, independent review and root fit release may this fixed trainer run:

```powershell
.venv-roads-export\Scripts\python.exe -X utf8 scripts/train_factory_stfpm.py --root-fit-released --contention-note "Actual coordinated workload recorded here"
```

Each of200/400/600 then requires its actual GPU numeric check before CAL and its own immutable score cache bound to model/protocol and ordered80 source/image/GT/native-geometry hashes. Complete CAL identity is checked **before any CAL NN compile/inference** and source hashes again afterward. Exact CAL identity remains `7b09d8cbf39129dad144fcad6ecbad0ade0bb19e39de6e8fbdfc634eb6ed6d07`. Cached maps may recover the same fixed CPU search; this does not rerun a neural candidate or change a point.

```powershell
.venv\Scripts\python.exe -X utf8 scripts/calibrate_factory_stfpm.py --step 200 --root-cal-released --contention-note "Actual coordinated workload"
# Repeat protected steps400 and600 only after their exact exports and GPU release.
.venv\Scripts\python.exe -X utf8 scripts/freeze_factory_stfpm.py
```

The predeclared wide score grid spans2^-14 through.99; areas1/2/4/8/16/32; growth values are fixed in the protocol. Inclusive thresholds,8-connectivity, top20 stable seed handling, original-pixel projection and metrics remain the runtime reference. The seed shortlist is the existing best-area/high then top6 heuristic, not an exhaustive optimum. Final guards remain normal FP≤1/70 and foreground precision≥.65. The fixed ranking prioritizes meaningful mask-IoU≥.1 recall, weakest-positive IoU, coverage≥50%, weakest coverage, micro IoU, fewer normal FPs and pixel recall. Calibration and freezer both verify the exact ordered raw Cartesian grid, stable top6 seed/growth coverage and recomputed selected rank winner; a missing or stale candidate cannot silently win.

No cache acceleration has been introduced here. The previous C all-nine actual reported elapsed sum was379.958s, not the calendar gap. Semantic-equivalence fixtures remain required before any later acceleration. Root alone may audit the single globally frozen winner on the exposed original comparison; all losing checkpoints remain unscored there. Official SDD2 test remains sealed. No application default was changed.

## Preparation validation

Eight original hardware-free NumPy/AST/source-metadata fixtures passed in .236s (receipt wall .240s): finite FP16 maximum/zero/tiny/opposite feature cases; geometric mean and exact half floors; matching/different directions; deterministic600-view coverage and observed crop bounds; source SHA/physical split tampering; false-alarm/precision/nonfinite/off-grid rejection; complete search/stable winner checks; separate student initialization/frozen teacher source contract. All implementation source files also parsed successfully without imports. No Torch model was instantiated for these checks. The additive v2 suite has **nine passing fixtures in .477s**, including an actual `SurfaceDefectModel.predict` call with a fake NN that proves metadata semantics pass through while arrays/candidates and the old/default description remain unchanged. It performs no neural inference.

Independent peer static review found no remaining material geometry/GT-leak blocker after full-BN finite-state and complete-search hardening. [Original fixture receipt](C:/Users/IAM/Documents/hackathon/opencv/project1/output/optimization/factory-round3l/preparation-fixtures.json) and [active v2 fixture receipt](C:/Users/IAM/Documents/hackathon/opencv/project1/output/optimization/factory-round3l/preparation-fixtures-v2-score-semantics.json) record exact source pins and zero image decodes/weight loads/NN execution. Current scientific outcomes: **accuracy not measured; speed not measured; fit not started**.

## First authorized execution: numeric guard failure retained

After root's explicit PREFLIGHTLRELEASE and project-heavy-job hold, protected v2 source pins verified unchanged. Export completed once, including the strict seeded teacher/student hashes and nonconstant fixed normal TRAIN map range .378027–.643247. Actual ONNX SHA is `0383b508411a4a86c6f7d0cef17b348a6f17b27f3076e0cdb2f48bf66168566d`; initial sidecar SHA `a79d03dbf490188caeecf9c6a2d5a7ff060b06045dff24642d48859862802e2b`; Torch reference SHA `0ac945a1673196860f3b78510b1ac12bd6b9c495b6bfefa2d129c9c769f88592`. The actual teacher feature state is `64ceb1279f4bf4bd460bb5fb3738fdb52795dff8d7b3fcda623a55e8be712feb`; random student state `d5b3b43c0639acd33562433bf7ea8011a5c87f1c3e1f2a124f86ca7c272e5f92`.

All64 complete FactoryProfile calls reached the actual NN. Fresh nonconstant preflight **18.38839 calls/s**, mean54.382ms, P5053.388/P9567.389/P9978.249/max81.201ms; startup10.392s. The mean-capacity gate passed. P95 exceeds a66.7ms15FPS interval, and these64 calls do not prove sustained deadlines, factory coverage or final fitted throughput. Every first-run tail is retained; the host was project-quiescent but ordinary services were not controlled.

The subsequent actual GPU.0 parity check **failed** before any fit/CAL: maximum absolute map differences versus Torch were **.0953511** on the fixed normal TRAIN view, **.0568452** on zero initialization and **.0597667** on all255 initialization. All scores were finite, bounded[0,1] and nonconstant, but each difference exceeded the protected **.01** limit. No tolerance, input size, graph, weights, BN, metric or threshold was changed to pass. The process exited1 and all own compute was released. Speed success cannot override this failure.

| Preserved receipt | SHA-256 |
|---|---|
| `spatial_round3l_stfpm_mobile-step0-export.json` | `df98576f1ef7f21c4bf5a8206afefb7224305e5bb907ed88901ca05883d51a0b` |
| `fullprofile-preflight.json` | `e90c0d84e4a189352708ce616d21b53cebd17b78a3ac0b0cf68c42199b52a7e6` |
| `spatial_round3l_stfpm_mobile-step0-gpu-parity.json` | `d4bd1178a2daa867fdb21ecdd6dfa156a328903015140b04d0d7d30afcd06fac` |

These files are under [the L output directory](C:/Users/IAM/Documents/hackathon/opencv/project1/output/optimization/factory-round3l). The inherited initialization report uses a generic warmup text mentioning sigmoid; actual `output_logits=False` and the forward/readback code perform no sigmoid for L. Score metadata explicitly identifies bounded non-probabilistic anomaly disagreement. Any subsequent precision/backend investigation requires a separately pinned diagnostic version; no retry or correction is silently substituted into these first receipts. **Fixed600 fit and all CAL remain blocked.**

Separately authorized L2 backend/precision diagnosis and ONE L3 no-Winograd candidate also failed to produce a numerically qualifying GPU deployment path. [Complete diagnostic outcomes](C:/Users/IAM/Documents/hackathon/opencv/project1/docs/factory-stfpm-numeric-diagnosis-round3.md) retain all sources, exact queried GPU properties and failures. CPU FP32 matches Torch near .000002, but the GPU discrepancy remains about .095 even with the explicit FP32 hint or disabled Winograd. Neither diagnostic version reached a subsequent speed test or training. The prepared600-step recipe remains unexecuted.
