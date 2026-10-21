# Round4: supervised model with more real TRAIN data

The previous teacher–student experiments did not produce an eligible detector. This experiment returns to the fully supervised C200 phase-max model: 10/11 meaningful defect views and one normal-view alarm in its exposed development comparison. Its existing speed evidence does not qualify any new weights. The goal remains higher meaningful localization recall, fewer alarms, and measured throughput of at least 15 complete profile calls per second.

## Real source inventory

| Source | Proposed fitting images | Positive / normal | Evaluation scope |
|---|---:|---:|---|
| Protected KolektorSDD1 TRAIN | 239 | 31 / 208 | Existing 80-view calibration and exposed comparison stay outside fitting |
| Official KolektorSDD2 inner FIT | 1,865 | 197 / 1,668 | Reserved 466 and official 1,004-image TEST remain unopened |
| Authors' Magnetic Tile dataset | 1,340 accepted of 1,344 originals | 388 / 952 | Four positive-category empty masks quarantined; all data TRAIN-only |
| **Total** | **3,444** | **616 / 2,828** | More source images are not more independent physical products |

The [official SDD2 source](https://www.vicos.si/resources/kolektorsdd2/) describes real production-item surfaces and pixel annotations, with scratches, minor spots and other imperfections. It is CC-BY-NC-SA4. The [Magnetic Tile authors' repository](https://github.com/abin24/Magnetic-tile-defect-datasets.) supplies actual images and pixel-level labels for six categories, and welcomes research use with citation; it does not provide an explicit commercial license. Its images can include the same tile under different illumination. MTD is therefore an additional training source, with no invented train/test independence or benchmark.

Earlier supervised runs sampled SDD2 sparsely or for a short rehearsal phase. Round4 instead prepares an explicit, hash-bound coverage ledger. Every accepted fitting source must appear, and all projected foreground pixels must occur in the mandatory real positive tiles. No SDD2 reserve, SDD1 calibration/comparison, or private user footage enters the training loss.

## Fixed supervised experiment

The serving architecture, input contract and half-grid max-logit reduction stay identical to C200. This is a new supervised weight fit, not another normal-only anomaly model or architecture/configuration grid.

The bounded proposal is 4,096 transfer updates followed by 900 SDD1-focused updates, CPU1/batch1. Only fine-stage checkpoints 300/600/900 become candidates. The complete source/positive-coverage ledger must fit within 4,096 transfer updates; preparation fails for explicit review if it cannot. It must never silently omit images or foreground pixels to fit the budget.

Transfer uses full-width 320×512 observed patches. SDD1 keeps its existing 512×1280 serving-input projection. New domains scale isotropically to width512, retaining native aspect ratio. Short final tiles may contain input padding, but unobserved pixels have zero loss validity: they are not certified background. Masks use explicitly described binary any-overlap occupancy after area resampling; there is no GT dilation, synthetic defect, or mask-based inference input. This is a training-grid coverage statement, not a claim to preserve every native camera pixel or defect component in one crop.

Every normal image supplies an actual observed crop. Positive sources supply nonoverlapping vertical tiles covering all projected GT foreground. Remaining transfer slots repeat positives across sources. Fine-stage 640×512 full-width SDD1 contexts use real positive points, existing TRAIN-mined hard locations and unbiased crops; every fifth update rehearses a fixed new-domain tile. Crops retain their actual full masks, including defects away from a selected point. Matched flips and modest photometric augmentation apply only to fitting.

The existing positive BCE, hard-negative BCE and FN-weighted Tversky objective is restricted to observed pixels. Targets are float32 and validity is boolean; actual logits train in float32. The teacher–student normalization floors and failed numerical guard do not enter this supervised objective. All model parameters are finite-checked, BN statistics/normalization buffers remain frozen, and all three checkpoints, optimizer states, references and losses are retained.

After the complete fixed fit, each checkpoint faces its own actual-GPU three-case numerical check (maximum score error≤0.01), then passing-only protected80 calibration. Before fitting, root declared two prospective policy arms: primary normal-view FP≤1/70, and secondary≤3/70. Both retain pixel precision≥0.65, the same full raw grids/growth options and weakest-positive ranking, with an independent eligible seed shortlist per arm. They share identical actual score caches: one neural forward per view. All three checkpoint outcomes and both global winners must freeze before either original audit; both results remain reportable. The secondary targets fewer alarms than the historical 5/69 baseline and does not claim fewer than C200's one-alarm record.

Root alone evaluates at most the two frozen arm winners on the exposed comparison after each fitted operating point passes a new complete-profile speed gate. Policy copies have byte-identical ONNX weights and distinct explicit calibrated sidecars. Final paired runtime must report all observations, drops, late frames and errors; no inherited FPS or production qualification is claimed.

## Preparation status

Nine geometry/loss/configuration/search fixtures passed, including complete foreground coverage across tile boundaries, exact zero logit-gradient/loss-invariance outside observed pixels, exact primary equivalence with the original pure search, and independent secondary-budget behavior. They use no model constructor, weights or neural forward. The first test collection found an eager legacy calibration import requiring `joblib` in the export environment; calibration-only imports were made lazy before the passing run. No training or inference occurred in that failure.

Independent source acquisition and root/peer review inspected all three fixed MTD raw/GT sheets. Native masks have antialiased gray boundaries, so the explicit TRAIN conversion uses raw mask>0, with original bytes and histograms preserved. This is our convention, not a claimed author binary threshold. Four MT_Uneven category images with completely empty masks remain quarantined, not normal examples. The curated manifest SHA256 is `53cb4a0eac6ed52e26eb713933fcf9739c2111b576c666c6ad806a34a3280158`; the cross-source screen reports no exact matches with protected source metadata and no close TRAIN-only pHash flags, without asserting physical independence.

The first full preparation stopped on an overstrict binary-code assertion: one protected SDD1 mask contains two gray85 pixels alongside 0/255. Existing C training and evaluation already interpret these original masks as mask>0. A TRAIN-only audit confirms SDD1 code counts 0:150,207,211, 85:2, 255:105,287; SDD2FIT codes are0/255. Root approved removing the unverified binary-only guard while retaining the exact existing threshold and all raw uint8 masks/histograms. The failed preparation source/receipt and first passing fixture remain preserved; no proposal, optimizer, model constructor or neural inference had occurred. The updated fixture uses gray85 explicitly.

The complete data pass succeeded. All 3,444 accepted sources occur in 3,678 mandatory tiles, within the fixed4,096 budget. The remaining418 slots are source-balanced positive repeats. The 900-update fine ledger is also fixed before training.

| Source | Mandatory tiles | Projected GT foreground pixels covered |
|---|---:|---:|
| SDD1 TRAIN | 251 | 122,325 |
| SDD2 inner FIT | 1,974 | 3,808,299 |
| MTD accepted TRAIN | 1,453 | 10,861,672 |
| **Total** | **3,678** | **14,792,296** |

Coverage applies to projected TRAIN foreground support, not a camera-resolution localization metric or complete normal-image pixel coverage. Full raw mask codes/histograms and source/native geometries remain in the source manifest. The fitting source count is 616 positive and2,828 normal images, with unknown physical-item independence in the additional domains.

The exact pre-fit proposal is `output/optimization/factory-round4-supervised/proposed-protocol-before-fit.json`, SHA256 `89adfa6e24de1c4dd3b2fbe13c2644b3948dc521cec520e840ceadf9ab7d1a2e`. Root approved this exact proposal and released CPU1 fitting. Protocol recording passed, executing sources were archived before optimization, and the fixed4,096+900 update fit started with OMP/MKL/OpenBLAS/Torch threads set to1. It may overlap root's GPU video diagnostics/media and ordinary host activity. These TRAIN walltimes are not inference-speed measurements.

The first50 transfer updates completed in58.283 seconds with finite batch loss0.002320; update100 completed at108.153 seconds with finite batch loss0.028382. The second50-update block took49.870 seconds. Batch type varies in the fixed shuffled ledger, so these are not averaged validation losses or defect-accuracy evidence. The observed pace suggests a substantial local fit; no schedule is shortened in response. Actual GPU numerics/calibration remain held until all updates finish and root releases the device.

Transfer update1,900 completed at1,021.491 seconds. Every recorded loss remained finite and per-update frozen-buffer/state checks continued to pass. The latest300 updates took94.865 seconds; the shuffled source/patch composition changes over the ledger, so the earlier startup pace is not a reliable fixed-duration estimate. No inference-capacity or accuracy claim follows from these TRAIN observations.

For the user's requested improvement, both historical comparators matter: old v10h has5/69 normal-view alarms; C200 already has1/69. Exceeding the best comparator on both requested accuracy dimensions requires11/11 meaningfully localized defect views and0/69 normal-view alarms, in addition to measured15+FPS. A secondary-arm result with two or three alarms could improve the older v10h record while worsening C200's alarm count; it would not meet the complete best-comparator target. Both prospective arms and their fixed selection rules remain unchanged.

The fixed4,096-update transfer stage completed at1,749.983 seconds with finite final batch loss0.015096. Its state and completion receipt were saved. The separate, declared fine-stage AdamW then started; fine50/900 completed at1,781.360 seconds (31.377 seconds after transfer), with finite batch loss0.020756. No checkpoint numerical or calibration feedback occurred between phases. All training walltime remains contended/local optimizer work, not an inference benchmark.

Observed workload note: the construction agent reported its five-case CPU1 gradient/assignment probe completed before its separate fixed900-step CPU1 fit began at2026-10-20T17:40:10.995098Z (session25674). That construction fit overlaps the remaining factory fine-stage optimizer work. Parent's private-video accuracy/media and ordinary host services may also have overlapped earlier transfer work. These concurrent jobs do not qualify any inference-speed result, and no walltime is subtracted.

The exact4,996 updates and all three exports completed successfully in2,356.700 seconds (39.278 minutes, observed contended optimizer/export/source-check walltime). All accepted TRAIN source hashes were unchanged; every updated state remained finite and all frozen normalization buffers stayed identical. Fine300/600/900 each retain PT state, optimizer state, ONNX, metadata, three fixed Torch reference inputs/maps and their hash-bound training receipt. Completion receipt SHA256:19d8422cb39d54ba72a99b22e944c5d7ef49d469ff5d96a898b26aadd6e880c0. No GPU numerical check, calibration or original-comparison scoring had run when training finished.

All three actual-GPU numerical checks passed the fixed0.01 maximum error guard. Errors for TRAIN/zero/255 were: fine300 [0.000265334,0.002619423,0.000419230]; fine600 [0.000596135,0.002410496,0.000186356]; fine900 [0.000212609,0.001312526,0.000187978]. Actual GPU.0/f16/one-stream/LATENCY/default Winograd configuration, finite bounded maps and nonconstant TRAIN output were verified.

The first calibration launch stopped before any CAL model initialization or forward because I used the trainer's compact JSON serializer for a historical source digest defined with default JSON whitespace. The same80 ordered paths, file/mask hashes and native geometries produce historical digest7b09d8cbf39129dad144fcad6ecbad0ade0bb19e39de6e8fbdfc634eb6ed6d07 and compact digestd653ae6f8ac22d1df253869b79567ef5da109c70c2d4b8324131876dfa6c746c. The guard preserved the failure. A separate source-pinned recovery wrapper restores only the historical serializer and adds correction lineage to new CAL/freeze reports; the original training/evaluator source, all weights, all three passing numerics, data, grids and policy guards remain unchanged. Two pure serializer/rebinding fixtures and independent static review passed before the corrected CAL release.

## Actual calibration results and frozen policies

All three new supervised graphs passed numerical validation, then each processed the exact same 80 calibration views once. The two policy arms reuse each checkpoint's actual score cache. These are reused development-calibration results, not independent generalization evidence.

| Fine checkpoint | Arm | High / minimum cells / growth | Meaningful localization | Normal-view alarms | Pixel precision | Pixel recall | Micro IoU | Worst positive IoU |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 300 | Primary | .995 / 4 / .7 | 10/10 | 0/70 | 66.262% | 86.335% | .599721 | .242832 |
| 300 | Secondary | .98 / 8 / .7 | 10/10 | 1/70 | 65.268% | 86.335% | .591569 | .242832 |
| 600 | Primary | .98 / 32 / .8 | 10/10 | 1/70 | 66.693% | 86.471% | .603916 | .256897 |
| 600 | Secondary | .98 / 2 / .8 | 10/10 | 2/70 | 66.094% | 86.756% | .600363 | .256897 |
| 900 | Primary | .98 / 32 / .8 | 10/10 | 1/70 | 68.815% | 85.194% | .614644 | .243310 |
| 900 | Secondary | .98 / 1 / .8 | 10/10 | 2/70 | 68.022% | 86.500% | .614936 | .243310 |

The predeclared rank selects fine600 in both arms because its weakest-positive IoU is higher. Fine300's zero-alarm calibration point is retained as a losing candidate; it cannot be substituted after looking at the original comparison. The secondary arm has its own raw-seed shortlist, so it need not dominate the primary arm even though it permits more alarms. Neither shortlist or ranking changed after scoring.

The global freeze is `output/optimization/factory-round4-supervised/winner-frozen-before-root-capacity.json`, SHA256 `822c795aa83350841998b8243a3d072a82192ee4f026c7d95553c8368ba40106`. Both winning ONNX files have the exact same learned bytes, SHA256 `aa93456bc2057c476d0ca616ce2d8aff41e2d70a75ec577b39459349607a6af4`; only the calibrated policy sidecars differ:

- Primary: `models/factory/supervised_round4_real_coverage-step600-primary.onnx`, metadata SHA256 `f968ce617f902a7e07c089088b9670aa22bc2d7cab1b0cfd2da505153c9ca999`.
- Secondary: `models/factory/supervised_round4_real_coverage-step600-secondary.onnx`, metadata SHA256 `094a97f6187a928a4b1ce7d798f42f5a97f73548db0b7839257181bad045e7d3`.

Each new CAL report and the global freeze references correction receipt `41651121ca828d121c43f3546c5438be0fe3179f3d59057a38a6388470108298`. The unchanged original fit protocol is still `89adfa6e24de1c4dd3b2fbe13c2644b3948dc521cec520e840ceadf9ab7d1a2e`.

The parent authorized construction accuracy jobs to share GPU resources near the end of this stage, while the construction CPU1 fit continued. All three factory score caches were already present when the construction agent announced its first shared GPU job; exact simultaneous-forward intervals were not instrumented. `accuracy-workload-observation.json` records this limitation. No accuracy-stage latency is an FPS qualification.

At this freeze, no round4 original-comparison inference, first64 capacity benchmark or sustained runtime benchmark had run. Root must complete those gates after all project fits, exports, other inference, media encoding and heavy I/O stop. The official SDD2 TEST and reserved466 remain outside this experiment, and no default deployment was changed. Calibration success alone does not establish the requested 11/11 localization, zero normal-view alarms or production-line readiness.

## Additional lower-alarm exploratory comparison

After calibration, and before any original-comparison inference or speed benchmark, root requested one additional comparison to address the user's fewer-alarms goal directly. This is explicitly POST-CAL exploratory, not an additional prospective confirmatory arm. It does not change either original arm, threshold search, training schedule or global freeze.

Only the three already selected PRIMARY points were considered. The fixed additional ordering maximizes meaningful view recall, then minimizes normal-view alarms, then uses the original rank and earlier checkpoint as tiebreaks. Eligibility requires the same meaningful recall as the frozen primary winner, pixel precision at least0.65 and strictly fewer calibration alarms. This selects the existing fine300 PRIMARY model and its unchanged .995 / area4 / growth.7 point:10/10 meaningful calibration views,0/70 normal-view alarms,66.262% pixel precision and86.335% pixel recall. No model, sidecar or point was rewritten.

The separate proposal SHA256 is `02d6260ed04337370e03115dc484092e2a2c852d514a8bf29d6a78fc3782577f`; root approved it, and the additional candidate freeze SHA256 is `1e7ae79cc097f7e0bd1bd5938eda690cc76524e64d9e621a097fdcb761263bc1`. Three pure ordering/precision/recall/strict-alarm-gain fixtures and independent static review passed. This candidate must face exactly the same first64 capacity gate, one exposed original audit, paired timing and sustained runtime checks. All three candidate outcomes remain reportable; an original result cannot be used to silently replace a failed candidate or turn this exploratory choice into prospective evidence.
