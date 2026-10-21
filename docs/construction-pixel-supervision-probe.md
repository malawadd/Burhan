# Crack-only pixel supervision: prepared proposal and bounded TRAIN probe

This is a separate preparation/probe experiment. The current construction YOLO fit and its protected data/calibration remain unchanged. **The300-step fit and all new GPU inference are unstarted and require the coordinator's later decision.** No spall head is added.

## Protected inputs and protocol

`output/optimization/construction-pixel-supervision-v1/prepared-train-manifest.json` SHA **`45bde41e6d8a11762e29396895009503c111a2d0ac70e30b24633fd99c820219`** binds141 unique raw TRAIN source pairs and prepared tensors:

- All41 accepted LCSD low-light pairs: explicit original `mask==2` foreground,0background,255ignore (none present in accepted41). Originally author VAL/TEST, already repurposed solely as OUR TRAIN transfer; no LCSD accuracy claim.
- All80 acquired DACL official TRAIN photos, using ORIGINAL full frames and complete original Crack/ACrack filled polygons rather than previous derived crops or ev-CIVIL boxes.40 are crack-positive and40 are crack-negative within the published crack annotation scope.
- The20 original DACL TRAIN target-negative photos, excluding all CAL/heldout/diagnostic names. Each saved polygon list is checked for exact equality against pinned full publisher metadata before treating absence of Crack/ACrack as negative. Spall or other damage can be present; these are not healthy structures.

All raw source hashes are checked before/after preparation. Exact duplicate raw bytes are rejected. Native image/mask geometry must agree; crack polygons with empty raster occupancy or positive targets erased by384 projection fail instead of becoming negative labels. Every actual positive survives this preparation. Original bytes and dataset folders remain unchanged; only separate output NPZ inputs are written.

Shared full-native `low_light_gain` runs **once**, then serving-matched384square INTER_LINEAR image resize. Masks/validity use INTER_NEAREST. Trainer reads already corrected uint8 BGR arrays directly, converts RGB float0..255 exactly like the author export, and never reapplies gain, /255 or ImageNet normalization. No GT-dependent gain/crop, box-filled semantic target or generated damage is used.

The final protocol, recorded before any probe forward, is `output/optimization/construction-pixel-supervision-v1/protocol-before-probe-and-fit.json`, SHA **`1f893e2bdcf025dbb5513a680a72b052729ea66405880900549f76afd24e32e4`**. It binds actual prepared/current preparation+gain implementations, original TUT checkpoint/model/config, source-only prior CAL identity receipt and executing helper sources. The initial dry protocol draft is retained separately; requested extra guards were incorporated before NN execution.

## Exact proposed continuation, not launched

Seed20261028; fixed300steps/save100/200/300, batch1/CPU1. Strictly reload the source TUT tensor checkpoint. Freeze the complete MobileViT encoder throughout; train the existing bottleneck/adapters/ECA/upsampling/CABlock/Fuse/final parameters. Entire model stays in eval mode with autograd enabled, freezing BN running state and disabling stochastic dropout/DropPath; non-encoder BN affine parameters may learn. This is explicitly a small-data continuation, not the author's full training recipe.

AdamW lr1e-4/weightdecay1e-4, global gradient clip5. Main raw-logit loss is valid-pixel BCEWithLogits(pos_weight2), plus.5softDice(smooth1) on GT-positive images only. Ignore pixels affect neither term; zero-valid targets fail. The author trainer computes auxiliary losses for logging but actually uses `total_loss=output_loss`; our main-only optimization preserves that verified output assumption while deliberately changing the loss formula from author Tversky. Roles cycle LCSD-positive/DACL-positive/DACL-crack-negative; source sampling is uniform within role with the fixed seed. No geometric/photometric augmentation beyond the one shared serving gain is proposed.

Future qualification must run **all three** checkpoints on unchanged real construction122 CAL identities, with current strict native connected-component decoder, fixed threshold grid and publisher-box IoU.5. Guard crack precision≥.8 and negative-frame false flags≤.1; reject if no point qualifies. Fixed per-checkpoint rank is recall/precision/higher cutoff; global rank is recall/precision/lower negative-frame flag rate/earlier checkpoint. No new original-photo test selection or checkpoint feedback. This all-three evaluator/freezer is currently **declared, not implemented by the probe**, and must be completed before any full fit.

## Authorized probe scope and checks

Only batch1 and2, CPU1, **three actual gradient updates per trial**, reset original checkpoint/all seeds each time; discard probe models/optimizers and save no weights. Source Export sigmoid is compared to sigmoid of the selected raw-logit path on the first real TRAIN batch. Require finite logits/loss/gradients/nonzero gradient norm and finite ALL model tensors after each optimizer update, no encoder gradients, unchanged full encoder state SHA and every BN buffer. Own-process RSS is sampled50ms with stage guards against6GiB. Timings include checks and represent contended TRAIN optimizer capacity only, never inferenceFPS.

Actual background workload is the ongoing construction CPU4 fit plus ordinary host services; curvature fitting/calibration has finished. No GPU is requested by this probe. Only complete forward/backward/optimizer steps2/3 count toward its small TRAIN timing sample; step1 includes an extra source parity forward and is excluded from throughput estimation.

## Reserved102 well-lit LCSD pairs: metadata only

The pinned [author-linked LCSD distribution](https://huggingface.co/datasets/Chrisathy/LCSD/tree/254e9a987d1ee9d925472a42750c6f5f58112d9f) metadata and `train.txt` list102 unique well-lit image/mask pairs, all present in that revision's file inventory. Their IDs do not overlap41 dark-pair IDs. **Zero102 image/mask pixels were downloaded or opened.** Metadata receipts are under `output/optimization/construction-pixel-supervision-v1/lcsd-102-reservation-metadata/`.

The proposed pre-opening protocol SHA is **`9dbd73d736270fb5732ad4025986fbf1565812b1196ead41b2a926c9fd5289ee`**. Reserve the102 as OUR pixel-test views, although originally author TRAIN; freeze model/checkpoint/policy from existing CAL before any future pixel opening. Author loader says class1normal-light crack/class2low-light crack. Future masks must be reconciled as unconverted2D codes0/1/255 with class1 present; unexpected encoding is quarantined without guessing inversion/nonzero labels.

Before any NN scoring, screen exact raw/native-pixel duplicates and fixed near-duplicate signatures against all141 TRAIN sources, especially41 potentially same-site/day-night images; inspect flagged pairs without prediction outcomes. Exclude confirmed shared-view groups and quarantine ambiguous groups from the primary report, reporting every exclusion. Source IDs alone do not establish physical-scene independence. No new fit/cal cache can consume102. Report original-resolution valid-pixel precision/recall/IoU/Dice and per-view coverage for frozen baseline/candidate; do not invent pixel video annotations, healthy-image FPR or an author benchmark result. TUT pretraining overlap and campus-scene overlap remain unresolved.

## Preserved implementation checks and limitations

Preparation's first launch failed before output creation because some unrelated metadata records omit `name`; selection now indexes named records and still requires every exact selected TRAIN identity. Its traceback is retained. Three hardware-free mask/raster/gain tests pass in the app environment; the Torch loss test skips there because Torch is absent, and passes separately in the training environment (finite loss, exactly zero ignored-pixel gradient). The training environment has no pytest; its failed pytest invocation is retained in the tool history rather than represented as a passing suite. A fixture initially selected a point outside its own triangle; correcting that coordinate made the intended polygon-class check pass, with no production rasterization change.

Whole-model LiteCrackSeg licensing is undeclared; DACL is CC BY-NC4.0; LCSD author-linked card declares Apache-2.0 with source-lineage caveat. All future trials remain experimental. Parameter count, a few optimizer updates and true still-image pixel GT do not establish real-time video accuracy or code-compliance inspection.

## Completed bounded probe

Receipt `train-gradient-memory-probe.json` SHA **`416a65b137830f73ae058f06694dee0aa4ad2b26071ce05d23533b87f15206e8`** reports all checks passed. The actual six optimizer updates have finite logits/loss/gradients and full model state,326 nonzero-gradient tensors per update, zero encoder gradients, unchanged full encoder state and BN buffers. Source Export sigmoid versus the raw-logit path has maximum error **0** in both reset trials. There are1,987,986 trainable and951,024 frozen parameters. Peak process RSS is **1,235,030,016 bytes (1.15 GiB)** against6GiB. No weights were saved and no GPU inference ran.

| Batch | Complete checked step2/3 seconds | Contended TRAIN images/s |
|---|---|---:|
|1|3.0436 /2.8479|.33947|
|2|4.8103 /4.9065|.41166|

Only two complete timing samples exist per trial. They include guards and overlap constructionCPU4 plus ordinary host services; they do not predict inferenceFPS or guarantee the300-step walltime. Step1 is excluded because its extra source-parity forward interrupts the timing boundary. The protocol's batch1 remains fixed; the faster batch2 observation does not change it.

## Concrete fixed-fit/CAL tooling, awaiting timing release

The coordinator authorized the exact300-step fit **only after the parent factory timing window releases**, with no extension. New separate sources are `scripts/train_construction_pixel_supervision.py` and `scripts/evaluate_construction_pixel_supervision.py`. They import the immutable prepared inputs/probe loss rather than changing the historical TUT trainer/export or1f893 proposal. No launch is implicit in module import.

An additive `training-execution-receipt-before-optimizer.json` must pin actual new source hashes, immutable proposal/probe/preparation identities and **canonical CAL GT** before the first optimizer. Canonical CAL identity includes ordered source/path/frame/negative flags and full expected box coordinates/classes/annotation IDs, as well as original source JSON hashes:122 frames,112 annotated/10 target-negative,125 crack/38 spall boxes,163 unique annotation IDs. This closes an image-hash-only gap before any fit. Calibration and freezer recompute this identity; all three report digests must match.

The fixed trainer preserves source/schedule/layer/loss semantics, refuses restart/extension, guards frozen state and finite full model tensors after each update, and exports100/200/300 immutable tensor/ONNX/provenance artifacts. Each also exports a fixed TRAIN and zero-input Torch sigmoid reference. Actual OpenVINO GPU parity must pass against those references **before CAL**. All122 cached selected mask/box/score signatures must match actual `infer()`. All planned threshold-grid entries are required exactly once, and global freeze requires all three completed exports/CAL reports and strict precision/negative guards. If none qualify it freezes rejection, not an invented passing point. Failed/partial parity/CAL artifacts cannot be silently overwritten.

The following commands are prepared; execute only in the authorized order and coordinated compute slots. The full-fit flag is deliberately explicit. The actual workload note must describe conditions at launch.

```powershell
.venv/Scripts/python.exe -X utf8 -m pytest tests/test_construction_pixel_supervision.py tests/test_construction_pixel_selection.py -q

.venv/Scripts/python.exe -X utf8 scripts/train_construction_pixel_supervision.py --record-execution

$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'
.venv-roads-export/Scripts/python.exe -X utf8 scripts/train_construction_pixel_supervision.py --fit --contention-note 'REPLACE WITH ACTUAL COORDINATED LAUNCH WORKLOAD'

.venv/Scripts/python.exe -X utf8 scripts/evaluate_construction_pixel_supervision.py --calibrate --step 100 --workload-note 'REPLACE WITH ACTUAL COORDINATED GPU CAL WORKLOAD'
.venv/Scripts/python.exe -X utf8 scripts/evaluate_construction_pixel_supervision.py --calibrate --step 200 --workload-note 'REPLACE WITH ACTUAL COORDINATED GPU CAL WORKLOAD'
.venv/Scripts/python.exe -X utf8 scripts/evaluate_construction_pixel_supervision.py --calibrate --step 300 --workload-note 'REPLACE WITH ACTUAL COORDINATED GPU CAL WORKLOAD'
.venv/Scripts/python.exe -X utf8 scripts/evaluate_construction_pixel_supervision.py --freeze
```

These additions supply the all-three enforcement that the historical probe proposal correctly described as not implemented *by the probe*. The original proposal bytes remain unchanged. No reserved102 pixel acquisition is authorized by these commands, and no inference speed qualification follows automatically from calibration.

After the timing hold released CPU-only checks,9 hardware-free source/loss-selection/freezer tests passed with1Torch-dependent skip in the app environment. The additive execution receipt is now recorded with SHA **`ea63aed306e6421bd75cfc1b77c92b756339576bcc071768dcb7edda301989ee`**. Canonical actual CAL GT digest is **`0964897aba2ff9282d743dc5c93fa0a206bc0d7d0997e3268507cb863291d6ac`**. No optimizer has launched; launch waits explicit PIXELFITRELEASE after any new C-density exclusive preflight.

An additive102-photo diagnostic proposal v2, SHA **`051f9f7cdbda2198776f019f267127061177b27d192238007ef6105ce2977034`**, preserves the original9dbd reservation and supersedes only its model-selection prerequisite. It fixes **original TUT versus FINAL step300 at strict probability>.50 now, before fit/pixels**, with identical gain/native projection and all clean reserved sources after leakage/encoding review. Actual final300 artifact identities must be bound before pixel download. All three CAL reports/global freeze must finish even if rejected; this diagnostic can then describe pixel quality without promoting a failed model or relaxing video box/precision/negative guards. Primary raw native pixel P/R/IoU/Dice and any filtered serving-mask diagnostics are explicitly separate. The102 pixel files remain unopened; the coordinator must review v2 before acquisition.

## First fit interrupted at checkpoint100: provenance export error

The exact authorized fit starts after PIXELFITRELEASE and completes100 finite optimizer steps with unchanged encoder/BN guards. Step100 loss is.5385298, optimizer-loop walltime204.963s and RSS511,557,632 bytes. It then saves the step100 tensor checkpoint, ONNX and fixed TRAIN reference, but **fails before writing model provenance**: protocol `source_pins` keys are Windows backslash paths, while the export dictionary requests a forward-slash literal. The resulting `KeyError` is an implementation error in metadata lookup, not an accuracy or numerical-model result.

No CAL, GPU inference,102 acquisition, source mutation or model comparison follows this partial trial. Original artifacts/logs remain unchanged. Executed scripts, immutable protocol/execution receipt, started/progress receipts and error console are copied/hash-verified into `failed100-executed-source-snapshot/`; archive manifest SHA **`116f8292dcc70cb1853ccd76930f2b71f52493a4e395ce419f9c48b33cf9e414`**. Partial step100 tensor SHA is `a13b23b20faf8d10f0acae4c8e74bcddf9c0ba3f76753503738c62798571af08`; ONNX SHA is `e21aa7f5cdd263ece9f530d36eecd7dfac84744ef18f00dfae5fe1864becb909`. Neither is treated as a calibrated candidate.

AdamW optimizer state was not serialized, so a weights-only step100 resume would change the training recipe. A corrected, separately versioned restart must begin from the ORIGINAL source checkpoint/seed, preserve this failed trial, validate metadata lookup before the first optimizer, and still complete its own fixed300 steps. That recovery has been proposed to the coordinator; no restart is implied by this record. The conditional102 pre-opening guards remain unsatisfied and no102 pixels have been downloaded.

## Separate metadata-only recovery v2

The coordinator had explicitly authorized the fixed300 recipe and sent PIXELFITRELEASE. During recovery its agent then reported a model-capacity error, so the inability to reply is recorded rather than presented as new approval. Independent static review by the construction agent found only a canonical Windows/POSIX source-key lookup correction and separate output roots: original checkpoint, seed, fresh AdamW, batch1/CPU1,300 updates, losses, sampling and frozen state remain unchanged. The prior authorization therefore applies to this concrete implementation correction. The failed100 checkpoint is never resumed.

The separate actual trainer/evaluator are `scripts/train_construction_pixel_supervision_v2.py` and `scripts/evaluate_construction_pixel_supervision_v2.py`. Their additive execution receipt is `output/optimization/construction-pixel-supervision-v2/training-execution-receipt-before-optimizer.json`, SHA **`bed5a4c06f24fe5d9799845726203607396d02850527a5208adabb7048cf85df`**. It pins the path helper and actual sources, original immutable1f893 proposal, prepared141 inputs, canonical CAL GT and failed-trial lineage. Source lookup and hash validation now run before optimizer creation. Six hardware-free path fixtures pass, including the actual immutable Windows-key protocol and conflicting-alias rejection.

The v2 fit starts from ORIGINAL TUT and successfully exports its complete first100 tensor/ONNX/TRAIN-reference/provenance set. It then reaches180/300 with finite checks intact. Its first100 loss sequence agrees with the preserved v1 run. This progress establishes recovery of the export path, not model accuracy. The fixed300 fit remains in progress; no checkpoint CAL or reserved102 pixel acquisition has run. Actual CPU1 training/export walltime may overlap C-density accuracy diagnostics and construction video exports and has no inference-speed qualification credit.

Conditional photo diagnostic tooling is separate in `scripts/audit_construction_reserved_pixels.py`. The coordinator approved the fixed TUT versus FINAL300 v2 methodology before pixels. A path-only v3 recovery addendum, SHA **`3e70d9bf5c796fc0a38b5f9bb00852f5333c3c8cbf8958366a9a24fee3460e54`**, changes FINAL300 paths to this v2 run and retains every pre-opening/all-three-CAL/numeric/serving/leakage guard and the strict>.50 cutoff. It does not consume102 pixels or authorize selection/promotion. Four hardware-free diagnostic fixtures pass; actual source acquisition still waits complete training, all-three CAL/global freeze, parity and an actual final300 binding.

## Completed fixed300 v2 fit

The separate restart completes **exactly300 updates and all100/200/300 exports**, EXIT0. Completion receipt SHA is **`df361a81003c9bcf2afe7e8e3a7a479d00041f320f52a5c1dd8a83f0de969f94`**. Its588.153s TRAIN/export walltime includes declared concurrent accuracy/media work and ordinary host services; this is not an inference-speed measurement. Peak process RSS is595,972,096 bytes. Every full-state finite guard passes, and the entire encoder state and all BN buffers remain identical. No CAL, heldout or reserved102 pixels are opened by the fit.

The recovered step100 tensor and ONNX are **hash-identical** to the preserved failed v1 exports, supporting that the correction changes metadata handling rather than the numerical recipe. Final300 ONNX SHA is **`b7cad346d11cb89be6a76fff38ab26fa3e8fd9778981af47498e6f8a99c6d3d7`**, tensor SHA **`eca58025b80dc3096f71bea5362365f9486e6538681b685a5d90bbf61a975da2`**, and model provenance SHA **`95a54eb7e5b72c6f7e94aab33ce31201d9a6ba74b3b309a8f46693f763745238`**. All-checkpoint actual GPU parity/CAL and global freeze remain pending the explicit C-density GPU handoff; completed training alone supplies no accuracy or acceptance claim.

Before any102 binding or pixel opening, independent static review also hardens the diagnostic: the fixed pair has explicit original/final order, known-reference inventory SHA joins the pre-opening binding, both endpoints of any ambiguous within-reserved pair are quarantined, and partial reports explicitly name failed/unprocessed clean sources. This is conservative pre-outcome screening, not favorable-case selection. Five hardware-free diagnostic fixtures pass. All original102 files remain unopened.

## All-three actual GPU CAL: rejected

All100/200/300 checkpoints complete actual GPU numeric and122-view calibration checks, and the unchanged global freezer returns **accepted=false, winner=null**. No point in the protected grid meets precision≥.8 and negative-image false flags≤.1. No trained checkpoint is promoted. Every fitted TRAIN/zero numerical guard passes (maximum TRAIN errors9.51e−6,8.23e−6,8.40e−6 respectively), and every selected failed-diagnostic cached mask/box/mean-score signature matches actual serving on all122 views.

The following rows are the preregistered best-F1 **failed diagnostics**, not qualified operating points. Recall uses125 publisher crack boxes and IoU.5; thin geometry is62 of those annotations. The38 spall annotations remain unsupported/unscored by this single-crack model.

| Checkpoint | Failed diagnostic cut | TP / FN / unmatched predictions | Precision | Recall | Target-negative images flagged | Thin matches |
|---|---:|---|---:|---:|---|---|
|100|.10|14 /111 /560|2.44%|11.2%|10/10|2/62|
|200|.85|19 /106 /584|3.15%|15.2%|10/10|2/62|
|300|.75|25 /100 /631|3.81%|20.0%|9/10|4/62|

Actual model initialization takes48.79s for100 and51.87s for200; model descriptions retain measured load time for every checkpoint. These accuracy runs do not establish inferenceFPS. Construction's CPU1 normal-tile feature fit/evaluation overlaps200: filesystem observations span2026-10-20T13:29:33.269Z–13:30:11.304Z, not an exact profiler interval. It exits before300 finishes; ordinary services and lightweight metadata work remain possible. No other project GPU inference overlaps these three runs.

Actual final300 pre-opening fixed-pair binding SHA is **`979778c96b6bca3467fb0163651f254dfab192610f27448a25442fe0dbca9a32`**. It is recorded only after complete training/all-three global rejection/numeric and serving parity, and before ANY reserved102 pixels. The separately approved fixed TUT-versus-FINAL300 strict>.50 photo diagnostic and three full crack development videos retain that pair/cutoff despite CAL failure; they cannot replace the rejected qualification or select another checkpoint. Reserved102 acquisition now starts under those guards; encoding/leakage reconciliation and actual inference remain separate subsequent stages.

## Reserved102 acquisition and pre-NN reconciliation complete

All102 original image/mask pairs are acquired from the predeclared source revision with204 per-file download receipts. Source-manifest SHA is **`a3b78f1e2a47d401df262a8c3b095f53678072be6a89ade017a01c022d0900f3`**. The encoding/leakage freeze SHA is **`cea69b8b3f6692d46bb7fac6f6f01ed9f5e2920c4fa9b3142d976ea500cb076a`**:102 clean,0 quarantined, all102 accounted. Actual mask codes are0background and1crack, with15,821,351 background pixels and498,649 crack pixels; no255ignore pixels occur. Every source has positive class1 GT and matching native geometry.

The pre-NN screen compares against1,233 unique reference paths (141TRAIN plus1,092 known materialized references), including exact file/native pixels and fixed conservative near-view signatures, and also checks all within-reserved pairs. No declared duplicate/near/ambiguous flags occur. This screen is limited evidence: source IDs and these signatures cannot establish independent physical sites or absence from TUT pretraining. No neural output is used in these decisions. Fixed-pair actual photo inference now waits the coordinator's GPU release after the full video comparison.

## Fixed102 real-pixel diagnostic: final300 regresses recall and IoU

The approved originalTUT versus FINAL300 pair completes actual local GPU inference on **all102 clean photos**, strict probability>.50, identical gain once and native probability projection. No image/checkpoint/cutoff is changed after outcomes. Report SHA is **`fbfc23c475e59e240226664e7e3f61719c8faf89527eda8f4601daf83664c06c`** at `output/optimization/construction-pixel-supervision-v2/reserved102-fixed-pair-diagnostic/all-clean-reserved-fixed-pair-report.json`. No source is silently dropped, none has an empty prediction, and all102 original-resolution raw mask/probability files and four-panel RAW/GT/TUT/FINAL evidence are saved.

| Primary raw native pixel metric | Original TUT | Fixed FINAL300 |
|---|---:|---:|
|Micro precision|59.47%|65.25%|
|Micro recall|67.64%|43.35%|
|Micro IoU|46.30%|35.22%|
|Micro Dice|63.29%|52.09%|
|Pixel TP / FP / FN|337279 /229878 /161370|216172 /115148 /282477|
|Macro all-view IoU|45.75%|36.43%|

The area4/top100 serving-mask secondary result is almost identical (micro IoU46.29%→35.20%), so that postprocessing does not explain this recall loss. The fitted model suppresses false-positive pixels but misses substantially more true crack pixels; it is **not adopted**, and CAL rejection remains unchanged. There is no healthy-photo negative population here: all102 contain crack GT. This is OUR fixed-pair reserved-photo diagnostic using an authorTRAIN partition, not an independent-site or author test benchmark; TUT pretraining overlap remains unknown.

The explicitly predetermined first clean source, **LCSD_0095**, is visually checked at `000-LCSD_0095-all-four-panels.png`: the original TUT marks a broad continuous trace, while FINAL300's narrower upper trace misses lower crack segments. This is the first source by frozen order, not a selected success example. All other101 panels remain equally available. Real camera video localization, inferenceFPS and Saudi code acceptance are separate questions; this still-photo result cannot validate them.
