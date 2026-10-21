# Round4 supervised data-coverage review

Scope: independent static review of `scripts/factory_round4_supervised.py`, `scripts/evaluate_factory_round4.py` and their hardware-free fixtures before protocol/optimizer execution. No model constructor, weights, inference, CAL pixels or original-comparison pixels were used by this review.

The proposed experiment materially changes genuine supervised source coverage while retaining the C200 phase/context serving graph. The exact accepted source inventory must be 239 SDD1 TRAIN +1,865 SDD2 inner FIT +1,340 independently curated MTD TRAIN =3,444. SDD2 reserved466 and official TEST remain excluded. Four original MTD positive-category empty masks are explicit quarantines, not normal examples. Original grayscale MTD labels require the declared TRAIN `mask>0` conversion; original files remain unchanged.

The reviewed crop implementation retains aspect for new domains, uses full-width 320-row positive tiles, and proves all **projected** foreground is represented exactly once in mandatory entries. This is projected occupancy coverage, not a claim that resized images preserve every native visual detail. Normals get one actual observed crop/source; complete normal-pixel coverage is not claimed. The mandatory ledger must fail and report its count if it exceeds the fixed transfer budget, rather than dropping sources. Positive examples retain their actual masks in all roles.

Padding is input-only and excluded from positive/negative BCE and Tversky terms. Native full-width 640-row target fine crops use an actual positive pixel with ±192-row jitter, keeping that selected point within the ±320-row crop radius. Hard and unbiased crops retain all in-view GT even if they contain actual damage; their names do not turn them into healthy labels. Matched horizontal/vertical flips preserve image/target/validity correspondence. Frozen normalization buffers and strict complete C200 state loading protect the known serving graph.

The transfer ledger's observed imbalance remains deliberate: every source receives mandatory exposure, source-balanced positive repeats fill only remaining budget, and 900 target-focused updates use 20% auxiliary rehearsal. This is a hypothesis for better target localization, not an accuracy guarantee. All fixed updates finish before any numeric/CAL feedback; all three checkpoints need terminal outcomes before each global arm freeze.

Two fail-closed improvements were requested before protocol freeze:

1. Recheck recorded C200 ONNX and parent-freeze SHA as well as the PT/metadata/hard-location files at actual execution.
2. Before CAL inference, bind each numeric report to the actual current checkpoint, its initial sidecar, three-input reference and training receipt. The original draft checked these in the final freezer but should reject stale reports before opening CAL inference too.

Both guards were subsequently applied by the owner. One material constructor bug was caught before any NN: the draft called `PackedPixelRefiner(None)` although the inherited context constructor unconditionally loaded its `pretrained` argument. A separately pinned original ImageNet constructor asset followed by strict complete C200 state loading, or a locally equivalent constructor, is required before execution. Existing archived shared model source must remain unchanged.

## Static closure

The owner corrected the constructor without changing archived shared code: use a pinned original ImageNet constructor file, then strict full C200 state loading before optimizer creation. Its SHA is rechecked in the execution plan. The curated MTD manifest and independent cross-source screen are pinned, and the exact source count is 3,444. The pre-CAL numeric lineage and parent ONNX/freeze guards are present.

The final evaluator computes one shared raw-map grid, then independently constructs the budget1 and budget3 seed/growth shortlists. It validates each complete ordered search and recomputes the eligible rank winner. Calibrated policy arms copy identical learned ONNX bytes to distinct paths/sidecars, keeping pending baseline sidecars intact. Every checkpoint terminal outcome and both arm winners are frozen before any original comparison. This preserves the primary guard and makes the secondary tradeoff explicit.

The owner's nine hardware-free geometry/loss/numeric/dual-arm fixtures passed, including exact comparison of the primary search with the prior pure selector on identical synthetic arrays and zero gradient from ignored padding. Those arrays test code contracts; they are not inspection evidence or synthetic defect training data. This independent reviewer performed source inspection only, with no additional NN/test execution.

**Static review PASS for data-only complete-ledger preparation.** The actual mandatory tile count remains an executable ≤4,096 gate. An eligible ledger and root-reviewed exact protocol are still required before any fit. No new accuracy, fitted speed or production-readiness claim is made here.

## First complete-ledger attempt: legacy GT code guard

The first complete preparation stopped before writing the full protocol/ledger and before NN: original TRAIN `kos20/Part4_label.bmp` contains native single-channel uint8 values 0/85/255, rather than the draft's assumed 0/255-only codes. Independent inspection confirmed exactly two pixels at85, 3,690 at255 and629,808 at0 (1267×500). This is not a BGR conversion effect. No CAL/comparison mask was opened.

Existing C-family training and the established localization audit already define annotation support as `mask>0`. A narrowly corrected grayscale-code guard therefore preserves that definition; changing the threshold would change existing supervision and is unnecessary. The owner is inventorying TRAIN-only code schemas, preserving this initial preparation failure, then updating the corresponding pure fixture/protocol before a new preparation. This is a source-schema correction before fitting, not an outcome-driven GT revision.

## Exact prepared proposal closure

After correcting that pre-fit code guard and refreshing the nine hardware-free fixtures, the data-only preparation completed. Exact proposal SHA: `89adfa6e24de1c4dd3b2fbe13c2644b3948dc521cec520e840ceadf9ab7d1a2e`.

The independent metadata audit recomputed the stored source/coverage/step ledgers, without reopening images or masks:

| Quantity | Verified actual value |
|---|---:|
| Unique TRAIN sources |3,444|
| Positive / normal sources |616 /2,828|
| Mandatory coverage tiles |3,678|
| Mandatory SDD1 /SDD2 /MTD tiles |251 /1,974 /1,453|
| Projected foreground pixels covered exactly once |14,792,296|
| Positive source-balanced repeats |418 (140 SDD1,139 SDD2,139 MTD)|
| Total fixed transfer entries |4,096|
| Target-positive /hard /unbiased fine entries |360 /180 /180|
| Fine auxiliary entries |180 (90 SDD2,90 MTD), every fifth step|
| Total fixed fine entries |900|

All sources appear in mandatory entries. Every positive-source transfer entry and target-positive fine crop contains actual projected foreground. Every source's sum of mandatory foreground occupancy equals its recorded projected GT count. All source/coverage/transfer/fine JSON hashes match the proposal. Both prospective policy budgets remain declared before fitting. The audit receipt is `output/optimization/factory-round4-supervised/independent-TRAIN-ledger-metadata-audit-v1.json`.

**Exact source/data/proposal static and metadata closure PASS.** Root still owns the fit compute release and all later comparison/speed decisions. No new learned accuracy or inference FPS exists at this preparation stage.

## Additive pre-CAL identity recovery review

After all three numeric reports passed, the first CAL command stopped before model initialization, forward calls, attempt markers or score caches: the evaluator imported the trainer's compact JSON serializer, while the protected historical CAL80 hash uses default JSON spaces. The original fit protocol, evaluator, weights and numeric reports remain intact.

The separate `scripts/evaluate_factory_round4_cal_identity_v2.py` was independently reviewed before recovery. Its only process-local method rebindings are the exact historical CAL serializer and a metadata writer adding the recovery receipt/wrapper identity. The two affected call sites are CAL source validation and final global-freeze source validation. The protected rows, pixels, GT, grids, source sampling, rank, arm budgets, thresholds, model math and execution recipe are unchanged.

Final static review passed after adding exact ordered-row/reference/digest/count checks, all three preserved numeric report SHA/PASS checks, verified failed evaluator/log archives, and a durable executed wrapper/helper/test source archive. The recovery receipt is `CAL-identity-v2-correction-before-CAL.json` (SHA `41651121ca828d121c43f3546c5438be0fe3179f3d59057a38a6388470108298`). Its record stage verifies the existing exposed CAL images/masks against historical source hashes and geometry; that is source-integrity decoding, not NN or model selection. This reviewer opened only source code and existing JSON metadata, never CAL pixels or models.

The root benchmark helper separately corrected its historical CAL serializer and binds the additive recovery lineage before any benchmark plan. Six pure fixtures and a metadata-only proof confirm the unchanged 80-source/10-positive protected identity `7b09d8cbf39129dad144fcad6ecbad0ade0bb19e39de6e8fbdfc634eb6ed6d07`. No benchmark, original scoring or NN was executed by this review.
