# Independent wall-data review

The exact230-photo source manifest SHA **`be5dec6e2022a065875f92b53a7841e244612dc32d3fa218fff1db2b71a33fa5`** passes independent source, publisher XML, seeded selection and bounded visual review. The receipt `datasets/construction/finishes-expansion/mbdd-wall-five-class-v1/independent-data-review.json`, SHA **`7ed8697f8e1fa910bf188279fe05e464b6dc4f79a7c58db5f841cc244f2df67f`**, approves inputs for a bounded TRAIN-only probe; it does not authorize the four-epoch fit or establish accuracy, speed, deployment or code acceptance.

All14,471 original XML members have publisher CRC/size checked. Exact pre-photo source/class selection is independently reproduced from the fixed seed and entire200-image block partition; all18 old diagnostic photos' blocks remain excluded. Each selected230 photo has original CRC/SHA/native geometry verified, and its manifest class IDs and continuous box coordinates match actual XML. Class order is crack/leakage/abscission/corrosion/bulge. Existing pHash≤6/exact cross-split quarantine is reproduced: TRAIN Hefei14274 remains excluded against CAL Hefei14423, leaving181TRAIN and48CAL photos with no backfill.

The first independent auditor applied native-bounds validation too broadly to every unused source XML and stopped. Its source/error console remain preserved. A complete separate inventory identifies116 out-of-bounds publisher objects, **all outside the selected230; zero selected boxes are invalid**. The corrected audit retains all full-inventory CRC/selection checks and requires strict finite positive native bounds for the actual selected data. No source label, coordinate, split, image or quarantine is edited or clamped. Future dataset expansion cannot assume unused source labels pass geometry checks.

| Class-positive photos after quarantine | TRAIN | CAL |
|---|---:|---:|
|crack|55|10|
|leakage|40|11|
|abscission|56|21|
|corrosion|40|10|
|bulge|39|10|

These image counts overlap because photos may contain multiple classes. Eight empty original XMLs are inventoried rather than treated as healthy negatives. Absent targets on other-defect images cannot establish healthy-building false-positive rates.

All11 predetermined raw-versus-complete-GT sheets are directly viewed: lexical first/middle/last for each of five classes in both partitions, plus every similarity flag. This covers30 source-class observations and29 unique photos including the flag pair; it is not visual certification of all230. Visible cracks, staining, detachment, rust-like discoloration and coating blister/bulge appearance are present, with boxes aligned to actual scenes. The flagged pillar pair is visibly the same view and remains quarantined. Repeated garage/pillar contexts occur elsewhere too: blocks and pHash cannot establish independent physical sites, and CAL can remain correlated with TRAIN.

Fine distant corrosion is sometimes subtle; staining does not demonstrate active leakage, abscission is broader than paint or concrete spalling, and a2D bulge label does not measure depth. Source labels support supervised appearance experiments, not engineering diagnosis. Original source attribution and bytes remain preserved; the data's CC BY4.0 does not remove inherited warm-start model/code/data restrictions.

## Fitter and evaluator static review

The five output channels are newly initialized under the declared semantic order, with the retained source backbone/box initialization distinguished from class relabeling. Frozen-stage BN is reapplied after framework train-mode transitions. The evaluator computes final per-class precision≥.8/recall≥.5 on actual joint serving predictions, with cached/actual decoder parity and isolated-versus-joint count equality required. Other-class absence is correctly reported without inventing healthy-wall FPR, and no camera heldout loader is used.

Before evaluator freeze/inference, provenance and failure-preservation hardenings are requested: verify the frozen fitted-model metadata SHA; bind the diagnostic point to its exact evaluation protocol and saved CAL/search/joint-count evidence; and fail closed on prior failed/partial calibration/diagnostic artifacts while saving actual outputs before a joint-count failure. These do not change the model, grid, selection rule or minimum guards. The source owner is implementing them; this initial static review does not claim those fixes or neural tests have already passed.

A subsequent code-only review confirms those changes: current fitted metadata is hash-verified, `checked_point()` recomputes matching/count arithmetic and guarded grid selection from actual saved CAL records, and started/completed/failed stage receipts protect partial work. Raw outputs and joint records are saved before a count-guard rejection. One final pre-diagnostic check is requested: the CAL execution receipt itself must say completed and bind the exact operating-point SHA before any18-photo forward, preventing a point left by partial completion from qualifying. No evaluator or fitted wall model has run as part of this review, and no performance test is claimed.

That final completed-CAL receipt/point-SHA check is subsequently verified in the diagnostic branch. A separately preserved initializer attempt exposes an actual architecture-scale issue: a generic YAML name lets the installed loader select a larger architecture instead of the intended nano. The new `train_mbdd_wall_detector_v2.py` fixes this through an explicit yolov8n filename and loader-scale assertion before the target constructor's ordinary stride-initialization forward. Before any real TRAIN forward it requires the exact five-versus-two classifier parameter-count delta, exact preserved source nonsemantic tensors and first10 backbone module states, then resets all five semantic channels. New training/model/evaluation paths are consistently v2, preserving the failed attempt.

Independent code-only review finds no remaining material blocker to the declared four-step batch4/CPU1 TRAIN probe or evaluator provenance/qualification controls. No probe or full-fit success is implied: compute release and the later explicit fixed-fit decision remain the coordinator's responsibility. No source or model mathematics is changed by this review.

## Fixed DACL appearance baseline on all18 old diagnostic photos

The separately frozen DACL512 baseline finishes actual local GPU inference on all18 source photos at strict>.50/minimum4-grid component, preserving protocol SHA `bd2d7052c1dd317ae9e8b526b69e861b4d272b65665eff29d9a68d2cebb8f0f2`. Its report SHA is **`bbfad5208abf563aab93f94b146019de813124766699bf2390e08daff4ddf540`**. A read-only independent class-aware one-to-one IoU≥.50 count audit also passes.

| Fixed declared proxy comparison | TP / FN / unmatched boxes | Precision | Recall | Class-absent photos flagged |
|---|---|---:|---:|---|
|Publisher leakage→DACL wetspot|2 /15 /8|20.0%|11.76%|6/14|
|Publisher abscission→DACL spalling|1 /15 /6|14.29%|6.25%|2/13|

These are broad visible-stain/material-loss appearance proxies, not verified moisture cause, paint-only loss or concrete-only spalling. Other three MBDD targets are **not run in this fixed two-output baseline**; that does not imply the full DACL model cannot detect cracks or corrosion. A new five-class candidate's aggregate cannot be directly compared with this two-class aggregate. Class-absent photos can contain other defects and do not establish healthy-wall FPR. Publisher boxes do not provide semantic pixel GT, and fragmented masks can produce unmatched whole-instance boxes.

An additive before-NN execution-order receipt, SHA **`a3dd597279498a557976ad87011525b94fa0a8e423aa9077fed1379813faa6d3`**, records the coordinator's explicit amended order: baseline now, then old1171 last4 CAL, then factory BN numeric work. The five-class exact-four-epoch fit plan is already frozen at `e6b7a6193f83ef157abb184db3193d850088c74efeb85685ff92e976e0979b24` before baseline outcomes. The scientific baseline protocol/model/source/threshold/proxy mapping remains unchanged, and no baseline result is used to change the candidate plan. The approved CPU1 fit overlaps this diagnostic, so no inference-capacity claim follows.

## TRAIN-only export diagnosis review after failed four epochs

The fixed four-epoch candidate produces no retained detections on48 CAL or18 old diagnostic photos at its frozen failed fallback points. Lowering a cutoff after those outcomes is not an approved rescue. A bounded TRAIN-only diagnosis instead freezes the first lexical TRAIN positive per each of five classes, deduplicates repeated sources, and compares fused Torch, ONNX Runtime CPU and actual serving GPU raw outputs on the exact same full-frame serving letterbox RGB/255 tensor. Static review of `scripts/diagnose_mbdd_wall_training_export.py` finds no material input, GT or coordinate blocker: prepared images preserve native source geometry, all five class-score channels and four decoded coordinate channels are checked, score error must be≤.01 and coordinate error≤1 input pixel, and publisher GT enters only post-forward raw-anchor IoU/score summaries. Those raw anchors are diagnostic evidence, not served detections or accuracy credit. This review itself executes no NN or test.

The completed five-source report subsequently passes all three-engine numeric checks. An independent JSON-only read finds worst class-score difference≈3.58e-7 and decoded-coordinate difference≈.003815 input pixels, well inside the fixed.01/1-pixel limits. Low class scores are already present in Torch; many GT boxes have high-IoU raw anchors with weak correct-class scores, while other GT boxes lack an IoU≥.5 anchor. This bounded TRAIN evidence does not reveal an export/serving numerical discrepancy and motivates investigating learning/representation. It is not proof of one training cause, broader engine equivalence, accuracy or a reason to lower the frozen operating thresholds.

## Separately proposed retained-head continuation: static review

The proposed v3 experiment keeps the exact entire five-class last4 checkpoint, including every semantic output, and changes training duration, backbone adaptation and learning rate together: **12 additional epochs (total16), fresh AdamW at.0005, freeze4, exact last12 checkpoint**. It uses the same181 TRAIN/48 CAL photos,640 input, batch/nominalbatch4, CPU1 and matched existing augmentation. No original head is reset, no best-epoch checkpoint substitutes for the last, and no operating-grid/minimum guard is relaxed. This combined intervention cannot identify one cause if it improves.

Independent static review of `train_mbdd_wall_detector_v3.py`, `evaluate_mbdd_wall_detector_v3.py` and `proposed-fit-plan-revision2.json` finds no remaining material blocker. Requested pre-fit hardening is verified: exact source metadata SHA joins PT/ONNX/data/diagnosis/evaluator pins, actual retained full-state/head/first4 tensor digests are recorded before optimizer construction, and every final-checkpoint tensor must be finite before export. The installed framework explicitly re-enables prior frozen floating parameters outside the new freeze list; modules4–9 therefore can adapt. Frozen first4 modules are returned to eval after train-mode transitions and their parameter/BN state must equal initialization in the actual final checkpoint.

The versioned evaluator reuses the immutable executed base evaluator with a separately pinned target/wrapper, retaining the exact threshold grid, P≥.80/R≥.50 class guards, actual serving parity, partial-failure preservation and point-before18 ordering. The18 photos are already exposed development diagnostics; they do not select weights, schedule or thresholds and do not provide independent accuracy. Existing four-epoch failure artifacts stay intact. This is code-only review, not a fit authorization or an empirical result; no additional NN or test is run by the reviewer.
