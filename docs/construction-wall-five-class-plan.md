# Separate real-wall five-class experiment plan

20 October 2026. **Preparation only: no wall TRAIN probe or fitting has run.** The parent must review data and release compute before execution. Existing construction weights, eight diagnostic videos, original122 camera CAL sources and heldout data remain untouched.

## Data identity and limits

- Primary release: [MBDD2025](https://zenodo.org/records/15622584), CC BY4.0.
- Original18 displayed photographs are diagnostic-only. Their entire numeric200-image capture-proxy blocks are excluded from the new source pool.
- Additional data root: `datasets/construction/finishes-expansion/mbdd-wall-five-class-v1`.
- Frozen pre-label/pixel protocol SHA256: `72385bbf468c2d81ffd27bea58a580bc31db18fa8f5fbe56a0a8fa9d6bbded10`.
- Original asset manifest SHA256: `be5dec6e2022a065875f92b53a7841e244612dc32d3fa218fff1db2b71a33fa5`.
- 230 real photographs,39,737,533 original image bytes;7,618,265 bytes for the single contiguous range containing all14,471 original XML annotations. Each acquired member passes original ZIP CRC/size and its own SHA256. Full2.53GB archive MD5 remains unverified.
- Fixed first40 TRAIN/10 CAL photos per publisher class by seeded metadata order, union deduplicated; no model-ranked selection or post-quarantine backfill. Whole numeric blocks are disjoint across TRAIN/CAL. One conservative cross-split perceptual-similarity endpoint is quarantined.
- Numeric blocks and perceptual screening do not prove physical-building/session independence. Within-split adjacent views remain correlated. Independent source/label/visual review is required before any prepared-input copying or numerical probe.

The complete source inventory has eight empty XMLs. These do not establish deliberately reviewed healthy-wall negatives. Current selected sources are positive imagery with other-class absences; healthy-wall false-alarm performance is unknown. DACL photos containing Wetspot are never used as moisture negatives.

Independent review approved the exact asset manifest for a TRAIN-only probe in `independent-data-review.json`, SHA256 `7ed8697f8e1fa910bf188279fe05e464b6dc4f79a7c58db5f841cc244f2df67f`. The post-quarantine inputs contain **181 TRAIN and48 CAL photographs**. All230 selected original photographs/XML members passed native dimensions, source CRC/SHA and bounding-box geometry checks. An independent whole-inventory audit also found **116 invalid publisher objects outside the selected230**. Those unused originals and the failed initial broad audit are retained; no boxes were clamped, no sources replaced and no assertion is made that the whole publisher release is geometrically valid. All11 fixed raw/GT sheets were inspected (29 unique photos), including the quarantined pair. Repeated garage/pillar contexts remain a correlation limit. This review does not authorize the four-epoch fit.

Source class order is **0 crack /1 leakage /2 abscission /3 corrosion /4 bulge**. The actual labels remain publisher names: leakage is a visible stain/wet-appearance proxy, abscission is generic surface detachment, and bulge is two-dimensional appearance. The model cannot measure moisture quantity, verify a hidden leak, identify paint versus plaster, measure bulge depth or issue code acceptance.

## Prepared source contract

`scripts/prepare_mbdd_wall_yolo.py` requires an independent data review receipt matching the exact asset-manifest SHA and explicitly approving a TRAIN probe. It copies only unquarantined reviewed originals byte-for-byte to a separate `prepared-yolo` directory. It retains **every** original class box in each image. Native publisher continuous XYXY becomes normalized YOLO XYWH, without a guessed one-based shift. Original XML and source photos are preserved.

The preparer requires at least20 positive TRAIN photos and five positive CAL photos per class after quarantine; it aborts rather than silently adding new images. Every copied image/label, class order, YAML, review receipt and preparer source is hashed in an additive input manifest. No derived source is called a video or new independent photograph.

## Exact bounded TRAIN-only probe

Proposed command, **after review and resource release only**:

```powershell
.venv\Scripts\python.exe -X utf8 scripts/prepare_mbdd_wall_yolo.py
.venv-roads-export\Scripts\python.exe -X utf8 scripts/train_mbdd_wall_detector_v2.py --stage probe
```

The source warmstart is the locally produced R4 head-best checkpoint SHA256 `5e48033ee0122d39a3a6301aa926dad6bd18b141e89468185a66d1871a5eef6c`. It has two original classes, spalling/crack. The new architecture has five classes. Compatible backbone, neck and box-regression tensors are transferred, but **all five terminal semantic output channels are explicitly reset**, with new normal(.01) weights and bias−4. No old joint/crack/spall channel is renamed humidity, corrosion or paint. Terminal five-channel shapes, reset-state hash, original checkpoint hash and transferred box-regression identity are checked before fitting.

The probe is **exactly four forward/backward/optimizer steps**, batch4, CPU1,640 input, channels-last, AdamW, fixed seed20261020. It samples16 TRAIN observations cyclically across all five positive classes. It uses full original images, standard letterboxing and all in-view labels. No CAL/diagnostic/heldout pixel is loaded. First10 backbone blocks have frozen parameters and evaluation-mode normalization. It records finite loss/gradients, nonzero new semantic-head gradients, RSS memory and actual elapsed time. Probe updates are discarded; no probe weights are saved.

This is a numerical/memory feasibility screen. Contended TRAIN images/s is not model inference FPS, video throughput, accuracy or promised future fit duration. If the source/library/shape contract fails, preserve the failed probe and review it before a separately versioned correction. Do not silently retry a modified numerical experiment.

The first attempted probe found an initializer bug before qualification: installed Ultralytics derives model scale from the YAML filename and overwrote the embedded `scale: n`. The original generic filename therefore selected the first scale, `l`. That wrong-architecture attempt was stopped; its original source, before-forward receipt, console and additive `invalid-scale-attempt-receipt.json` remain under `output/training/construction-wall-five-class-v1`. No probe weights were saved and no four-epoch fit followed it. The separate v2 source/output uses explicit `yolov8n-wall-five-class.yaml`, checks the actual framework loader resolves `n` before its standard constructor stride-initialization dummy forward, then verifies the exact expected parameter count, every nonsemantic source tensor/shape, all first10 backbone states and regression transfer before any real TRAIN forward. The reviewed data, class semantics and four-step CPU1 method remain unchanged. This correction is not an accuracy result.

The independently reviewed corrected v2 probe completed all four steps (EXIT0). Its receipt is `output/training/construction-wall-five-class-v2/train-probe.json`, SHA256 `a10a46b07d7fc6b55cbe634b42a3f5714d27d873a32c550d48ae93777339600f`. The target has exactly3,011,823 parameters (source3,011,238 plus585 additional semantic parameters),349 compatible tensors transferred and156 box-regression tensors verified. All nonsemantic source tensors and the first10 backbone states matched before real TRAIN forward. All losses and gradients were finite, every new semantic channel had nonzero gradients, frozen backbone/BN state was unchanged, and no probe weights were saved. Losses were204.07/199.88/103.88/239.12 over different TRAIN batches. Observed RSS peaked at485,605,376 bytes and the four-step optimizer phase lasted9.486s; these are local feasibility observations, not accuracy or serving-capacity results.

## Conditional fixed fitting

`--stage fit` requires a separate `approved-fit-plan.json`, created **after** the independent review and probe. The plan pins trainer, exact prepared input manifest, source PT and probe hashes, evaluator source/grid/mapping/guards, and explicit parent approval of the fixed four epochs. The parent-approved v2 plan was frozen before optimizer work, SHA256 `e6b7a6193f83ef157abb184db3193d850088c74efeb85685ff92e976e0979b24`; the exact four-epoch CPU1 fit completed (308.93s including export, with other accuracy work overlapping). The prepared-input manifest SHA256 is `b83cde9db9a14826e1155cca0af0a46d0acd748dfe220bda853e912b7497cb31`, and executed v2 trainer SHA256 is `1f4473583d5766971359d6899203cce1fdf500d2d0d15a5cdc4032dd0ad68ce5`.

Proposed fixed settings:640/batch4/nbs4, CPU1, channels-last, freeze10, AdamW lr.0015→.00015, warmup0, four epochs, no wall-clock cap or early-stop substitution. Full original RGB is used with modest recorded photometric/geometric training augmentation; no generated defects, mosaics, copy-paste or fake video. Frozen backbone BN is enforced after the framework's `model.train()` transition; installed trainer hook compatibility must pass source review first.

The exported candidate is **exact last.pt after epoch4**, predeclared before fitting. Intermediate generic-mAP best checkpoints are not silently substituted. An execution receipt is written before optimizer work; an existing receipt prevents silent restart. Exported class order and ONNX/source/input/license identities are recorded. No learning/threshold adjustment uses the existing18 diagnostic photos.

## Calibration and evaluation still to be frozen

A separate five-class evaluator must freeze its policy before post-fit CAL search. It must use the actual local serving decoder, full frames and all five class annotations, with selected operating-point parity. Report per-class TP/FN/unmatched boxes at one-to-one IoU.50, precision/recall and class-absent flags. Class-absent images with other known damage are not healthy-wall negatives. Do not invent pixel-mask IoU from boxes or imply paint-only/physical moisture truth.

Only after the candidate and operating point are frozen may all18 existing real-photo diagnostics be evaluated, including every miss and wrong-class alert. They are correlated development evidence, not blind site validation. Actual video evaluation additionally requires licensed raw moving-camera footage, original PTS and frame/event ground truth; none is supplied by the current MBDD ZIP. No automatic promotion or production/installation/SBC claim follows from passing a prototype calibration guard.

The separate implementation is `scripts/evaluate_mbdd_wall_detector.py`. Its fixed confidence grid is .10/.15/.20/.25/.35/.45/.55/.65/.75/.85. It maximizes each class's CAL recall subject to precision≥.80; an infeasible class retains the best-F1 point explicitly as failed diagnostic evidence. The joint decoder must reproduce cached outputs and per-class counts. The minimum prototype review guard is precision≥.80 and box recall≥.50 for every class; these are not release or safety criteria. Before any18-photo forward, the evaluator binds the frozen model, metadata, source, input manifest, operating-point protocol, retained actual CAL records, raw outputs and recomputed selection arithmetic. Started/failure receipts preserve partial attempts and prevent silent reruns. No old camera heldout loader is imported.

## Actual v2 result: failed, no usable new wall capability

The exact last4 model is `models/construction/mbdd-wall-five-class-v2/wall-five-class-last4-640.onnx`, SHA256 `3a275f83d529c3839626f7f301283928a046b2e397dcca7f0a77d1ccacbbdbcf`. All48 CAL inputs and selected joint serving parity completed. No grid point localized any source annotation. At the lowest frozen .10 threshold, only leakage produced predictions: four unmatched boxes and zero matches. The other four classes produced none. With F1 tied at zero, the predeclared fallback selected .85 for every class. That selected point abstains completely; its zero alerts are **not** successful false-alarm performance, and precision is undefined.

| Source class | CAL boxes missed / total | Fixed18 diagnostic boxes missed / total |
|---|---:|---:|
| Crack | 19/19 | 12/12 |
| Leakage appearance | 22/22 | 17/17 |
| Abscission / material detachment | 60/60 | 16/16 |
| Corrosion appearance | 23/23 | 15/15 |
| Bulge appearance | 10/10 | 9/9 |

All18 raw/publisher-GT/model comparison images and every per-class grid trial are preserved under `output/evaluation/construction-wall-five-class-v2`. Point SHA256 is `0a75a23f66eea564e5931186daeb178d70b23008325b07915138f0a4e12309cc`. Every minimum accuracy guard failed. This model must not be presented as an added working capability. No threshold rescue, unplanned fitting or camera-heldout inference followed the result.

## Bounded TRAIN diagnosis and conditional v3 continuation

The parent then authorized a separately frozen diagnosis of the first lexical TRAIN-positive photo per class, deduplicated: Hefei10172/10468/11287/111/12604. It used full native photos and identical serving RGB/255 letterboxing in fused Torch, ONNX Runtime CPU and actual OpenVINO GPU. GT was inspected only after inference; no GT-derived input crop or operational threshold change occurred. `scripts/diagnose_mbdd_wall_training_export.py` saved the before-forward protocol, all three raw outputs and original/publisher-GT images.

All numerical guards passed. Maximum coordinate discrepancy was0.003815 NN-input pixels (guard1pixel); maximum class-score discrepancy was3.58×10⁻⁷ (guard.01). Output shape, finite values and five-class order matched. Report SHA256 is `be0db1d57718f205af6ddd124d9130c8aa8e8ca480ee3f8c568fa9e8034874d9`. This provides no evidence of an export/backend mismatch causing the failure. Even seen TRAIN photos have weak semantic confidence: the selected leakage source's true leakage box had a best own-class score of only.00592 among anchors with IoU≥.50. Some small bulges had no raw anchor at IoU≥.50. These observations support underfitting and object-scale limitations; raw-anchor overlaps are not served detections or accuracy credits.

`scripts/train_mbdd_wall_detector_v3.py` defines one distinct continuation from the exact retained five-class last4 PT, with **12 additional epochs (total16), fresh AdamW lr.0005, freeze4 instead of10, unchanged640/batch=nbs4/CPU1/channels-last/data/grid/guards, exact final additionalepoch12**. It retains all learned semantic channels and resets no class. The prior failed model, point, outputs and18 diagnostics remain unchanged. `scripts/evaluate_mbdd_wall_detector_v3.py` binds a versioned wrapper to the immutable executed v2 evaluator, preserving matching, grid, parity and all18 diagnostics. The new source verifies PT/ONNX/metadata/data/diagnosis hashes before optimizer work, records full-state/head/frozen-layer digests, and rejects nonfinite final state. Independent static review passed revision2, SHA256 `b90636a740fb3f373a3aa2e9c34f214288c22cae5054633bdfb502bfa4b49e5a`. The parent approved that exact recipe and the before-optimizer plan was frozen at SHA256 `5c34652802e6333d288bef8fe4519b4c3e7666db4da89cf80f0f2087cc0c8531`.

## Actual v3 result: some localized findings, still failed

All 12 additional epochs completed; the actual final checkpoint, rather than a generic-mAP best checkpoint, was exported. Fitting and export took 1,319.32 seconds under ordinary host activity and other agents' CPU training. This is an observed contended training duration, not inference speed. Frozen first-four parameter and BN state remained unchanged; trainable-parameter L2 change was 14.1231. No extra epoch, threshold rescue or protected camera-heldout inference followed the result.

The resulting model is `models/construction/mbdd-wall-five-class-v3/wall-five-class-total16-last12-640.onnx`, SHA256 `e413db1d7f855140753f2343d96fd6620c62f3db7946d82c06b43ea7590da15c`. Its metadata SHA256 is `d5ed07d46b23cd85d8dd423970baf703827e03bef36886be410759bb742a7908`. All 48 fixed CAL inputs, the predeclared grid and actual joint-serving prediction/count parity completed. The wrapper-bound evaluation protocol is `6dd5db2cadc846dbb703fc049af55891ffdc0ce58c3c461ff05c56550bf9f31f`; selected-point SHA256 is `3d9a06062b95cdfc33d8e7026bce3584dd1f13761cf78604f000c4567bf30619`.

| Appearance class | Fixed cut | CAL TP / missed / unmatched | CAL precision / recall | All 18 diagnostic TP / missed / unmatched | Diagnostic precision / recall |
|---|---:|---:|---:|---:|---:|
| Crack | .85 | 0 / 19 / 0 | Undefined / 0% | 0 / 12 / 0 | Undefined / 0% |
| Wet/stained appearance | .15 | 3 / 19 / 12 | 20.00% / 13.64% | 4 / 13 / 6 | 40.00% / 23.53% |
| Surface material detachment | .20 | 8 / 52 / 1 | 88.89% / 13.33% | 0 / 16 / 1 | 0% / 0% |
| Corrosion appearance | .10 | 1 / 22 / 2 | 33.33% / 4.35% | 0 / 15 / 0 | Undefined / 0% |
| Bulge appearance | .15 | 1 / 9 / 0 | 100% / 10.00% | 2 / 7 / 2 | 50.00% / 22.22% |

Each count uses class-aware, one-to-one native-box IoU ≥ .50. No mask or physical-dimension accuracy is inferred. Every class still fails the minimum precision ≥80% AND recall ≥50% guard. The crack fallback abstains, after all frozen grid points produced zero crack matches; this is not successful negative detection. Some actual wet-stain and bulge annotations are localized, but the broad requested wall task remains unsolved. In particular, CAL detachment matches did not transfer to any of the 16 annotations in the original 18 diagnostic photos.

All 18 photos were run only after the point was frozen, with every miss and unmatched candidate retained. They remain previously exposed and correlated development photos, not an independent-site benchmark or video. The DACL fixed two-output baseline on these same 18 had wet-stain-proxy 2 TP /15 missed /8 unmatched and material-loss-proxy 1 /15 /6; the new five-class model has wet-stain 4 /13 /6 and material detachment 0 /16 /1. These are different models and class definitions crosswalked by appearance, not proof of a universally better detector.

There are no certified healthy-wall negatives, so healthy-wall false-positive rate remains unavailable. The source photographs do not establish measured humidity, concealed leaks, paint material, adhesion, bulge depth, installation flatness or Saudi-code compliance. The candidate is suitable only for explicitly failed research diagnostics, not automatic capability promotion.

`output/evaluation/construction-wall-five-class-v3/integration-proposal.json` contains the diagnostic-only variant proposal, all class metrics, exact model/metadata/point hashes and the 18 original/publisher-GT/model comparison paths. It does not edit app configuration. Before releasing the shared runtime source for subsequent registration changes, `scripts/archive_mbdd_wall_v3_result.py` archived the exact executed trainer, evaluator, wrapper and runtime source. The archive manifest is `output/evaluation/construction-wall-five-class-v3/executed-source-archive/manifest.json`, SHA256 `77a49e977da64cc8ae2e0224c3ce55c691c475fa94ea0080be44ffb1a7384c03`; auxiliary dependencies are explicitly distinguished from the four pre-pinned core identities.
