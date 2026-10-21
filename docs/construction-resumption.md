# What was implemented, why I stopped, and the follow-through

21 October 2026. The previous turn completed application infrastructure and bounded experiments. It did **not** deliver reliable humidity, mounting-plate-gap or installed-tile fault detection. I used completion of those experiments and reports as the stopping condition. That was the wrong condition for the requested product.

## Actual implementation versus detection

The app does accept real local videos/images/cameras, runs local ONNX models, offers three task modes and material/reference controls, separates the raw latest-frame preview from inspected evidence, and pairs each inspected image with its actual frame/session/timestamp. Those behaviors exist in executable code. Eight installed construction checkpoints were trained and measured, but every new category failed qualification. Software tests establish mechanics, not defect accuracy. The prior preview result of23.76fps and inspected result of12.88fps were measured separately; they do not establish the requested detector quality.

The owner plate-wall separation remains missed. Publisher bolt/nut Loosen targets do not cover arbitrary mounting-plate separation. The owner damp clip had cluttered or absent candidates, and the tile path had not been properly exercised on a real installed floor. More catalog labels, boxes or model files would not resolve those failures.

## Implemented repairs and actual diagnosis

1. **Wall training loss:** the installed TorchVision SSD loss selected zero negative anchors on every empty-target photo. Eight actual TRAIN negatives produced zero classification loss and zero parameter gradient. A training-only corrected loss now selects at least3hard background anchors per empty image and uses a denominator of at least the batch size. The same real batch produces classification loss0.848351 and nonzero gradient. This is a combined negative-sampling/normalization intervention: normalization also changes six low-foreground positive batches in the200-update stream. Inference, anchors, scores and NMS are unchanged.
2. **Matched control:** a fresh original-loss200-update control uses the same current constructor/helper, seed, optimizer and exact TRAIN batch IDs as the corrected200-update fit. It reproduces every historical saved parameter with maximum absolute error0.0. This closes a provenance gap: the imported helper hash differed from the old experiment's protocol, and the first new protocols omitted that helper pin. Their unmodified protocols remain; supplemental source snapshots explicitly record the late pin. The control's helper snapshot was taken before its run.
3. **Fastening labels and anchor matching:** identical photos had incomplete or contradictory labels in separate publisher files. The corrected public dataset contains 814 exact-unique photos, combines compatible exact-photo annotations and quarantines the contradictory group. A tiny TRAIN fit exposed an additional SSD matching failure: two nearby bolts shared one forced anchor, leaving one labelled target with zero positive anchors despite low training loss. A matched 64-update control changes only the training matcher to reserve a distinct anchor for each target: 8/9 correctly localized defects become 9/9, with zero unmatched predictions, on the SAME eight training photos. This demonstrates corrected learning, not unseen accuracy. The checkpoint is not installed in the app. See the fastening record for actual assignments/overlays. The plate-gap task remains unsupported.
4. **Real tile input crash:** OpenCV returned Hough segments as N×4; the implementation assumed N×1×4 and crashed on all four real-photo/resolution checks. Both shapes now work. The repair also fixes reversed endpoint overlap, checks actual pixel support along edge bands, suppresses duplicates, reports spatial width variation and renders readable widths. These remain edge observations, not missing-grout or tile-fault classifications.
5. **Actual app verification:** the restarted server processed the licensed damaged-floor original through HTTP, the source reader, the real profile and the paired-JPEG endpoint. Three image-space bands were returned. The source reader used1920×1080analysis despite requested camera dimensions640×360; the JPEG is resized for delivery. That is a different pipeline/resolution from the640px diagnostic's eight bands. The large broken centre still is not detected as a fault. The actual displayed labels were inspected and enlarged with a dark background. The previous local-file configuration was restored and the app left stopped.
6. **UI evidence:** each of the eight known model hashes now shows its failed measured results and precise scope. Empty candidates and capture guidance cannot appear to establish that an installation is correct. The actual old fastening box was on the rail beside the connection, not the visible plate-wall gap.

## Wall learning results — calibration only

The same48CAL photos contain11positive photos/22publisher stain boxes and37other-defect target-negative photos. Physical sites are unknown. Other-defect negatives do not establish healthy-wall false-alarm rates, and source "leakage" boxes include mixed materials/exterior stains rather than verified damp interior walls. All results below are development/calibration evidence, not independent accuracy. Matching uses correct-class box IoU≥0.5.

| Fit | Fixed confidence | TP | Annotation-unmatched predictions | FN | Precision | Recall | Target-negative photos flagged |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stock loss, 200 updates | 0.15 | 8 | 6285 | 14 | 0.13% | 36.36% | 37/37 |
| Corrected loss, 200 updates | 0.15 | 5 | 1704 | 17 | 0.29% | 22.73% | 37/37 |
| Corrected loss + later features, 800 total updates | 0.15 | 3 | 76 | 19 | 3.80% | 13.64% | 21/37 |
| Stock loss, 200 updates | 0.20 | 8 | 5548 | 14 | 0.14% | 36.36% | 37/37 |
| Corrected loss, 200 updates | 0.20 | 2 | 122 | 20 | 1.61% | 9.09% | 24/37 |
| Corrected loss + later features, 800 total updates | 0.20 | 3 | 31 | 19 | 8.82% | 13.64% | 12/37 |
| Stock loss, 200 updates | 0.25 | 5 | 1026 | 17 | 0.48% | 22.73% | 37/37 |
| Corrected loss, 200 updates | 0.25 | 2 | 17 | 20 | 10.53% | 9.09% | 4/37 |
| Corrected loss + later features, 800 total updates | 0.25 | 2 | 13 | 20 | 13.33% | 9.09% | 3/37 |
| Stock loss, 200 updates | 0.35 | 0 | 23 | 22 | 0.00% | 0.00% | 13/37 |
| Corrected loss, 200 updates | 0.35 | 0 | 1 | 22 | 0.00% | 0.00% | 0/37 |
| Corrected loss + later features, 800 total updates | 0.35 | 1 | 2 | 21 | 33.33% | 4.55% | 2/37 |
| Stock loss, 200 updates | 0.50 | 0 | 0 | 22 | 0.00% | 0.00% | 0/37 |
| Corrected loss, 200 updates | 0.50 | 0 | 0 | 22 | 0.00% | 0.00% | 0/37 |
| Corrected loss + later features, 800 total updates | 0.50 | 0 | 1 | 22 | 0.00% | 0.00% | 1/37 |

The corrected 200-update fit suppresses many proposals but also misses more targets at 0.25. That does not meet the joint quality objective. The longer fixed 600-update continuation starts from corrected 200 weights, balances four positive/four negative TRAIN photos per batch, updates later MobileNet/extra layers at 0.00005 and the head at 0.0002, and freezes backbone running statistics. It is an additional experiment with several changes, not an isolated ablation of one variable. Neither experiment reads TEST or owner video pixels, and no new checkpoint is automatically installed or promoted. The full threshold table is retained to avoid presenting a convenient threshold as a solved detector.

After the continuation, a separate TRAIN-only localization diagnostic evaluates all 181 TRAIN photos. At confidence 0.25 it matches 95/113 publisher boxes (64.19% precision, 84.07% recall), with 53 unmatched predictions and six of 141 target-negative photos flagged. The same checkpoint on CAL matches only 2/22 (13.33% precision, 9.09% recall), with 13 unmatched predictions. This large TRAIN/CAL gap shows that learning these source examples has not produced useful transfer. The additional 600 updates took 648.51 seconds under other local work; that is training time, not an inference benchmark. Training source labels still do not establish dampness in interior walls.

A separate TRAIN annotation/anchor audit also found two of the 113 wall targets had zero positive anchors under the old matcher, both in `Hefei10777`. The new matcher gives every target at least one anchor. The completed wall continuation used the original matcher; its poor metrics remain reported and have not been replaced by an unrun retraining claim. Fixing two missing TRAIN targets does not establish the cause or cure of the much larger CAL failure.

Additional public photos were acquired safely, but the new publisher leakage positives were overlapping views of one peeling/mottled exterior wall. They were excluded from these humidity-oriented experiments: more filenames would not provide independent damp-wall evidence. The AV-Water Damage source is a relevant lead, but its export remains inaccessible without authorized credentials; no training archive or thumbnail substitute was used.

## Evidence, source and verification

- [Fastening diagnosis, corrected data, tiny learning tests and anchor coverage](construction-resumption-fastening-diagnosis.md).
- [Matched 64-update control, actual nine-target localization and regression checks](construction-resumption-fastening-tiny.md).
- [Real installed-floor before/after execution, source licenses and defect-transfer failures](construction-resumption-geometry.md).
- [UI changes and actual video audit](construction-resumption-ui.md).
- [AV water-damage primary-source/access audit](construction-resumption-water-data.md).
- Actual final HTTP evidence: `output/evaluation/construction-resumption-geometry/live-api-v2/` (public photograph, not video). Photo: Malcolm Koo/Mk2010, CC BY-SA4.0; source and license are recorded in the protocol.
- Full fixed CAL comparisons and real-photo training history: `output/evaluation/construction-wall-learning/`. These are photographs and loss diagnostics, not fault videos. Fixed-point contact sheets use source-ID hashes rather than choosing attractive predictions.
- Loss implementation: `scripts/construction_ssd_background_loss.py`; fit/control recipes: `scripts/diagnose_wall_stain_learning.py`, `scripts/control_wall_stain_stock_loss.py`. Completed run directories preserve exact source snapshots and protocol/model hashes.
- 85 affected runtime/profile/loader/gate/provenance tests passed. After the overlay adjustment, 50 guided profile tests passed. Two Torch loss regression tests and eight matcher regression tests passed in the training environment. Matcher tests skip when Torch is absent from the serving environment. These are software checks, not additional detection samples.

The requested usable detectors remain unfinished. The confirmed learning/data defects above are concrete failures that needed repair before more full training; completing another fixed-budget fit alone does not satisfy the product goal.
