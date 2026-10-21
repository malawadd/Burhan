# Factory and construction improvement: rounds4–5

20 October2026. The bounded training, calibration, localized-mask, real-video and laptop benchmark experiments below have completed. **The requested model-quality target is not achieved.** Training loss, output activity and software tests are not inspection accuracy. Failed candidates remain labelled as diagnostics; no new candidate replaces the retained defaults.

## Requested result and measured starting point

| Factory model / scope | Defect-view localized recall | Unaffected-view alarms | Laptop throughput evidence | Decision |
|---|---:|---:|---|---|
| Previous v10h, exposed80-view development set |10/11 (90.9%) |5/69 |Previously completed16FPS replay |Recall target unmet |
| C200 phase-max refinement, same exposed set |10/11 (90.9%) |1/69 |19.662 complete-profile capacity FPS;960/960 at16FPS, no drops/late, P9563.4ms |Fewer alarms, recall target unmet |
| New step600 PRIMARY / SECONDARY |10/11 (90.9%) |2/69 each |Fresh19.19/19.25 paired full-profile FPS; sustained replay drops/late decisions |No promotion |
| New step300 POST-CAL exploratory |10/11 (90.9%) |1/69 |Fresh20.43 paired full-profile FPS; sustained replay4 drops/50 late |No improvement over C200 |
| Round5 thin-structure step600 |10/11 (90.9%) |1/69 |Fresh19.645 first64 capacity FPS; sustained955/960 processed,5 drops,25 late |Recall/false-alarm target unmet |

The goal is **greater than90.9% localized recall, fewer false alarms and at least15 complete-profile FPS** on this laptop. Both baselines matter: reducing alarms below5/69 improves the retained v10h model; improving on the best C200 result of1/69 requires0/69 on this tiny comparison. A new model with11/11 localizations and2/69 alarms would improve v10h while worsening C200's alarms, so it must not be called a complete improvement over the best prior candidate. This reporting distinction changes neither frozen calibration arm nor checkpoint policy.

The old80 views come from ten defective physical items;69 are unaffected views of those items, not a verified population of wholly good products. This development benchmark therefore does not measure production good-product false rejects. Pixel overlap, region localization and all misses remain part of the comparison rather than relying on image-level scores alone.

The [independent factory study/review](factory-round4-supervised-independent-review.md) and [dataset research](factory-model-data-research-round4.md) record the real sources and fixed protocol. The new fit uses3,444 accepted TRAIN images across SDD1, protected SDD2 inner-FIT and1,340 newly acquired MTD photographs. Four MTD defect-category images with empty masks are quarantined. Mandatory crops retain all projected annotated defect pixels, including thin strokes. Different illumination views are correlated; source counts do not imply independent parts/sites. Official SDD2 TEST remains sealed.

The three snapshots and both prospectively declared calibration-policy arms must freeze before the exposed comparison is read. A candidate must first demonstrate actual fitted-model15FPS capacity; architecture history cannot supply speed credit. Root will then run the original localization comparison and paired complete-profile16FPS replay without project training/export/inference contention. Host background workload is recorded rather than disabled.

The [root benchmark protocol](factory-round4-root-benchmark-protocol.md) specifies the exact gates, comparators, first64 calls, localized-mask audit, balanced paired calls and60second/960-observation runtime checks. It is prepared source, not a completed speed result.

### Factory calibration completed; exploratory comparison declared

All three snapshots completed the fixed80-view public calibration (10 defect views,70 unaffected views), and both original policy arms selected step600. Its PRIMARY point localizes10/10 at mask IoU≥.1 with1/70 unaffected-view alarm; SECONDARY localizes10/10 with2/70. These are calibration results, not new test accuracy. The arms independently shortlist seed thresholds, so their outcomes need not be monotonically better as the alarm budget grows.

Step300's existing PRIMARY point also localizes10/10, with0/70 alarms, pixel precision .662620 and recall .863348. Before any root speed measurement or original-set inference, a separately labelled **POST-CAL exploratory comparison** was declared: rank only the three already selected PRIMARY points by meaningful recall, then fewer alarms, then their existing localization rank. No refit, new threshold grid or additional source scoring is allowed. This selects step300 (.995 seed/4-pixel component/.7 growth) as an additional candidate while preserving both original step600 choices. It must pass exactly the same capacity, localization and runtime gates; no original-result fallback is permitted. [Exploratory proposal](../output/optimization/factory-round4-lower-alarm/post-CAL-exploratory-proposal-before-any-benchmark.json).

### Completed factory fit and export checks

The complete4,096+900-update fit finished in2,356.70seconds (39.28minutes) under ordinary host/project contention. All three fixed checkpoints, optimizer states and own Torch reference outputs are saved; all TRAIN-source hashes and frozen normalization buffers remain unchanged. [Completion receipt](../output/optimization/factory-round4-supervised/training-complete.json).

All three actual OpenVINO GPU.0/f16/one-stream numerical checks passed against each checkpoint's own three Torch cases. Maximum score-map absolute errors were .00261942 (step300), .00241050 (600), and .00131253 (900), below the fixed .01 guard. Outputs are finite, bounded and nonconstant. These checks validate export consistency, not recall or FPS.

The first calibration command stopped **before CAL model initialization/inference**, because the new identity helper used compact JSON while the protected historical80-source hash used JSON with spaces. The same ordered paths, image/mask hashes and native geometry are retained. A separately reviewed recovery wrapper restores only that historical metadata serialization and adds executed-source/receipt lineage. It reuses the untouched three numerical results and leaves models, calibration data, grids, thresholds and rank rules unchanged. The failed source/log and recovery receipts remain in the round4 output folder. The identity proof itself reads authorized CAL inputs; it is not a claim that no CAL pixels were opened.

## Construction: train real masks, retain genuine video tests

The construction fit adds64 real CUBIT-InSeg photographs with publisher crack/spalling instance polygons. Existing DACL official TRAIN polygons, actual ev-CIVIL camera boxes and verified two-target-negative photos provide complementary supervision. The prepared1,296 TRAIN views retain source photograph identities; crops are not new photographs. The private owner videos never enter training or calibration.

Mask learning uses actual source pixels. Camera box annotations supervise detection only, with unknown classes/anchors ignored; rectangular boxes never become segmentation truth. Five fixed, source-reset TRAIN-only gradient/memory checks precede a fixed900-update fit with snapshots300/600/900. Each snapshot requires Torch/export/runtime numeric checks and full serving parity before the unchanged122-frame public calibration. All outcomes and one global choice freeze before full public-video and owner-video comparisons. Failing guards means no supported-model promotion.

The [construction plan](construction-round4-mixed-supervision-plan.md) records exact sources, losses, licenses, checkpoints and pass/fail rules. Existing public thin-crack and spalling failures remain visible and must be compared using complete source timelines. Actual pixel-mask quality and publisher whole-instance box matches are different measurements.

### Actual training checks and loss optimization

The first real probe exposed one-ULP OpenCV mask-resize roundoff (`1.000000119`). V1's failure and source archive remain preserved. V2 accepts only bounded±1e-6 roundoff, locally clamps coverage to[0,1], and rejects larger/nonfinite values; source masks/strokes remain unchanged. All five V2 real-data gradient/memory checks passed.

V3 groups anchors sharing an actual GT instance/ROI, retaining the same per-anchor BCE/Dice and foreground normalization. This removes repeated full-prototype gradient scatter. Eight tiny arithmetic fixtures and all five actual-output loss/gradient comparisons passed. The real probe uses one real TRAIN forward per case; scalar-reference comparisons operate on detached copies of those same outputs, with no second real-input network call. Trusted model-constructor initialization is separate.

| Fixed real TRAIN case | V2 forward/backward check seconds | V3 seconds |
|---|---:|---:|
| Crack photo,17 GT /149 assigned anchors |24.172 |4.285 |
| Spalling photo,2 GT /20 anchors |5.325 |3.045 |
| Crack camera box |1.470 |1.327 |
| Mixed camera boxes |1.369 |1.254 |
| Maximum-count crack photo,46 GT /272 anchors |40.684 |7.428 |

These are contended **training-check timings**, including gradient diagnostics, not model inference FPS or accuracy. V3 reference maximum loss-vector difference is9.54e-7 and maximum coefficient-gradient difference1.33e-8. Peak memory including reference math stays below847MB. Frozen early layers/BN remain exact, ignored-label gradients are zero, and box-only samples have zero mask/prototype-head gradients. Some GT instances receive no detector-anchor assignment (16/17 and44/46 in the two crack cases), which is explicitly recorded rather than called complete supervision.

Receipts: [V2 actual checks](../output/training/construction-round4-mixed-v2/train-gradient-probe.json), [V3 actual checks](../output/training/construction-round4-mixed-v3/train-gradient-probe.json). The fixed900-update CPU fit completed in1,644.07seconds (27.40minutes), with all finite/frozen-layer checks passing. Actual sampling used502 distinct prepared views, including60 of64 newly acquired InSeg photographs and75 existing DACL photographs; acquisition/preparation is not a claim every photograph was sampled. All three snapshots are immutable and exported.

Step300's first actual public calibration stopped on source21 because cached direct inference and the complete serving profile produced different scores, boxes and native masks. The saved raw cache reproduces its own original decode exactly, ruling out cache overwrite in this case. A separately frozen six-call same-source technical diagnostic captures both routes' actual tensors, raw outputs and native masks. It does not score ground truth, tune thresholds or waive the failure. Cross-route differences also include two separately compiled instances and cannot alone identify a serving-code cause. Step600/900 calibration remains held until this is resolved transparently. [Independent parity review](construction-round4-step300-parity-failure-independent-review.md).

## Four new owner recordings

The owner identified wall cracks, wall humidity damage, upper-wall cracks and incorrect fastening. All878 original exposures completed a three-model BEFORE run. An independent audit verified all original hashes/timestamps and2,515 saved actual masks. These recordings have no complete frame ground truth: candidate counts are unscored, not recall/precision or verified fault counts.

The [owner-video report](new-construction-video-review-round4.md) links all four complete videos, native masks, per-frame reports, qualitative failures and actual dashboard proof. The [independent audit](user-construction-videos-round4-independent-audit.md) records exact checks/counts. In the dashboard, enable failure diagnostics and choose an Owner… three-model BEFORE case. Original raw-file presets are available separately from the annotated comparison.

The reported fastening issue is **fastener/anchorage installation**. This category remains unsupported by installed models. Visible wet/stained appearance also does not measure wall moisture or room humidity. Broad appearance labels on furniture, floors or boundaries are not useful confirmation of the owner's wall concern; those failures are retained in the recordings.

## Software verification so far

The updated evidence API preserves unknown metrics as null for wholly unannotated/unscored recordings while retaining evaluated-match geometry checks. Full serving-environment suite: **462 passed,3 skipped,1 dependency deprecation warning in49.03s**. Training-only Torch arithmetic fixtures run separately in the export environment. [Local delivery checks](../output/evaluation/user-construction-videos-round4/localhost-delivery-v1.json) verify all four video/report routes, twelve original-file presets and the unsupported fastening category. Software checks do not establish model accuracy or15FPS.

## Actual factory result and continuing work

The [complete measured factory report](factory-round4-measured-results.md) records all three localized-mask audits, all five balanced speed comparisons and all five first60second runtime attempts. **No new policy exceeds90.9% meaningful recall.** Step600 raises2/69 unaffected-view alarms; step300 raises1/69 while reducing pixel recall below C200. Every new policy misses `kos12/Part5.jpg` completely. Four-panel evidence and exact native GT/prediction masks are saved for every defect and normal alarm, including this failure.

All three new first64 capacity checks exceed15FPS (19.783/17.543/20.477), and all five paired averages exceed19FPS. Sustained replay exposes deadline tails and losses: primary954/960 processed,6 drops,28 late; secondary950/960,10 drops,92 late; exploratory955/959,4 drops,50 late. Historical C200's fresh run also deteriorates to950/959,9 drops,99 late; v10h processes960/960 but has2 late. No current run passes the stricter zero-loss/100ms-deadline check. Reused-still replay remains an engineering measurement rather than real production-line coverage.

The separate [Round5 thin-structure experiment](factory-round5-cldice.md) completed600 fixed updates in475.48seconds (7.93minutes), with all three checkpoints/numerical/calibration outcomes frozen before original scoring. The selected step600 uses the same serving graph and training-only connectivity loss. It still localizes10/11 defects and flags1/69 unaffected views. Pixel recall rises75.40→79.45% versus C200, while precision falls61.54→58.75% and strict mask IoU≥.5 hits fall4/11→3/11. The same view remains completely missed. [Fresh Round5 measured results](factory-round5-measured-results.md) retain all first runtime attempts; no model is promoted.

### Construction completion and actual failures

CPU technical recovery completed the same-author baseline plus all three trained checkpoints on the unchanged122 public CAL frames. Numeric, identical-input and complete-profile native-mask parity checks all pass. Every trained checkpoint fails the quality guards. Final900 improves spall box matches0→7/38 but raises51 unmatched spall predictions; thin matches regress7→2/62. Its exact asset is used only as a **FAILED diagnostic** across all subsequent videos/photos. [Complete actual tables and native-pixel results](construction-round4-measured-results.md).

All five complete public recordings (683 original exposures) and all four private owner recordings (878 exposures) have actual before/after exports. Independent audits verify original/replay pixels, timestamps, source hashes, masks and report counts without further NN calls. Public outputs contain1,092 nonempty native masks; private outputs385. These counts establish saved-output integrity, not useful segmentation. Thin challenge remains0/8 matches and spalling75 remains0/77. The supplementary spalling76 gains47 box matches but its inspected foreground spreads over window/wall context rather than faithfully tracing material loss. Daytime and night results also retain substantial misses/false foreground.

The55-photo actual polygon benchmark runs both models on CPU: crack pixel recall2.93→5.35% with precision49.09→33.39%; spall recall remains about0.23% while precision collapses85.31→2.78%. This is not a thin-crack/spalling fix. The [owner-video report](new-construction-video-review-round4.md) includes all four full AFTER recordings, exact native masks, diagnostic activity counts and inspected failures. Wall humidity and fastener adequacy remain unsupported; generic boundary masks do not establish either reported fault.

## Final software and local delivery

The nine new AFTER recordings and nine matching original-input presets are integrated as FAILED diagnostics, preserving existing defaults and earlier evidence. The new construction route requires explicit CPUORT/CPU/one-thread options and checks the actual provider/session. Final serving software suite: **496 passed,9 skipped,1 dependency deprecation warning in40.86s**. Training-only loss/probe fixtures passed separately; software tests do not repair model accuracy or runtime losses. All nine localhost report/video routes and source presets pass independent delivery checks. The [owner-video delivery section](new-construction-video-review-round4.md#final-local-after-dashboard-delivery) includes the real UI proof and provenance.
