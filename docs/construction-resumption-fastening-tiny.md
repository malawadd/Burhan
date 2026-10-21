# Fastening learning sanity: the missing target had no positive anchor

Date: 2026-10-21. **These are repeated TRAIN photographs, not validation accuracy.** Two fixed 64-update experiments demonstrate that the local SSDLite pipeline can learn its explicit targets, and isolate a target-assignment failure. Neither checkpoint is installed or promoted.

The fixed eight-photo subset contains seven unseated-region targets, two absence targets and four publisher Fixed photographs. Two pure-category photographs per defect class and four Fixed photographs were chosen by a declared source-byte hash from the corrected TRAIN set before inference. Their source identities are `Loosen-12`, `Loosen-91`, `Missing-128`, `Missing-171`, `fixed-RAW_(6)`, `(77)`, `(271)` and `(194)`.

Both runs start from **the same existing SDNET SSDv1 checkpoint**, SHA-256 `aaf27e5c14e636a0226fbbce65ca5201a863b9fbdd5a0705fc479c0d41176e1d`. They use the same eight originals, 64 full-eight-image batches in the same seeded order, head LR .001, late-feature LR .0001, AdamW weight decay .0005, frozen backbone running statistics and background-aware loss. Only late `features.1` / `extra` backbone parameters and the heads train. Actual Torch thread count is one in every update. No CAL/development/owner pixels are read.

## Actual before/after localization

Matches require the correct category and box IoU ≥0.5. “Unmatched” is relative to source annotations, whose completeness is still unverified.

| Same eight TRAIN photographs | Confidence | TP | Unmatched predictions | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Pinned SSDv1 before either fit | .25 | 3 | 32 | 6 | 8.57% | 33.33% |
| After64, stock forced-anchor matcher | .25 | 8 | 0 | 1 | 100% | 88.89% |
| After64, unique forced-anchor matcher | .25 | 9 | 0 | 0 | 100% | 100% |
| Pinned SSDv1 before either fit | .50 | 0 | 0 | 9 | — | 0% |
| After64, stock matcher | .50 | 8 | 0 | 1 | 100% | 88.89% |
| After64, unique matcher | .50 | 9 | 0 | 0 | 100% | 100% |

With the unique matcher, unseated-region localization is **7/7**, mean matched IoU **0.956**; absence is **2/2**, IoU **0.805**. At confidence .25, the four Fixed TRAIN photographs change from **4/4 flagged, 20 boxes** to **0/4 flagged** in both fitted runs. At .50 the initial detector already predicts no boxes anywhere, so fewer boxes at that threshold alone would not establish improvement.

The first fit's mean total loss falls 3.233→0.047 across its first/last eight updates; the unique control falls 3.403→0.0467. Similar low loss concealed a missing supervised object in the first fit. Elapsed fitting times were 101.7s / 67.9s under different concurrent workloads; they are **not a matcher speed comparison**.

The following full photograph is shown with publisher GT, the identical initial model and the unique-matcher 64-update result. All five explicit unseated targets in this TRAIN photograph now have separately localized boxes. A visible large hole has no absence annotation in this particular publisher record, so the panel does not establish coverage of every possible physical issue.

![Same TRAIN photograph, GT / initial model / unique-matcher 64-update result](../output/evaluation/construction-resumption-fastening-tiny-unique-v1/full-frame-overlays/exact-5827dcf3199bc714879c.jpg)

Per-original before/after panels for all eight photographs are in `output/evaluation/construction-resumption-fastening-tiny-unique-v1/full-frame-overlays/`. The renderer places captions outside the original image. It displays confidence ≥.25, at most 20 boxes, while metrics include every qualifying prediction. Green/cyan polygons in the GT panel are genuine source masks; **SSDLite predicts boxes, not masks**. No predicted-mask accuracy is claimed.

## Verified assignment failure and narrow repair

For `Loosen-12`, two adjacent GT boxes both choose anchor **1101** under the stock best-anchor rule. The original matcher overwrites one forced match. Its actual per-GT positive-anchor counts are **[1,0,1,1,1]**. The target with zero positive anchors is exactly the remaining missed object after the first 64 updates. Low training loss did not mean all targets had been supervised.

`scripts/construction_ssd_matcher.py` preserves ordinary IoU-threshold positives, then applies rectangular `linear_sum_assignment` on the actual GT-by-anchor IoU matrix to reserve distinct forced anchors. It requires at least one positive anchor per GT when enough anchors exist. The model graph, priors, decoding, score thresholds and inference NMS remain unchanged. This helper changes only the training instance; it does not patch TorchVision or the runtime.

The matched control preserves stock-versus-unique assignment counts for **all 256 positive-image calls** across 64 updates. Every GT receives at least one positive anchor. Initial predictions are exactly identical between runs, as are selected records and batch schedules. The common experiment protocol is intentionally identical; the additional wrapper protocol separately binds the matcher source, SciPy version/binary and changed behavior.

The Torch export environment initially lacked SciPy; that import failed before experiment creation or gradients. The existing project serving environment contains compatible SciPy **1.18.1**. It is reused through a process-local appended search path, with the assignment binary SHA recorded. No package was installed and neither environment nor site-packages was modified.

## Original wall TRAIN coverage audit

The same unchanged 3,234 SSD320 priors were checked against all **40 original wall TRAIN-positive photographs / 113 source-stain targets**. One already selected SDNET TRAIN photograph supplies constant feature-map geometry; wall target coordinates are metadata only. No wall CAL/development/owner pixels or additional wall fitting are involved.

The stock matcher leaves **2/113 targets unassigned**, both in `Hefei10777` (capture block53), GT indices **9 and10, zero-based**. Its 12-target counts are `[2,8,1,1,1,1,1,1,1,0,0,1]`; the unique matcher gives `[2,8,1,1,1,1,1,1,1,1,1,1]`. The three trailing targets share original best anchor1682. This is a real supervision defect but affects too few wall targets to explain the large TRAIN/CAL gap by itself. It does not establish moisture semantics or generalization.

The full IDs, boxes, per-GT IoUs and assignment counts are preserved in `output/evaluation/construction-resumption-fastening-tiny-unique-v1/wall-original-TRAIN-anchor-coverage.json`.

## Regression checks and reproducible evidence

`tests/test_construction_ssd_matcher.py`: **8 passed** in the Torch environment; **8 skipped** in the serving environment where Torch is intentionally absent. Tests cover shared-best-anchor collisions, unchanged noncollision threshold positives, empty targets, nonfinite/wrong-shape inputs and too few anchors.

Relevant implementation and machine records:

- `scripts/sanity_fit_sdnet_tiny.py`: original background-aware 64-update experiment; exact executed source is archived with the output.
- `scripts/sanity_fit_sdnet_tiny_unique.py`: matched matcher-only control and schedule verification.
- `scripts/construction_ssd_matcher.py`: training-only unique assignment.
- `scripts/audit_sdnet_tiny_anchor_coverage.py` and `scripts/audit_wall_stain_anchor_coverage.py`: actual assignment audits.
- `output/evaluation/construction-resumption-fastening-tiny-v1/`: original protocol, pinned initialization, final state, full loss/gradient/batch history, predictions, metrics and real-photo overlays.
- `output/evaluation/construction-resumption-fastening-tiny-unique-v1/`: matched control, all-update coverage, integrity receipt, wall audit and full-frame overlays.

The original final-state SHA is `f153d55b48e3ed62dbdc524bcbe0e7e4ca11dd019ca3457b434c088c33aea99b`; the unique-matcher final SHA is `5fe28497975e34c339e501823888eefe11d574222ffee471c799b2226eb628a0`. These tiny memorization checkpoints stay outside the installed-model registry. The next generalization experiment must use the corrected larger TRAIN data, preserve CAL-only selection and obtain new independent evidence before any release.
