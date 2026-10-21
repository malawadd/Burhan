# Why fastening detection is still weak, and the next experiment

Date: 2026-10-21. This investigation performs actual one-thread CPU-ONNX inference on **24 preselected TRAIN photographs**, audits source annotation bytes and prepares corrected public data. Two later fixed 64-update tiny TRAIN fits investigate learning and assignment mechanics; no owner-video training or CAL/reused-TEST tuning is performed. See the [completed tiny-fit comparison](construction-resumption-fastening-tiny.md).

The strongest concrete problem is contradictory/incomplete supervision: an identical photograph can occur twice with different target fragments. More training on those records does not fix the contradiction. The model also remains poorly fitted on TRAIN, and increasing inference tiles alone makes its localization worse.

## Actual source-label failures and correction

TRAIN contains 601 records but only 591 distinct exact image hashes: nine repeated-photo groups, seven containing both publisher annotation files. One exact photograph (`Loosen-158` / `Missing-126`) labels the same two locations as different classes, with box IoUs **0.854 and 0.595**. Another exact pair (`Loosen-161` / `Missing-60`) supplies complementary labels: the first omits the foreground absence target present in the second. The following evidence displays the original photograph and each original source annotation; no fault is synthesized:

![Exact-photo annotation conflict and incomplete target fragments](../output/evaluation/construction-resumption-fastening-diagnosis/annotation-pairs-conflict-and-partial.jpg)

The implemented correction groups **only identical image SHA-256 bytes**. It inherits each original split and near-duplicate group. Complementary source annotations are unioned; overlapping same-class fragments from different source rows retain a real larger source box/polygon, with both annotations recorded. Opposite-class overlaps at IoU ≥0.5 quarantine the entire exact photograph. Labels are never transferred between near-duplicate images with different pixels.

The new immutable manifest is `datasets/construction/expansion-public-v2/sdnet2025-corrected/manifest.json`, produced by `scripts/prepare_sdnet_corrected_annotations.py`:

| Role | Before records | Corrected exact photographs | Unseated instances | Absence instances |
|---|---:|---:|---:|---:|
| TRAIN | 601 | 590 | 362 | 296 |
| CAL | 127 | 127 | 102 | 67 |
| Previously exposed development | 98 | 97 | 65 | 71 |

One conflicting exact-photo group is quarantined; nine remaining repeated groups are reconciled, including two same-class annotation deduplications. These are annotation corrections, not new independent scenes. Broader cross-class completeness, grouped-versus-individual target consistency and physical-site identities remain unverified. The former TEST remains reused development after reconciliation.

## TRAIN learning and localization evidence

The diagnostic selected eight TRAIN photographs per publisher file/Fixed family by a declared source-name hash before inference. It binds data/model/script hashes and stores every prediction in `output/evaluation/construction-resumption-fastening-diagnosis/diagnosis.json`. The deployed model runs through the actual one-thread local adapter. Confidence changes below occur only in the diagnostic process; deployed manifests and calibration thresholds are unchanged.

| TRAIN diagnostic, confidence ≥0.1 | TP | Annotation-unmatched | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Whole photograph: unseated region | 6 | 2 | 7 | 75.0% | 46.2% |
| Whole photograph: absence appearance | 10 | 12 | 2 | 45.5% | 83.3% |
| Four fixed quadrants: unseated region | 0 | 3 | 13 | 0% | 0% |
| Four fixed quadrants: absence appearance | 4 | 24 | 8 | 14.3% | 33.3% |

Whole-photograph matched-box mean IoUs are **0.848 unseated / 0.816 absence**. At confidence ≥0.5, unseated detection is **0/13 even on this TRAIN sample**; absence is 4/12. At ≥0.025, absence reaches 12/12 but has 43 unmatched predictions. Merely lowering a threshold does not provide a useful detector.

The four-quadrant experiment doubles target scale but introduces crop boundaries and a distribution the model did not learn. It uses four network calls per photograph and **worsens both precision and recall**. It is rejected as an improvement; it is not enabled in the application.

Small targets are nevertheless a real data issue: at 512 input resolution, the TRAIN absence short-side median is **17.6 pixels**, with **45.5% below 16 pixels** and **74.6% below 32 pixels**. For unseated targets these are 38.7 pixels, 13.7% and 39.9%, respectively. Upsampling cannot recover details already lost in the publisher's 640px derivatives.

The extended fit's classification loss falls from 3.59 at epoch 1 to 1.85 at epoch 8; box loss falls from 2.06 to 1.40. Both are still falling late in training. This supports further learning diagnostics but does not prove that another long fit will generalize. Intermediate validation was disabled, so zero entries in its intermediate CSV validation columns are **missing validation**, not zero-error measurements. Resume used saved EMA weights and did not restore an uninterrupted RNG trajectory; it is already documented as a distinct candidate.

![TRAIN ground truth and actual predictions at confidence ≥0.1](../output/evaluation/construction-resumption-fastening-diagnosis/train-localization-contact-sheet.jpg)

## Learning sanity completed; next larger experiment

The bounded **tiny TRAIN overfit test is complete** on eight exact-unique corrected TRAIN photographs containing both categories and four Fixed examples. The first 64-update background-aware SSD fit improves 3TP/32unmatched/6FN to 8/0/1 at confidence .25. An actual anchor audit finds that the remaining missed GT receives zero positive anchors because two targets share one forced best anchor. A matched second 64-update control changes only forced assignment to reserve unique anchors: it achieves **9/0/0 at both .25 and .50 on these same repeated TRAIN photographs**, with unchanged initial weights, images, batch schedule and inference. This establishes that the tiny explicit targets can be learned after the training repair; it is not validation accuracy. Full losses, assignments and per-original localization panels are in the [tiny-fit report](construction-resumption-fastening-tiny.md).

The next larger experiment should use the reconciled 590-photo TRAIN set, collision-safe assignment, verified background gradients, balanced category/background sampling and real-image scale augmentation that matches any eventual guided ROI input. Compare whole-frame inference and the trained ROI path on **CAL**, with a fixed training budget and validation recorded during fitting. Retain candidate checkpoints and select on CAL only. Quarantine manually ambiguous source targets instead of silently assigning a preferred class. A fresh, physically grouped public holdout is still needed for release; the old development set cannot become fresh through renaming. The two tiny checkpoints remain uninstalled.

The SSDLite architecture has a separate confirmed training pitfall: TorchVision's standard foreground-ratio hard-negative mining can assign zero classification loss to an entirely empty-target image. The root agent is comparing an explicitly corrected negative loss on the wall task. Any later SSD fastening fit must verify nonzero finite background gradients on true negative images before treating the 235 original TRAIN Fixed records as effective negative supervision.

## The owner's plate-to-wall gap is a different target

SDNET's loose-nut annotation focuses on a nut/connector separation region; its absence class covers missing fastener-seat appearances. The [authors' annotation discussion](https://link.springer.com/article/10.1007/s41062-025-01860-y) describes those targets. Neither the source labels nor a good SDNET score establishes detection of an arbitrary **mounting plate separating from a wall**. The owner's recording remains diagnostic only, without entry into TRAIN/CAL or threshold selection.

A targeted public-data search did not establish a suitable licensed, annotated real plate-to-wall-gap dataset with correct installations and construction-layout references. [NPU-BOLT](https://arxiv.org/abs/2205.11191) labels bolt views, not plate-wall installation gaps. [SRConED](https://amaelkady.github.io/SRConED/index.html) is structural connection experimental data with a noncommercial license, not a verified installation-gap image benchmark. The [Brackets_data author repository](https://github.com/oyekanjohn/Brackets_data) relates to an aerospace bracket experiment; its repository landing page does not establish construction mounting-gap annotations or training rights for this task. CAD/synthetic bracket corpora and isolated inspection-manual illustrations are not accepted as the required real public fault-training set.

This is a source search result, not proof that no suitable dataset exists. A plate-gap category needs its own real labelled interface views, intact examples, and an expected flush/stand-off layout. SDNET class names will not be broadened to hide this gap.

## Additional wall-data acquisition requested during diagnosis

`scripts/prepare_wall_stain_additional_train.py` acquired **26 real public MBDD originals**, each checked against publisher ZIP member size/CRC plus SHA-256; the original XML range and its members were verified. Selection was frozen before downloading pixels. Strict exclusion of all old TRAIN/CAL/TEST/protected blocks initially left zero free source blocks. The root then explicitly allowed existing TRAIN-only blocks absent from CAL/TEST/protected.

Only 13 valid unused publisher-leakage originals existed under that relaxed rule, so acquiring 100–200 new positives from those blocks was impossible. Thirteen hash-selected other-defect negatives were also acquired. Two new-TRAIN near duplicates were quarantined. A stricter follow-up in `scripts/finalize_wall_stain_additional_train.py` additionally excludes eight photos from blocks exposed by the old 120-photo TEST selection, including its subsequently quarantined views, and two near duplicates of existing TRAIN. No replacements were chosen after viewing pixels.

The resulting **publisher-stain research manifest** is `datasets/construction/expansion-public-v1/wall-stain-additional-train-v2/prepared/manifest.json`: 14 additions (10 source leakage / 4 other-defect), with old CAL48/TEST98 records and bytes unchanged. All ten positive additions depict overlapping views of **one mottled/peeling exterior wall**, not ten independent dampness cases. The root reviewed these and explicitly declined to add them to a humidity fit. These originals and all exclusion receipts remain auditable; no humidity-training success is claimed.

![Acquired public wall photographs, publisher leakage boxes in green](../output/evaluation/construction-resumption-fastening-diagnosis/additional-wall-train-photos.jpg)

All these steps use public-data downloads or local CPU computation. The two tiny experiments each perform exactly 64 updates on the same eight TRAIN photographs, with one Torch thread and no CAL/development/owner pixels. No larger-data fastening fit, model installation or application model promotion is performed by this diagnosis. Eight matcher regression tests pass in the Torch environment and skip in the serving environment.
