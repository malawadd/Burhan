# Local factory surface model

**Accuracy-study update:** later experiments did not improve genuine localized-defect recall on the now-reused 80-image comparison; the best remains 8/11 (72.7%). A review setting flagged 9/11 defective images, but its extra box was away from the real defect. See the [complete accuracy, mask-overlap, cross-validation and sustained-speed study](../factory-accuracy-improvement.md). It also records rejected learned models, precision/false-alarm costs, abstentions and the unchanged baseline artifact. No production acceptance claim or automatic accept/reject authority follows from these results.

**Performance update:** the model, features and thresholds below are unchanged, but inference now supports exact threshold cascades and a locally compiled CPU loop. The original timing and 1 FPS recommendation below are historical baseline measurements. See the [before/after factory performance report](../factory-performance.md) and generated comparison JSON for the current backend, speed and limits.

The factory profile now has two explicitly selected modes: the existing approved-reference inspection for an operator's fixed camera/product, and a compact supervised surface-defect model for Kolektor-like grayscale commutator close-ups. The latter uses real industrial pixels and learned parameters. It is not a general model for PCB inspection, all machinery, or arbitrary camera scenes.

## Public dataset selection

[KolektorSDD, official ViCoS Lab page](https://www.vicos.si/resources/kolektorsdd/) provides images from a controlled industrial acquisition, pixel masks, and physical item folders. The downloaded archive contains 399 inspection images: 347 without a visible annotated defect and 52 with defects, across 50 items. Most images are 500 pixels wide and approximately 1,250 pixels high. Its relatively small download and explicit item identity make it suitable for this laptop and a reproducible item-separated experiment.

[MVTec AD, official page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) was also considered. Its multiple object and texture categories would support broader anomaly experiments, but each family still needs separate adaptation and validation. It was not downloaded or evaluated in this implementation; no MVTec accuracy claim is made.

A normal label here means **no annotated defect is visible in that image**. Another view of the same physical item can contain a defect. The profile issues per-view observations and cannot certify an entire item from one normal face.

KolektorSDD is **CC BY-NC-SA 4.0**, with commercial use directed to the dataset authors. This downloaded dataset and derived model are for a non-commercial research prototype, not an unrestricted commercial artifact. Preserve attribution and the license when sharing adaptations; obtain suitable rights or replace training data for commercial deployment. Code and dataset/model rights are distinct.

Attribution: Domen Tabernik, Samo Šela, Jure Skvarč, Danijel Skočaj, “Segmentation-Based Deep-Learning Approach for Surface-Defect Detection,” Journal of Intelligent Manufacturing, 2019, [DOI](https://doi.org/10.1007/s10845-019-01476-x). This project uses the dataset and masks, not the authors' neural network or their reported benchmark results.

## Acquisition and provenance

| Artifact | Value |
|---|---|
| Acquisition reviewed | 20 October 2026 |
| Official page | https://www.vicos.si/resources/kolektorsdd/ |
| Official short link | https://go.vicos.si/kolektorsdd |
| Resolved archive | https://data.vicos.si/datasets/KSDD/KolektorSDD.zip |
| Local archive | `datasets/factory/KolektorSDD.zip` |
| Archive bytes | 101,831,129 |
| SHA-256 | `65dc621693418585de9c4467d1340ea7958a6181816f0dc2883a1e8b61f9d4dc` |
| Extracted dataset | `datasets/factory/KolektorSDD/` |
| Model | `models/factory/kolektor_surface_rf.joblib` |
| Metadata | `models/factory/kolektor_surface_rf.joblib.json` |
| Machine-readable evaluation | `output/evaluation/factory-real-input.json` |

The checksum is pinned from this acquisition, not an independently published author checksum. The download helper checks size, checksum, safe extraction paths and expected image count. There are no dataset or model downloads during inference. Public artifacts remain in ignored local directories rather than being silently committed.

## Reproduce training

From the project root, with the project's dependencies installed:

```powershell
.venv/Scripts/python.exe scripts/train_factory_surface.py --download
```

If the pinned archive already exists, the script verifies it and reuses local data. The seed is `20261020`. Physical item IDs are shuffled once and split into 30 training, 10 calibration and 10 test items. All views of each item remain together. The split item IDs, image counts, threshold choices, complete test predictions and timing are written to the report. It is a **custom holdout**, not the authors' official three-fold benchmark.

Training samples annotated defect pixels and up to 2,000 random normal pixels per training image. Up to 400 positive pixels per defect image are repeated twice within training to reduce class imbalance. The implementation resizes each image to width 256 while preserving aspect ratio, extracts 11 OpenCV grayscale/multiscale texture features, and fits a 32-tree Random Forest with maximum depth 12 and minimum leaf size 4. There is no pretrained backbone. A local statistical model is trained from this dataset using [scikit-learn's RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html); OpenCV performs feature extraction, connected components and overlays.

The pixel score is the mean of tree leaf defect fractions. It is not a calibrated defect probability. Thresholds and minimum connected-component area are selected on calibration images only from a documented finite grid, maximizing image F1 with precision/recall tie-breaking. Test image labels do not enter fitting or threshold selection. A test image is positive when at least one component passes the selected thresholds; this is not a pixel segmentation benchmark. Tiny defects can disappear during resizing.

OpenCV 5.0.0 in this environment has no `cv2.ml`; the first attempted RTrees implementation failed that capability check. The delivered implementation uses scikit-learn 1.7.2 instead of downgrading OpenCV. No GPU, PyTorch, paid inference or cloud upload is used.

## Use actual inputs

Start the factory profile with an explicit model configuration:

```json
{
  "defect_model_path": "models/factory/kolektor_surface_rf.joblib",
  "confirmation_frames": 1
}
```

Use the held-out positive example `image:datasets/factory/KolektorSDD/kos04/Part3.jpg`, normal example `image:datasets/factory/KolektorSDD/kos04/Part0.jpg`, or mixed-view folder `images:datasets/factory/KolektorSDD/kos04`. These are test items, not training pixels. The positive example is one of the correctly detected test images; known misses are listed below. Preserve portrait geometry; the supplied model explicitly supports height/width ratios 2.3–2.7. Nonmatching geometry becomes `inconclusive`; blank/poor-quality input becomes `needs_input`. These checks are acquisition guards, not reliable out-of-distribution detection. A similarly shaped unrelated surface can still receive an incorrect decision.

For a real camera inspecting a different product, restart **without** `defect_model_path`, present an approved focused good product, and use the existing explicit reference calibration. This works on real input and does not apply the commutator model to an unrelated product. Train and independently validate a new task-specific model before using model mode for that family.

Use one-frame confirmation for independent images. For continuous video, the default three overlapping observations and event latching remain available; these frames are not independent samples. Manual unit triggers use per-unit deduplication and the configured triggered confirmation count. Neither mode promises every-item coverage or includes a reject actuator.

Missing model/metadata, checksum mismatch, incompatible feature version and invalid thresholds fail explicitly. The model mode refuses the reference calibration action; restart in reference mode instead of silently changing inspection semantics. The model hash is exposed in profile metadata and per-result state for event attribution.

The artifact is generated locally using joblib, which is pickle based. **Load only artifacts produced by the trusted local trainer.** Checksums guard accidental mismatch, not malicious files accompanied by attacker-authored metadata. Do not download a third-party pickle/joblib and load it. See [scikit-learn's persistence guidance](https://scikit-learn.org/stable/model_persistence.html).

## Measured results and limits

The generated evaluation report is authoritative for the local run on this laptop (Intel i7-8550U, CPU only, OpenCV 5.0.0, Python 3.13.5, scikit-learn 1.7.2, two forest workers). The final model is 1,721,433 bytes with SHA-256 `549d8528c1e2b8ca483b772e9ada4f71284782dca2cd631be7127171cadd0473`.

| Split | Physical items | Images | Defective views | Normal views |
|---|---:|---:|---:|---:|
| Training | 30 | 239 | 31 | 208 |
| Calibration | 10 | 80 | 10 | 70 |
| Test | 10 | 80 | 11 | 69 |

Training used 502,582 sampled feature rows: 478,000 normal and 24,582 positive after repetition. The selected threshold is **0.995**, with minimum component area **8 grid pixels**. Calibration found 10 true positives and 70 true negatives, but the independent test is the meaningful check:

| Held-out image result | Value |
|---|---:|
| True positives / false negatives | 8 / 3 |
| False positives / true negatives | 1 / 68 |
| Precision | 88.9% |
| Recall | 72.7% |
| F1 | 0.800 |
| Accuracy | 95.0% |
| Image AUROC using maximum pixel score | 0.900 |

Accuracy is dominated by normal views and must not substitute for the **three missed defective views**. The misses are `kos02/Part6.jpg`, `kos12/Part5.jpg`, and `kos38/Part1.jpg`; the false positive is `kos07/Part2.jpg`. These results were not used to retune the delivered thresholds. Initial training/calibration had excessive false positives; normal sampling and the threshold search were improved using training/calibration data before final test evaluation.

Model-only feature/scoring timing over the 80 held-out images was **446.8 ms p50, 657.9 ms p95, 701.9 ms p99 and 706.3 ms maximum**. This excludes file read, connected components, capture, queues, evidence, UI and actuation. Shared laptop load and startup affect measurements. Use **1 input frame/second and a 1,500 ms observation deadline** as the conservative initial preset for this supplied model. This is not an industrial deadline guarantee. The approved-reference mode has its own faster timing and must be evaluated separately.

The complete profile is additionally evaluated with:

```powershell
.venv/Scripts/python.exe scripts/evaluate_factory_surface.py
```

Its report, `output/evaluation/factory-profile-real-input.json`, includes quality/view abstentions, all statuses and full `FactoryProfile.process` timing with an independent unit ID per image. Capture, source decoding, runtime queues, UI and actuation remain excluded; use runtime telemetry for host-to-decision timing.

The complete profile accepted all 80 supported held-out images with **0 abstentions** and the same 8 TP / 3 FN / 1 FP / 68 TN decisions. Its separately measured process time was **365.2 ms p50, 546.9 ms p95, 587.6 ms p99 and 701.2 ms maximum**. This run had different shared-machine load from the model-only timing; the lower measured values do not imply that extra pipeline work makes inference faster. Representative annotated examples are saved beside the report for true positives, misses, false positives and normal views.

A separate real-source runtime smoke repeated held-out image `kos04/Part3.jpg` three times at 1 FPS with a 2,000 ms deadline, network connections blocked by a test patch, and event-image saving off. All three observations processed, produced one surface candidate, and had zero drops, late decisions or inspection errors. Host-receipt-to-decision timing was 300.2 ms p50 / 594.1 ms p95 / 626.8 ms maximum; model hash attribution was present. See `output/evaluation/factory-runtime-smoke.json`. These are repeated still observations, not three independent examples or evidence of sustained production performance. The test did not change the laptop's physical network settings.

The small test split and controlled image acquisition do not establish factory field performance. Expected failures include changing lighting/finish, reflections, camera/view changes, unseen defect types, normal texture confusion, small defects and resizing losses. A “clear” result means no selected candidate under this model and threshold; it is not machinery-health, dimensional-tolerance or electrical-function certification.
