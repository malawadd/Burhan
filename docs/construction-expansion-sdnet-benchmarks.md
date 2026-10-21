# SDNET fastening experiments: before and after

Date: 2026-10-21. **Neither first candidate passes the release gate.** Real training and localized evaluation were completed; weak results are retained rather than presented as successful fastening inspection.

## Data and evaluation contract

See [the source audit](construction-expansion-data-audit.md). The fixed data contains 826 unmarked public photograph derivatives and actual polygon annotations. The split is 601 TRAIN, 127 CAL, 98 TEST; physical structure/site independence is unknown. TEST has 65 annotated unseated-connection instances in 40 photographs and 71 absence instances in 28 photographs. Thus even photograph-level positive counts fall below 50 per category.

Detection matches require the correct category and **IoU ≥0.5**, one prediction per ground-truth instance. The numerical FP column means **annotation-unmatched predictions**, not proven physical false alarms. A subsequent manual TRAIN audit found that opposite-class annotation completeness is not established: a loosened-photo file need not label every missing-seat appearance, and vice versa. This is localization evaluation relative to available labels, not merely classification of a defective photograph. Real photographs are not represented as video.

“Before” below means a **new task head before supervised fitting**, transferred from an official COCO initializer. It is not the previous crack detector or a previously working fastening model. Each architecture has its own initializer; cross-architecture baseline scores are not directly comparable.

## Fixed-budget first comparison

| Candidate | Fixed fit | Threshold selected using CAL | TEST TP / FP / FN | Precision | Recall |
|---|---|---:|---:|---:|---:|
| YOLOv8n | 2 epochs, 320px, batch 8, freeze first 10 layers | 0.50 | 4 / 0 / 132 | 100% on only 4 predictions | 2.94% |
| SSDLite320 | 200 updates, batch 8, frozen MobileNet backbone | 0.25 | 25 / 227 / 111 | 9.92% | 18.38% |

The first threshold policy prioritizes ≥95% aggregate CAL precision and then recall; when none qualifies, it uses F1. For YOLO, this selects a sparse point which detects no unseated connections. Its apparent 100% precision is **not useful category coverage** and is not a release-quality success. The stronger continuation uses per-category calibration instead of allowing one category to mask another.

At the common fixed threshold 0.50:

| Candidate | Before TP / FP / FN | After TP / FP / FN |
|---|---:|---:|
| YOLOv8n | 5 / 69 / 131 | 4 / 0 / 132 |
| SSDLite320 | 0 / 0 / 136 | 0 / 0 / 136 |

### Per-category results at each CAL-selected threshold

| Candidate / category | TEST TP / annotation-unmatched / FN | Annotation-relative precision | Recall | Mean IoU of matched instances | Flagged photographs without that label annotated |
|---|---:|---:|---:|---:|---:|
| YOLO / visibly unseated | 0 / 0 / 65 | 0% | 0% | — | 0 / 58 |
| YOLO / absence appearance | 4 / 0 / 67 | 100% on 4 predictions | 5.63% | 0.787 | 0 / 70 |
| SSD / visibly unseated | 19 / 160 / 46 | 10.61% | 29.23% | 0.670 | 33 / 58 |
| SSD / absence appearance | 6 / 67 / 65 | 8.22% | 8.45% | 0.686 | 20 / 70 |

The last column is retained from the initial protocol as **photographs without that category annotated**, including opposite-class photographs; it is not a verified healthy-negative count. There are only **30 TEST photographs in the publisher's Fixed cohort**. Fixed-only flags are the useful separately scoped normal-view comparison, and even those lack verified physical-site independence and fall below 50 negatives. Later scope-aware summaries preserve the frozen thresholds rather than retuning them from manual inspection.

Post hoc Fixed-only analysis at the already-frozen thresholds:

| Candidate / category | Flagged Fixed photographs | Predicted boxes on Fixed photographs |
|---|---:|---:|
| YOLO / visibly unseated | 0 / 30 | 0 |
| YOLO / absence appearance | 0 / 30 | 0 |
| SSD / visibly unseated | 21 / 30 | 67 |
| SSD / absence appearance | 8 / 30 | 17 |

SSD counts come from the actual CPU-ONNX served prediction records, with a hash-bound receipt in `fixed-negatives-posthoc.json`. YOLO's zero counts follow from its zero predictions on every photograph lacking that label at its frozen point. No thresholds or checkpoints changed for this post hoc annotation-scope review.

## Speed observations and their limits

| First experiment | Observed speed | Timing scope |
|---|---:|---|
| YOLO CAL, batch 8 | 18.21 photographs/s | Includes image decode, preprocessing and NMS; batch throughput, not a live single-frame rate |
| SSD CAL, batch 1 | 9.85 photographs/s; p50 97.1ms, p95 157.8ms | Torch forward with internal preprocessing/NMS; excludes external decode/resize |

These were collected during the research run while other training could be active. They are **not isolated deployed CPU-ONNX benchmarks**, preview FPS, or an assurance of 15fps inspection. The application separately measures preview and inspected-frame rates.

The initial YOLO run requested one Torch thread, but Ultralytics CPU device selection can replace this with its internal `NUM_THREADS`. Actual first-run training threads were not instrumented; do not call it a verified single-thread measurement. The continuation explicitly pins Ultralytics' value and records actual thread counts each epoch.

## Execution records and export integrity

- The initial run completed YOLO fitting and evaluation, then failed before SSD fitting due to an incorrect classifier-head constructor. Its exact executed source is archived as `trainer-executed-initial.py`, with matching original protocol hash.
- SSD used the corrected **SSDLiteClassificationHead**, a fresh process and separate `protocol-before-ssd-retry.json`. The first protocol was retained. Its loss mean decreased from 9.682 over the first 10 updates to 5.596 over the last 10; lower training loss does not establish acceptable recall.
- YOLO training took 259.15 seconds; SSD training took 200.15 seconds. These are training durations, not video latency.
- The YOLO ONNX passed the ONNX checker and has no external weight data. On a fixed TRAIN photograph, native Torch versus CPU-ONNX output shape is `[1,6,2100]`; maximum absolute error is 0.001007 and mean absolute error 0.00001082, within declared tolerance. Both inference engines use one thread for this parity check.
- SSD also passes the checker with no external weights. On the same fixed TRAIN photograph, its 300 output boxes differ by at most 0.0000763 input pixels, labels are identical, and maximum score error is 0.000000298. This establishes export parity on that input, not detection quality or video generalization.
- SSD includes its TorchVision classwise NMS at IoU 0.55 inside ONNX. Serving must preserve this behavior rather than applying a second stricter NMS and reporting the original metrics.

Machine reports are in `output/training/construction-expansion-sdnet-v1/`. Installed research manifests are in `models/construction/expansion-sdnet-v1/`; both carry the source-faithful labels and explicit research status. The dataset and model rights are separate; see the source audit.

The actual local owner fastening recording has a [side-by-side original/research video](../output/evaluation/construction-guided-expansion/owner-fastening-ssd-v1/original-vs-research.mp4). It is a diagnostic playback of real input, not a training source or an accuracy benchmark: no expert frame-level fastening ground truth or expected connection layout is available. Candidate boxes must not be interpreted as verified fastening errors. The public-data photo overlays likewise remain identified as photographs.

## Stronger continuation

`train_sdnet_expansion_v2.py` declares a separate experiment before its first forward: fresh official COCO initialization, 8 fixed epochs, 512px, batch 4, freeze first 4 layers, and one verified Torch thread. Per-category thresholds are frozen from CAL before reading the previously exposed 98 TEST photographs again. Those photographs become **reused development**, not a new independent test.

The continuation preserves all v1 failures and cannot promote a category without genuine independent validation. Its final results, when completed, are stored separately under `output/training/construction-expansion-sdnet-v2/` and summarized below rather than replacing this table.

### Measured CPU performance continuation

The initial 512px fit required approximately 7m14s for its first 151-step epoch, so a TRAIN-only speed probe compared real 512px forward/backward work without optimizer updates or quality-based selection:

| Threads | Actual contiguous layout | Actual channels_last layout |
|---:|---:|---:|
| 1 | 2.625s | 2.396s |
| 2 | 2.230s | 1.711s |
| 4 | 1.467s | 1.226s |

These are microprobe medians, **not full optimizer-step time, video FPS or inference speed**. The selected 4-thread channels_last configuration is 2.142× faster in that probe. All 162 gradient shapes were finite/matching; selected loss difference was 8.37e-7 and relative gradient L2 difference 8.82e-6, with unchanged parameter hashes. Full records: `output/optimization/construction-expansion-training-cpu/results.json`.

The first complete epoch checkpoint was copied, checked across stable reads, and validated before stopping its original process. SHA256: `ec7151f2eddcd192acf193b0912c8073072a1eeb324241f2bb8fafc402e0cece`. Its 162 optimizer states and saved EMA model values are restored exactly, with separate before-resume receipts. Standard Ultralytics resume loads those EMA values as the resumed model; this does not reproduce the uninterrupted raw training model. Unsaved initial-process epoch2 steps were discarded; no outcomes were used for checkpoint selection.

`resume_sdnet_expansion_performance.py` runs **epochs2–8, seven remaining epochs rather than eight additional epochs**, with the same data, resolution, batch, frozen layers, optimizer schedule and CAL threshold policy. It records the thread/layout schedule and possible floating-point trajectory changes. Ultralytics ignores its channels_last option on CPU, so explicit trainer hooks apply and verify the actual model/input layout. The first resumed batch is `[4,3,512,512]`, channels_last, with four Torch threads. Output is separate: `output/training/construction-expansion-sdnet-v2-performance/`.

The probe's speed ratio is not assumed to transfer to the whole fit. Actual resumed epoch times are recorded independently. This training-only optimization does not change the application's one-thread CPU-ONNX inference contract.

**RNG limitation:** Python, NumPy, Torch RNG and DataLoader sampler states are not serialized/restored by this continuation. Reapplying the seed does not reproduce the uninterrupted shuffle or horizontal-flip sequence. Together with standard checkpoint precision/EMA and thread/layout changes, this makes a distinct research candidate with the same eight logical epochs, not an augmentation-identical or numerically identical execution. The unchanged before-resume protocol and a separate `rng-and-standard-resume-note.json` retain this distinction.
# Final extended-fit outcome

The eight-epoch 512px candidate and its actual deployed CPU-ONNX adapter have completed evaluation. See [extended-fit results](construction-expansion-sdnet-v2-results.md) for the frozen calibration cuts, identical reused development photographs, localization and normal-view flags. Unseated-connection recall rises from SSDLite v1's 29.23% to 36.92%, while annotation-unmatched predictions fall from 160 to 79 and flagged publisher Fixed photographs fall from 21/30 to 9/30. This remains inadequate: annotation-relative precision is only 23.30%. Absence appearance remains sparse at 4 matches / 71 annotated instances (5.63% recall), despite no unmatched predictions. No category passes the release gate.

The final raw Torch/CPU-ONNX check passes declared combined absolute/relative tolerances (maximum absolute difference 0.00216675; mean 0.0000152676). The deployed adapter reproduces native evaluation counts exactly. All eight logical epochs completed; the original `config.torch_threads: 1` describes the initial protocol, while the explicit continuation receipts record four actual threads for the remaining seven epochs. Performance-only continuation does not preserve uninterrupted RNG trajectories.

After completion, the exact helper executed by the SSD retry and extended fit was archived as `output/training/construction-expansion-sdnet-v1/trainer-executed-ssd-retry.py`, SHA-256 `3f164be3cab3d9f5e9050a97425b8cbf5b49a17bc018efe82d4d6b4b35452b95`, matching both frozen protocols. The current helper now refuses replay into existing fit/weight directories before writing another protocol; this prevents Ultralytics auto-increment directories from being confused with an earlier checkpoint. This safety guard changes neither trained model nor benchmark outputs.
