# Factory CPU optimization

The supplied KolektorSDD surface model uses a fixed 256-pixel-wide feature grid, 32 trained decision trees, a pixel threshold of 0.995 and an eight-grid-pixel component minimum. The previous implementation evaluated every tree at every grid pixel. Most normal pixels cannot reach that high threshold after a single low tree vote.

## Exact candidate decisions through early rejection

`auto`, the default backend, selects the explicitly built local `native_cascade` when its source and artifact checksums match. Otherwise it selects `threshold_cascade`, which uses compiled scikit-learn tree traversal. The actual backend, requested backend, native-code hash and fallback reason are visible in `describe()`. Neither path compiles or downloads code during startup.

If `S` is the accumulated tree vote, `R` is the number of trees left and `T` is the frozen pixel threshold times the number of trees, the pixel can be rejected when `S + R < T`. Every remaining tree contributes at most one, so no skipped vote can rescue that pixel. Both implementations use the original tree order. A conservative floating-point margin of `1e-12` prevents rejection at a numerically ambiguous boundary.

The native implementation is a reviewable, dependency-free C loop in `edge_runtime/native/factory_cascade.c`. It traverses the same original nodes/float64 thresholds and accumulates the same leaf values. One `ctypes` call replaces Python calls for every tree, active-index allocations and feature-submatrix copies. `ctypes.CDLL` releases the Python GIL during the loop. The wrapper enforces contiguous float32 feature matrices, dimensions and bounded row counts. Only the trusted locally trained model is unpacked into native arrays; this does not make untrusted joblib files safe.

Model-mode quality evaluation also stores the original uint8, ksize-1 Laplacian in int16 instead of float64. Its integer range is [-1020, 1020], so values are represented exactly. Native double-precision `cv2.meanStdDev` reductions replace large NumPy float64 variance intermediates. An input within `1e-6` of a three-decimal rounding boundary uses the original reduction to avoid changing a quality decision through floating-point summation order. The original baseline backend retains the original quality implementation; noise, constant and gradient tests compare rounded quality metrics. Float32 feature normalization and variance operations reuse temporary arrays, and `cv2.merge` interleaves the unchanged feature maps without NumPy's repeated strided writes. Tests and the held-out comparison check bitwise feature parity.

This changes execution only. There is no quantization, downsampling, smaller forest, changed feature, changed learned parameter, threshold selection or test-set fitting. For retained candidate pixels every tree is evaluated and the exact mean vote is used for the region score. Connected-component filtering, original-image boxes, quality/view guards, temporal confirmation and unit events are unchanged.

The returned score map is a **lower bound at rejected pixels**. It must not be treated as the old exact complete score map or used to compare pixel-ranking AUROC. Metrics explicitly expose `model_score_map_exact=false`, `model_score_semantics=exact_candidate_mean_rejected_pixel_lower_bound` and `model_score_max_lower_bound`. Candidate finding scores are still exact mean tree leaf defect fractions, not calibrated probabilities. Select `defect_backend=sklearn` when a complete exact score map is needed.

## Reproduction

```powershell
.venv/Scripts/python.exe scripts/build_factory_native.py
.venv/Scripts/python.exe scripts/benchmark_factory_speed.py --verify-pixels
.venv/Scripts/python.exe -m pytest tests/test_factory_surface.py tests/test_factory.py -q
```

Clang 13.0.1 was already installed on this laptop at `C:/Program Files/LLVM/bin/clang.exe`. The build script uses Clang and LLD on Windows and does not need MSVC runtime libraries; it uses no network or downloaded compiler. The generated x86-64 DLL is 3,072 bytes; `llvm-readobj --coff-imports` confirmed it has no import dependencies. Stop processes using the DLL before rebuilding it on Windows. The artifact and its metadata live under ignored `models/runtime/`, rather than treating generated machine code as source. On a machine without a compiler, the visible Python/C scikit-learn cascade remains available. Explicitly requesting `native_cascade` fails if the local artifact is absent, stale or corrupt; `auto` records a fallback reason.

The script predecodes the identical frozen 80 test images from the existing ten held-out physical items. Both backends receive two excluded warmups. Before/after order alternates by image to reduce order-related CPU load bias. Timed work is complete `FactoryProfile.process`, including original-resolution input quality tests, model features/scoring, regions, temporal/unit handling and annotation. File read/decode, capture, host queues, browser preview, disk events and actuation are excluded. Those need separate runtime measurements.

Each output row records the image SHA-256, path, ground-truth label, per-backend status, original-coordinate boxes, candidate scores, method metrics and timing. `--verify-pixels` additionally compares every pixel's threshold decision and each positive pixel's score outside the timed interval. The model and metadata hashes, exact split IDs, thresholds, software versions and scope are recorded in `output/evaluation/factory-speed-comparison.json`. Previous evaluation reports are preserved; this benchmark does not overwrite them.

## Intermediate Python/C cascade measurement

The first paired 80-image run, before the native loop and quality reduction, is preserved at `output/evaluation/factory-python-cascade-comparison.json`. It measured full-profile p50/p95 **353.85/515.02 ms before** and **73.45/100.96 ms after**, a 5.10× p95 speedup. Mean serial capacity increased from 2.76 to 13.51 observations/s. This intermediate implementation preserved every thresholded pixel decision, region box and final image status across all 80 images. Positive-score differences were at most `1.11e-16`, from floating-point addition. These figures do not describe the newer native implementation; its separate final comparison is authoritative.

## Final paired native measurement

The final run used the laptop CPU (Intel i7-8550U), Python 3.13.5, OpenCV 5.0.0 with two OpenCV threads, the original two-worker sklearn forest baseline, and the single native C loop. Other agents' inference benchmarks were paused during timing. Both implementations used the same unchanged model SHA-256 `549d8528c1e2b8ca483b772e9ada4f71284782dca2cd631be7127171cadd0473`; native code SHA-256 was `071a90db6cb580a89713970cbcd358c3850f2090c8e476cbc4eb7426eea1d6a3`. See `output/evaluation/factory-speed-comparison.json` for every measured row.

| Complete profile metric | Original sklearn | Optimized native |
|---|---:|---:|
| p50 | 315.52 ms | 35.02 ms |
| p95 | 479.99 ms | 53.27 ms |
| p99 | 510.72 ms | 69.24 ms |
| Maximum | 548.28 ms | 88.99 ms |
| Mean | 341.21 ms | 35.48 ms |
| Unpaced serial capacity, 1000/mean ms | 2.93 observations/s | 28.18 observations/s |
| Accepted / abstained images | 80 / 0 | 80 / 0 |
| TP / FN / FP / TN | 8 / 3 / 1 / 68 | 8 / 3 / 1 / 68 |
| Precision / recall / F1 | 88.9% / 72.7% / 0.800 | 88.9% / 72.7% / 0.800 |

The final p95 speedup is **9.01×**. All 80 images had **zero feature-value disagreements, zero rounded quality-metric disagreements, zero thresholded-pixel disagreements, zero box/status disagreements and zero candidate-positive score difference**. Threshold, minimum region area, resolution, forest and test split were never tuned. Native/backend regression checks plus reference-mode checks passed **25 tests in 12.90 seconds**. The unchanged model still missed three annotated defective views.

The original 1 FPS preset was a conservative input setting, rather than a measured maximum capacity. The final mean capacity of 28.18 observations/s must not be described as a guaranteed 28 FPS camera system or 30 Hz production line: p95 is 53.27 ms, which exceeds a 33.33 ms period, and some observed frames took 88.99 ms. A proposed starting point is **15 input observations/s with a 100 ms host deadline**, subject to the complete runtime's separate drop/late/trigger and representative-scene stress measurements. Use that runtime evidence, rather than unpaced process capacity, to choose the final laptop preset.

A separate short diagnostic repeated the supported held-out image `kos04/Part3.jpg` ten times and excluded the first two iterations. Median stage times were **2.23 ms input quality, 12.54 ms features, and 8.05 ms native tree scoring**. This probe excludes connected components, annotations and all runtime work; it is useful for locating remaining cost, not replacing the varied 80-image measurements. Its warm repeated-still cache behavior can differ from the paired alternating full-profile benchmark.

The completed [shared runtime study](performance-study.md) supersedes the proposed 100 ms preset above. At 15 FPS over 60 seconds, all 900 varied held-out frames were processed without drops, but the strict 100 ms run had 11 late decisions. The selected **15 FPS / 200 ms** run had zero drops/late/errors, host p95 **51.5 ms** and max **106.9 ms**. Public presets use 200 ms explicitly; neither the initial 30 FPS target nor a strict 100 ms guarantee was achieved.

## Accuracy and deployment limits

The 80-image holdout has only eleven annotated defective views. Its previous 8 TP / 3 FN / 1 FP / 68 TN result and appearance restrictions remain material. Faster execution cannot recover defects missed by the unchanged model. A normal view cannot certify all views or functional health of a physical item. Retain the dataset's non-commercial research license restrictions and the [model provenance](models/factory-surface.md).

Process capacity is not a camera exposure-to-actuation guarantee. A production-line claim requires a matching trained product family, representative independent accuracy data, known camera shutter/transport delay, motion/illumination control, physical trigger association, deadline monitoring, and a measured reject path. An 80-still-image CPU benchmark cannot establish sustained thermal stability or every-item coverage. Configured source FPS must be chosen against the complete runtime's measured tail latency and overflow/missed-unit counts.
