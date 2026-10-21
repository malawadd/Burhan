# Factory curvature augmentation: separate fixed experiment

This experiment starts from immutable C200 and changes **training augmentation only**. Its context encoder, packed-pixel refiner, learned weights at initialization, full-resolution loss and half-grid max-logit serving topology are preserved. The proposal is exploratory: the TRAIN orientation audit's 32 horizontal and one diagonal component do **not** measure curvature or prove that training lacks wavy cracks. Exposed-development failures motivated this hypothesis, but no comparison pixels, filenames, IDs, masks, or ROIs enter fitting, mining, or selection.

All evaluation photographs, videos, and calibration inputs remain untouched real observations. Smooth image/mask remapping is an ordinary paired TRAIN augmentation, not additional real defects, synthetic inspection footage, or evaluation evidence. No still images are converted into videos.

## Protected execution

- Prefix: `spatial_round3j_native_curvature`.
- [Protocol recorded before NN/optimizer](C:/Users/IAM/Documents/hackathon/opencv/project1/output/optimization/factory-round3/spatial_round3j_native_curvature-protocol-before-fit.json), SHA-256 `9d64a1b07154f4579f8a6d8bc47493ae9abeadbc35e8d22422d6d108e9ef8420`.
- C200 PT SHA-256 `ff3b5571295e80978b4be873b6cce24b7cb947ae25415959a4121337dfe22112`; ONNX `ca9b90e33f1cacc8cd83ccc8e406b32faa8f45947fd2d672dc48e678b20d7318`; source freeze `c3c2a6d86f86d625f9f6499c4d7477a25c2b9610b26a870be994a29ac969db40`. Source metadata, implementation files, and reused architecture receipt are also pinned in the protocol.
- Seed `20261027`; fixed **600 steps**, all **200/400/600** checkpoints mandatory; batch one, one CPU thread; no wall-time stop or adaptive schedule. The complete C state loads strictly: learned context/refiner weights are retained, with no head reset.
- Native 640H×512W crops, no resizing or 90° transposition. Five-step cycle: SDD1 positive, positive, TRAIN-mined hard, unbiased, SDD2 inner-FIT rehearsal. Rehearsal alternates positive and actual normal. Exactly 20% of batches use the already frozen SDD2 inner-FIT partition.
- Same existing optimizer, sensitivity loss, frozen batch-normalization statistics and clipping: encoder/context LR .00005, decoder .0001, refiner .0003; AdamW weight decay .0001; gradient norm ≤5.

The [trainer](C:/Users/IAM/Documents/hackathon/opencv/project1/scripts/train_factory_round3j.py) verifies C source hashes, all 239 SDD1 image/mask identities against the geometry fixtures, and each consumed SDD2 image/mask against the immutable split on cache miss. It refuses to restart over existing checkpoints/history. No reserved SDD2 images or original comparison images are loaded.

## Remapping contract

The [sampler](C:/Users/IAM/Documents/hackathon/opencv/project1/scripts/factory_curvature_sampling.py) uses a .5 Bernoulli probability for every role, including actual normals. For requested curves:

`source_x = destination_x`

`source_y = baseY + destination_y + A*sin(2*pi*x/P + phi)`

Amplitude is uniformly selected from {8,16} input pixels, period from {128,256,512}, phase uniformly from [0,2π). Integer `baseY` lies in [16,624], ensuring every remapped pixel lies within the observed 1280×512 source. Images use linear interpolation; actual binary occupancy uses nearest interpolation. No border pixels are invented. Hard/unbiased targets include **all** real occupancy sampled by the full remap, not just an ROI or chosen component.

For SDD1 positives, the entire selected actual connected component's transformed `y−f(x)` bounds constrain placement, with a one-pixel rounding safety margin. If curved placement is infeasible, the declared identity fallback retains that component. SDD2 positives retain the sampled actual positive pixel; very large SDD2 components are not promised whole-component containment. The fallback does not relabel absent damage or create borders. Paired 0/180° turns/horizontal flips and existing contrast/brightness/mild blur follow the geometric operation.

Each actual training sample records its source/role, requested/applied curve, top, source-coordinate range and fallback in `spatial_round3j_native_curvature-augmentation-history.jsonl`. These are augmentation receipts, not inspection findings.

## Pre-fit verification

[Hardware-free pairing fixtures](C:/Users/IAM/Documents/hackathon/opencv/project1/scripts/check_factory_curvature_sampling.py) passed before protocol recording and before any model forward:

- 127 coordinate/boundary/fallback/invalid-geometry checks, including rejection of an impossible identity crop.
- 200 complete coordinate-fixture role batches: 100 positive, 40 hard, 40 unbiased, 20 empty-GT normal-role fixtures. Curvature applied respectively 51,21,18,10 times; normal-role fixtures retain empty GT. This verifies role-independent application rather than a positive-only augmentation shortcut; these are software fixtures, not additional real dataset observations.
- 396 paired remaps across **all 33 actual TRAIN components in 239 source views**, covering both amplitudes, every period, and two fixed phases. Selected GT occupancy survived, pixel pairing matched OpenCV's actual nearest source indices, and every sampled coordinate stayed in bounds.
- Fixture SHA-256 `9805e7f858c465e4c081853953a1c203f8ce5ba297900dbfc68e3c96882396c2`.

Synthetic coordinate ramps used by these software fixtures are never saved or evaluated as defect examples. Actual source decoding was restricted to original TRAIN masks; no NN was used by the fixtures.

All ten [freezer guard tests](C:/Users/IAM/Documents/hackathon/opencv/project1/tests/test_factory_curvature_freeze.py) passed in .028s: all-three success, early/duplicate checkpoint rejection, changed source identity despite recomputed digest, duplicate CAL image, stale source digest, the exact preexisting original-CAL source pin, and strict search-policy equality despite differing observed CPU contention. These checks do not alter the fit/protocol or search rules. Each metadata execution scope must still equal its own actual report.

## Calibration, freezing, and speed

The earlier F2 experiment finished, froze/rejected its own all-nine result, and archived its executed code before the separate `--curvature-augmentation` flag was added to [the calibrator](C:/Users/IAM/Documents/hackathon/opencv/project1/scripts/calibrate_factory_round3.py). Existing experiment flags retain their prefixes and semantics.

```powershell
.venv-roads-export\Scripts\python.exe -X utf8 scripts/train_factory_round3j.py --contended-training
.venv\Scripts\python.exe -X utf8 scripts/calibrate_factory_round3.py --variant context --attempt3 --phase-max-refinement --curvature-augmentation --steps 200 400 600 --contended-calibration --cpu-threads 1 --contention-note "Concurrent fixed factory CPU1 and construction CPU4 training"
.venv\Scripts\python.exe -X utf8 scripts/freeze_factory_round3j.py
```

GPU calibrations require explicit scheduling; no competing inference is permitted. Cached maps bind the exact checkpoint and ordered original 80 calibration image/mask hashes/native geometry/class. The original threshold/area/growth grid, normal FP≤1/70, foreground precision≥.65 and weakest-positive ranking stay fixed. The top-six seed shortlist remains a heuristic search, not an exhaustive optimum. [The new freezer](C:/Users/IAM/Documents/hackathon/opencv/project1/scripts/freeze_factory_round3j.py) requires completed fixed training and all three actual calibrations before any winner, and binds their source identity to the preexisting original-CAL digest `7b09d8cbf39129dad144fcad6ecbad0ade0bb19e39de6e8fbdfc634eb6ed6d07`. The training owner does not score original comparison images.

The prior C architecture's measured 19.66 complete calls/s is an **architecture receipt only**. This training run overlaps other approved CPU accuracy jobs, so its elapsed time and calibration latency are not speed benchmarks. Final new-weight/policy capacity and video runtime must be measured after **all fits finish**, with the ≥15 FPS requirement preserved. No current accuracy improvement, production readiness, or independent-test result is claimed.

## Completed training receipt

The fixed fit completed all 600 steps without restart or parameter/schedule change, in 609.92 seconds during approved CPU contention. That duration is not an inference-speed benchmark. Actual roles were 300 positive,120 hard,120 unbiased,60 normal. There were 308 requested curves,306 applied curves and two declared observed-geometry identity fallbacks. The [per-sample receipt](C:/Users/IAM/Documents/hackathon/opencv/project1/output/optimization/factory-round3/spatial_round3j_native_curvature-augmentation-history.jsonl) has SHA-256 `57678a97463d26e66d87880159904427fae14b62395d0274e82727913bb242bf`.

The [initial executed-source snapshot](C:/Users/IAM/Documents/hackathon/opencv/project1/output/optimization/factory-round3/curvature-executed-source-snapshot-v1/snapshot-manifest.json) preserves 14 source files, SHA-256 `1f84fcf75a6a6b13d0e6b2382deef6a778be6a5f1abf05789955950dce58140c`. A separate [freezer/test snapshot](C:/Users/IAM/Documents/hackathon/opencv/project1/output/optimization/factory-round3/curvature-freezer-source-v2/snapshot-manifest.json), SHA-256 `aa3a6cac1cc4de32790b514e3a0384408a2f340dfd380ba78dfbbb3ca2c41059`, records pre-freeze handling of different actual CPU workloads. It preserves identical fixed search fields while retaining each checkpoint's execution note; original training/protocol/source snapshot remain unchanged. Checkpoints 200/400 calibrated while factoryCPU1 and constructionCPU4 fits overlapped. Checkpoint600 calibrated after the factory fit exited, with constructionCPU4 plus metadata/source preparation. Every calibration is explicitly contended accuracy-only, with no capacity qualification.

## All-three calibration and frozen winner

All 200/400/600 checkpoints were evaluated on the same original80 CAL views with exact source identity `7b09d8cbf39129dad144fcad6ecbad0ade0bb19e39de6e8fbdfc634eb6ed6d07`. Failures are retained. The original C200 row is its already frozen calibration result, provided for context rather than a new comparison-image audit.

| Model/checkpoint | Meaningful recall (mask IoU≥.1) | Normal alarms | Pixel precision | Pixel recall | Micro pixel IoU | Worst positive IoU | High / area / growth |
|---|---:|---:|---:|---:|---:|---:|---|
| Original frozen C200 | 10/10 | 0/70 | .66901 | .86983 | .60813 | .30239 | .999 /1 /.8 |
| Curvature200 | 10/10 | 0/70 | .66051 | .85298 | .59300 | .16121 | .99 /1 /.8 |
| Curvature400 | 9/10 | 1/70 | .81747 | .71869 | .61931 | 0 | .995 /32 /.98 |
| Curvature600 | 9/10 | 0/70 | .82492 | .73535 | .63608 | 0 | .995 /1 /.95 |

The fixed weakest-positive rank selected **curvature200**, only after all three completed and fixed600 training exited. This does not establish improvement over C: its calibration pixel metrics and weakest positive localization are worse than C, despite the same10/10 meaningful count. A separate frozen original-development audit is pending with the coordinator; this training task did not read or score those inputs.

- [Global freeze](C:/Users/IAM/Documents/hackathon/opencv/project1/output/optimization/factory-round3/winner-curvature-augmentation-freeze-before-comparison.json), SHA-256 `863441430b420175f114b6b4c4d47f9fcc66eeaef5b42e6436c32fb67560cef3`.
- Winner ONNX SHA-256 `7da051dabc4eefd024fbc7b0a1482b040e00087252a33a278cc086f42dbe7e81`.
- Winner metadata SHA-256 `118f69cac35bf96add6d6b937fac03620d84bc44978c710a2eca1b4a98ff3505`.
- [All-three raw/grid/report data](C:/Users/IAM/Documents/hackathon/opencv/project1/output/optimization/factory-round3/spatial_round3j_native_curvature-calibration.json) and each actual GPU score cache are preserved.

For completeness, observed calibration `predict` API p50/p95 latencies were68.57/87.67ms (200),66.08/82.41ms (400),62.16/68.35ms (600). These contended cache-extraction calls used the calibrator's common `.999/16` extraction policy, not each final frozen policy, and are **not** isolated capacity, camera FPS, or final new-weight runtime benchmarks. Final timing was pending at this stage and was later completed separately below. All experiment processes exited and CPU/GPU were explicitly released after the no-NN freeze.

## Independent original-development audit: failed improvement

After freeze, the coordinator independently evaluated **only the globally selected200 winner**, retaining every defective view and false alarm in [the audit and evidence](C:/Users/IAM/Documents/hackathon/opencv/project1/output/optimization/factory-round3j/after-localization-audit.json). This review/training task did not open those image/mask pixels or score alternative checkpoints.

| Frozen-winner original-development result | Actual value |
|---|---:|
| Meaningful mask localization (IoU≥.1) | 10/11 |
| At least50% GT pixel coverage | 8/11 |
| Normal false alarms | 2/69 |
| Pixel precision / recall | .6139785 /.7158977 |
| Micro pixel IoU | .4936884 |
| Mask IoU≥.5 | 3/11 |
| Runtime box IoU≥.5 | 8/11 |
| Abstentions | 0 |

All11 defective views and both false alarms were saved. The attempt does not improve the known C baseline's10/11 meaningful localization with one false alarm. **No adoption and no fresh official SDD2 test opening follow.** The remaining defect miss is retained rather than rescoring losing checkpoints.

## Completed project-quiescent timing: capacity passes, accuracy still fails

After all fits/calibration/probes exited and the station stopped, the coordinator ran the already declared paired plans once, sequentially. Every first tail/failure is preserved. Inputs are predecoded actual public CAL stills; repetitions are timing observations rather than new accuracy samples.

| Paired240 complete-profile CAL calls/model | Baseline | Frozen curvature200 |
|---|---:|---:|
| Capacity | 27.8288 calls/s | **21.1491 calls/s** |
| Candidate mean/P95/P99/max | — | 47.2833/56.2139/59.7047/64.1620ms |

| Sequential60s CAL replay at16inputFPS | Baseline | Frozen curvature200 |
|---|---:|---:|
| Captured /processed | 960 /959 | 960 /960 |
| Drop /late /error | 1 /8 /0 | 0 /0 /0 |
| Decision P50/P95/P99/max | 30.8/66.0/85.7/156.6ms | 44.2/55.8/67.4/88.2ms |

Late means decision age above100ms. Source skips and inspection errors are zero. These are software replays of real photographs, **not moving production-line footage or proof of physical-item coverage**. Capacity excludes disk decode/capture/queues/UI/evidence/actuation; replay measures its declared software queue/decision pipeline. The measured≥15 target passes here but does not repair10/11 original localization or two false alarms.

The project was quiet; the whole host was not exclusive. Read-only snapshots retained an unrelated Python process26076 at102.5/105.5% of one logical core and normal desktop activity. No unrelated process was controlled. The first baseline's actual drop/eight late counts and queue tail remain visible rather than replaced by its different density-paired run. [Coordinator method and trace interpretation](C:/Users/IAM/Documents/hackathon/opencv/project1/docs/inspection-improvement-round3.md:368) and [timing artifacts](C:/Users/IAM/Documents/hackathon/opencv/project1/output/optimization/factory-round3j-capacity/paired-calibration-capacity.json) retain the evidence. The model remains rejected.
