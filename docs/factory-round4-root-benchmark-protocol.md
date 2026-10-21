# Round4 root benchmark protocol

Status: **source prepared and independently reviewed; no benchmark plan or NN run yet.** The fixed4096+900 fit is still running. This document does not claim new speed or accuracy. Root executes only after the actual all-three/two-arm global freeze exists and grants the required compute window.

New helper: `scripts/benchmark_factory_round4_root.py`. Five pure fixtures pass; they exercise timing outliers, missing/duplicate sequences, all runtime counters, fixed64 source/stage evidence and actual GPU recipe checks. They load no images, model or weights. The neural owner independently reviewed the source and requested protected split/CAL identity and finite stage-timing checks, which are present.

## Frozen eligibility

The helper requires exact round4 protocol SHA `89adfa6e24de1c4dd3b2fbe13c2644b3948dc521cec520e840ceadf9ab7d1a2e`, completed4096 transfer +900 fine updates, all300/600/900 terminal outcomes, and both prospective arm winners frozen before any original scoring. It verifies model/sidecar/policy hashes, actual numeric/CAL/cache/terminal lineage and executed-source archive bytes. Live historical prep/test files are not transitively reverified: their executed archived bytes remain evidence. Actual serving and benchmark source bytes are separately pinned for the new measurements.

Primary keeps CAL normal FP≤1/70; secondary keeps≤3/70; pixel precision≥.65 and the original fixed search/rank remain unchanged. Only the global winner of each nonempty arm may be benchmarked. No losing checkpoint, alternate threshold, ROI, input resolution, compiler setting or backup policy may be substituted. If neither arm has a winner, recording the plan says `NO_ELIGIBLE_ROUND4_WINNER_NO_BENCHMARK`, and execution stops. If an arm fails capacity, retain its failure; it cannot earn original-accuracy or real-time qualification. Both arms' outcomes must be reported.

Comparators are the exact retained old `spatial_unet_v10h_sdd2-step300.onnx` and C200 `spatial_round3e_phase_max_refine-step200.onnx`, with original sidecar hashes/policies. C200 remains the prior10/11 meaningful-localization,1/69 normal-alarm development result. Neither that result nor prior speed is transferred to new weights.

## Root quiet-window prerequisites

Finish **all** project training, exports, model inference, data hashing/decoding, media encoding and tests. Stop the local station and pause browser video playback. Capture a workload snapshot before the measurements; ordinary host services stay uncontrolled and must be disclosed. No process suspension, termination or watchdog is used. If contention occurs, retain that observed result; do not silently retry a favorable timing window.

The root-only read-only snapshot helper is `scripts/capture_round4_benchmark_host.py` (run once after those prerequisites). It records one-second observed CPU/process-name workload, memory and runtime versions, checks the station is stopped and rejects visible project script/FFmpeg workers. Root also checks agent completion and paused browser playback; this process scan is not a claim that unrelated host workload is controlled. It never records process command lines or stops other applications.

Preparation hashes metadata/artifacts only and may precede the quiet window. Every execution stage requires `--root-quiet-release`; original scoring additionally requires `--root-original-release`. These are internal coordination gates for root, not a new user permission request.

All commands below run from the project directory using existing `.venv` (OpenVINO runtime; no Torch needed). They are **instructions, not commands executed by this document**.

```powershell
.venv\Scripts\python.exe -X utf8 scripts/benchmark_factory_round4_root.py --stage prepare
```

This creates `output/optimization/factory-round4-root-benchmark/plan-before-any-benchmark.json` only when the future freeze exists. Review its exact models, policies, source pins and eligible-arm list before release. No existing artifact is overwritten.

## 1. First64 complete-profile fitted capacity

For each eligible learned arm, in primary→secondary order, run only the commands for present arms:

```powershell
.venv\Scripts\python.exe -X utf8 scripts/benchmark_factory_round4_root.py --stage capacity --name primary --root-quiet-release
.venv\Scripts\python.exe -X utf8 scripts/benchmark_factory_round4_root.py --stage capacity --name secondary --root-quiet-release
```

The exact16 real normal SDD1 TRAIN images/hashes are inherited as a source recipe from immutable protocol `522a1f549ca08e964fa46c2787f32370da0b4787dba3fe1ab3809efc1f87b32c`. No source is chosen from model results. Images are decoded before timing. The ordinary unchanged `FactoryProfile` is used throughout, with its actual native quality checks, preprocessing, GPU NN/readback, sigmoid/components/hysteresis, findings and annotated-frame construction. This is **not NN-only timing**.

Each fresh profile performs the existing3 zero initialization calls, then4 specified real TRAIN warmups. After reset, the **first64** measured calls cycle all16 real images four times. All64 must actually infer, with finite/nonnegative stage evidence and the frozen source order. All timings, including the slowest call, remain in the receipt. Gate: `1000 / mean(full profile ms) ≥15`. Report mean/P50/P95/P99/max, actual queried GPU/f16/one-stream/LATENCY/default-Winograd properties and initialization separately. No cropped inputs or zero-only speed test is credited.

The helper intentionally requires capacity outcomes for all eligible arms before beginning downstream workload. Failed capacity arms are skipped downstream and remain failed. If both fail, stop.

Optional identical first64 measurements for comparator context use `--name v10h` and `--name C200`; they do not qualify a failed new arm.

## 2. Root original80 localization audit

Only after the arm's selected fitted first64 gate passes, root may score that exact arm once:

```powershell
.venv\Scripts\python.exe -X utf8 scripts/benchmark_factory_round4_root.py --stage original --name primary --root-quiet-release --root-original-release
.venv\Scripts\python.exe -X utf8 scripts/benchmark_factory_round4_root.py --stage original --name secondary --root-quiet-release --root-original-release
```

The helper calls the existing `scripts/factory_localization_audit.py --partition test`, with a fresh output and `--visuals-dir`. All11 defective views, including misses, and all flagged normal views get RAW / publisher GT / model mask / difference panels plus exact native masks. Primary outputs are view maskIoU≥.1 meaningful localization, half-GT coverage, pixel precision/recall/IoU, one-to-one region IoU.1/.25/.5 and normal-view alarms; actual runtime-box IoU is supplementary. A one-pixel hit does not replace meaningful localization. All abstentions remain explicit misses rather than disappearing from denominators.

The80 views/10 items are **repeatedly exposed development data**, not a new blind test. The requested recall improvement beyond10/11 means11/11 on this small development comparison; it still does not establish production reliability. Report normal alarms against both old v10h and C200, and report pixel localization quality too. A secondary arm's improvement over old5/69 alarms must not be described as improvement over C200's1/69. Do not select or tune an arm from these outcomes. SDD2 reserved466 and official TEST remain sealed.

Fresh comparator localization, if root wants current-source paired evidence, uses the same command with `--name v10h` and `--name C200`. Historical comparator audit files may instead be cited with their exact source/method limitations; no new model inference is required merely to recreate already available evidence.

## 3. Paired full-profile speed

```powershell
.venv\Scripts\python.exe -X utf8 scripts/benchmark_factory_round4_root.py --stage paired --root-quiet-release
```

The group is old v10h, C200, and each capacity-passing learned winner. Each processes the identical predecoded protected CAL80 frames three times (240 measured calls/model). Per-view model order rotates by `(view index + repetition) % model count`, distributing order/cache effects. Each gets the same3 initialization +4 real TRAIN warmups. There are no extra favorable-case model calls, timing exclusions or selected image saves. All per-call total/stage times and statuses are retained; only full-profile timing is compared. Repeated CAL observations add timing samples, not new accuracy evidence.

This new group helper avoids `benchmark_factory_speed.py`, whose old RF/sklearn backend and exposed-test workload are unsuitable here. It also avoids the extra untimed illustrative inference in `benchmark_factory_accuracy.py`. Model/profile arithmetic remains the ordinary runtime implementation.

## 4. Actual16FPS runtime,60seconds per model

Execute sequentially in fixed order v10h→C200→primary→secondary, omitting absent or capacity-failed arms:

```powershell
.venv\Scripts\python.exe -X utf8 scripts/benchmark_factory_round4_root.py --stage runtime --name v10h --root-quiet-release
.venv\Scripts\python.exe -X utf8 scripts/benchmark_factory_round4_root.py --stage runtime --name C200 --root-quiet-release
.venv\Scripts\python.exe -X utf8 scripts/benchmark_factory_round4_root.py --stage runtime --name primary --root-quiet-release
.venv\Scripts\python.exe -X utf8 scripts/benchmark_factory_round4_root.py --stage runtime --name secondary --root-quiet-release
```

Each delegates unchanged to `benchmark_runtime.py --backend auto --varied --partition calibration --fps16 --seconds60 --deadline100 --preview-fps6 --trace`. The existing source clock, latest-frame buffer, worker, preview throttling, event publication and storage path are retained. It cycles all80 real CAL stills; this is an engineering software-clock replay, **not a production-line video or physical acquisition/coverage claim**. Source decode is pre-timing; camera/USB acquisition cost is unmeasured.

Expected actual captured/submitted and processed count is960. The strict unchanged runtime checks require exact trace sequences1…960 once each, zero drops, source skips, inspection errors, coverage gaps and deadline-late decisions, plus clean EOF. A959-frame run is retained as959 and fails that check. All counters and failures are reported; no rounding, shortened source clock or reset removes a miss. The receipt also records all960 diagnostic profile and host-to-profile-return latencies, mean/P50/P95/P99/max, and every trace row over100ms. Official runtime decision latency additionally includes publication and remains in the raw runtime report; do not replace it with NN or profile-only latency.

A passed first64 mean is only a capacity prerequisite. It does not excuse later drops, deadline tails, accuracy failures, bad localization or normal alarms. No adoption follows solely from speed.

## Outputs and practical budget

All new files stay under `output/optimization/factory-round4-root-benchmark`. First-attempt markers block overwrite/retry. Model/metadata/policy/source pins are checked again after each stage. Preserve raw reports, exact commands, all runtime counters/tails and every original localized/missed example.

At most two learned arms plus two historical comparators are involved. The four required60second runtime blocks cost four minutes; first64 and paired profile runs add roughly a minute on a qualifying graph, plus initialization and accuracy/evidence writes. This is a bounded sequence, not a threshold/model search. If no model qualifies, stop rather than benchmarking alternative checkpoints.

## Pre-benchmark historical CAL identity correction

Before any benchmark plan or measurement, the factory evaluator stopped before CAL model initialization because its compact JSON serializer did not reproduce the protected historical CAL80 digest. That digest uses `json.dumps(rows, sort_keys=True)` with default spaces. The benchmark preparation had the same mistake; its dedicated `historical_cal_identity()` now uses the exact historical serialization. Source order, path strings, image/mask SHAs, native geometry, protected digest, policies and eligibility guards remain unchanged.

The original unexecuted benchmark helper, tests, document and five-fixture receipt are preserved under `source-before-CAL-identity-recovery/`. Additive six-fixture receipts preserve the correction and final peer hardening separately; the final preparation receipt is `source-preparation-fixtures-v3-CAL-identity-peer-closed.json`. An independent metadata-only proof confirms that the exact original 80 rows/10 positives reproduce `7b09d8cbf39129dad144fcad6ecbad0ade0bb19e39de6e8fbdfc634eb6ed6d07`. No images, masks or NN were opened for that proof.

Future benchmark preparation additionally binds `CAL-identity-v2-correction-before-CAL.json`, root's recovery decision and exact wrapper SHA, all three preserved numeric report SHAs, the failed pre-CAL source/log archive, and the recovery wrapper's executed source archive. Both the final global freeze and each selected actual CAL report must carry that exact receipt and wrapper identity. Archived preparation/test bytes are verified without demanding those historical tests remain live forever. The actual runtime and benchmark source pins remain protected during execution. No benchmark plan, capacity measurement or original comparison has been executed by this correction.
