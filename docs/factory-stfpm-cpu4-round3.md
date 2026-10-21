# L5: one CPU FP32 / four-thread backend

This separately named experiment changes the inference backend only. It uses the **exact same L4 BN-fitted ONNX bytes, saved full state, and three numeric reference inputs/expected score maps**. It does not repeat the64-view BN fitting pass, modify the model graph, choose new sources or change any score/localization threshold. At preparation time, no L5 CPU inference or capacity measurement has run.

L4's three-input GPU numeric check failed with maximum errors .08950025/.04305449/.04961184 above the fixed .01 guard. Before editing any shared runtime hook, its executed source/runtime snapshot was saved at `output/optimization/factory-round3l4-bn64/failed-L4-executed-source-snapshot`, manifest SHA `01a12c2181f95bfb0be9e666e514f1d3c5d897bc8fb1ada5e821e5100ecf9e13`. L1/L2/L3/L4 files and failed results remain unchanged.

## One fixed backend and preserved defaults

The one L5 configuration is OpenVINO **CPU**, inference precision **f32**, `PERFORMANCE_HINT=LATENCY`, **one stream**, and `INFERENCE_NUM_THREADS=4`. There is no automatic fallback, thread sweep, GPU optimization flag, resized input or precision search. The native input remains1×3×1280×512, with output1×1×640×256 and the original bounded multiscale disagreement semantics, not defect probability.

`FactorySegmentation` has a strict metadata-controlled opt-in. CPU requires f32 and exactly four integer threads, and rejects GPU-only no-Winograd configuration. Existing models without device/thread fields retain the identical GPU/f16 compile configuration. Existing GPU metric labels, initialization report fields and scheduling caveat remain unchanged; CPU receives the actual `openvino_cpu_neural` backend/device/threads labels and CPU warmup caveat. The older separate GPU feature encoder and RF model paths reject explicit device/thread metadata rather than silently ignoring it.

The CPU probe queries only CPU-supported configuration categories, including actual execution devices, inference threads, stream count and precision; it does not request GPU properties. Requested metadata and queried compiled/runtime precision provenance are recorded separately.

## Exact state, model and reference identity

| Retained L4 asset | SHA-256 |
|---|---|
| BN-fitted full state | `3a8030a7e50d0e2d1c7795bc6cad1139d5d9427aa6020ca09347cbaf096a4207` |
| BN fitting receipt | `dba5c934745c030417006d58927f531112e0250ccd0c31fe228bc553e02bcf00` |
| Initial ONNX, copied byte-exact into a new L5 name | `3eb5f1cd0936342098658e4a808176e4b6bdb8bc1e2103dc3873e7542ed07ace` |
| Three-input Torch reference NPZ, copied byte-exact | `5e870bccd9243f1835f2bc2d68e666110bfe4d4903ddcbc605bd577b19e08409` |

The initial CPU preparation helper has no BN-fitting stage and cannot re-export step0. It only copies these fixed bytes and writes a distinct CPU sidecar. The original64 BN source identity, finite state, all18 counters64, preserved epsilon.001, restored momentum.01 and full parameter/teacher invariants stay bound to the saved L4 receipt.

## Gates and possible future training

Before any CPU inference, a new immutable protocol/source snapshot, mock fixtures and independent static review are required. The fixed numeric check uses first real normal TRAIN, zero and all255 initialization cases; all three must be finite/bounded/nonconstant where declared and have maximum error≤.01. Zero/white tensors are numerical initialization cases, never BN training data or inspection footage.

Only a new root-released quiet window can run the16-source/64-complete-profile capacity check. All64 must infer, mean throughput must be≥15 calls/s, and all first-run tails/startup/ordinary-host limitations remain retained. Previous GPU speed or CPU numeric agreement on a different BN state cannot transfer. Failure stops this single configuration before optimizer training.

All possible future L5 training/calibration/freezer files are separately versioned. The trainer loads the same saved BN-fitted full state before AdamW, restores student BN momentum.01 and keeps the teacher frozen/eval. Batch1/CPU1/fixed600 gradient updates, original208 normal TRAIN views and save points200/400/600 stay unchanged. Each actual checkpoint must pass the **CPU4** numeric guard, then score the same protected80-view CAL set on CPU4. Exact raw/growth search completeness/ranking, normal false alarms≤1/70, pixel precision≥.65, counter64+step guards and all-three global freeze remain required. Root alone may evaluate the one frozen winner on the exposed comparison. No original/CAL fitting, official SDD2 test opening or final speed credit is authorized here.

Pure/mock checks cover legacy GPU compile/default reporting, exact CPU configuration, CPU availability without a GPU, no automatic fallback, invalid metadata, CPU metric/warmup labels, no GPU property queries, retained model/reference/state identities, no BN repeat, exact initialization before the optimizer, and CPU use in every future numeric/CAL path. A pre-protocol AST fixture initially matched the word `torch` inside a reference filename; it was corrected to check actual AST names, without changing the model recipe or executing NN.

Actual immutable protocol/fixture/review receipts and numeric/capacity results will be appended as stages are separately released and completed. No600-step fit has been launched.

## Frozen preparation, before any CPU inference

Independent static review passed after correcting an inherited GPU calibration description and adding a queried-configuration guard. Actual compiled deviceCPU, float32 precision, four inference threads, one stream and latency mode must be reported as supported and match the fixed recipe; unexpected/unsupported values fail the numeric and capacity gates. This is additional configuration validation, not a relaxed numerical tolerance.

The final **24 pure/mock checks pass in0.395 seconds**, including reused exact GPU default/precision/no-Winograd/score/search checks. There were zero source-image decodes, zero weight loads and no actual NN; fake compilation/request results exercised the metadata/reporting branches. No old executed L4 code or receipt was changed.

| Artifact under `output/optimization/factory-round3l5-cpu4` | SHA-256 |
|---|---|
| `protocol-before-CPU4-NN.json` | `b936306945168221d4cca112197cb9459b804bd9727d1ead54fc60a073d0cf81` |
| `preparation-fixtures.json` | `bb8ce16386f46e55ae465c41c0700eae3f6f1adc5afa5e7d5e2df4170d56ae1a` |
| `pre-CPU4-NN-source-snapshot/snapshot-manifest.json` | `b05f7207a8900d992c4f1f591a3764a00a96dfd754e5ea119d3c1fe0174ab074` |
| `spatial_round3l5_stfpm_bn64_cpu4-step0-export.json` (byte-copy receipt) | `82c7fff02789a2f00d686e73a7905a6e8518442a5e80c460f70053ef314daf89` |

Metadata-only copy verification confirms initial L5 ONNX SHA `3eb5f1cd0936342098658e4a808176e4b6bdb8bc1e2103dc3873e7542ed07ace` and reference SHA `5e870bccd9243f1835f2bc2d68e666110bfe4d4903ddcbc605bd577b19e08409`, byte-identical to L4. The distinct CPU sidecar SHA is `e1523a858726b2811252a2f89936b3ef6e173b0436c351df3a3bdedd6cbab572`. CPU runtime source hashes are encoder `ea3acb5be9293b6e75f1a5f73d4ea02670755693f6553c14081da1eb345e02b6` and defect adapter `565d583c45da709456a80687c802ba1645f7c6cd0e547670bb2407904e8a29d8`; their pre-hook L4 versions are archived in the failed-source snapshot.

The only next numeric/capacity commands, after root's explicit quiet release, are:

```powershell
.venv\Scripts\python.exe -X utf8 scripts/probe_factory_stfpm_cpu.py --parity-step 0 --root-released
# Only if the first command passes; preserve its first result and all64 timings.
.venv\Scripts\python.exe -X utf8 scripts/probe_factory_stfpm_cpu.py --preflight --root-released
```

No CPU compile/inference, new BN fitting, capacity call or optimizer update has run at the time of these preparation receipts. The flag does not authorize a stage that root still holds.

## Actual fixed CPU numeric gate: passed

Root subsequently released the exact two staged commands, conditional on numeric/configuration success. The first three CPU cases completed with exit0. Queried compiled properties match the one fixed recipe: execution deviceCPU, inference precisionfloat32, inference threads4, stream count1, and latency mode. The explicit actual-configuration guard passed. All maps are finite, bounded and nonconstant where required.

| Same fixed L4 input/reference | Maximum absolute CPU–Torch score error |
|---|---:|
| First real normal TRAIN source | .0000026822090 |
| Zero initialization tensor | .0000008940697 |
| All255 initialization tensor | .0000008642673 |

All three errors are below.01. The exact numeric receipt SHA is `61ac0c614bd182587288d5b5a505f790b76a23efdca0ae4f78c1b1a430048668`. This is numerical agreement, not measured defect recall. Per-node runtime-info values serialized as opaque `<OVAny class>` objects; therefore their saved histogram does **not** establish individual node execution precision. The property-level float32 hint/configuration and actual numerical result are available; no stronger per-node precision claim is made, and no new inference was run to improve telemetry.

## Actual first complete-profile capacity: failed

Under the same released quiet window, the fixed numeric pass permitted the single capacity command. All other project model/data jobs were held, with root station stopped and ordinary host services uncontrolled. The exact pinned implementation uses16 real normal TRAIN sources, the profile's three initialization-tensor calls and four additional real warmup calls, followed by **64 measured complete-profile calls**. This clarifies that16 denotes sources, not16 warmup calls.

| First preserved CPU4 profile measurement | Result |
|---|---:|
| Model calls inferred / measured |64 /64|
| Mean complete-profile time |78.8874ms|
| P50 /P95 /P99 |78.2341 /91.5285 /93.5400ms|
| Maximum |94.5544ms|
| Serial complete-profile throughput |**12.6763 calls/s**|
| Startup including declared initialization |2.7732s|
| Actual CPU/f32/4-thread/one-stream/latency guard |PASS|
| Required mean≥15 calls/s |**FAIL**|

The first capacity report, including every call/tail and initialization details, is `fullprofile-preflight.json`, SHA `b9eeed190a125577ac30b44219d0f85f728add7807c47f21e218738b916311e0`. The process exited1 because the protected speed guard failed. These are full software-profile serial call timings on predecoded TRAIN images; they do not measure camera capture, decoding, queue delay, PLC actuation or physical every-item coverage.

**The single L5 configuration stops here.** There was no new BN pass,600-step optimizer training, checkpoint CAL, exposed-comparison inference or official SDD2 test access. No thread count, input size, threshold, numerical tolerance or first-run result was changed or retried. The original L4 fitted state and byte-exact copied graph remain retained. The post-execution identity check verifies the protected protocol/source/model/reference identities remain unchanged. All own compute exited and ownership returned to root. Numeric correctness does not compensate for missing the user's15FPS requirement, and this candidate is not adopted.
