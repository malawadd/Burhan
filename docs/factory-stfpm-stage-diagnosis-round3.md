# L6: bounded stage-by-stage numeric diagnosis

This new diagnostic investigates the retained L4 GPU-versus-CPU mismatch. It is **not training**, a configuration search, a defect-accuracy benchmark or a new production model. No CAL, exposed comparison or official SDD2 test input is used. At preparation time, no L6 inference has run.

L4's actual GPU maximum score errors were .08950025/.04305449/.04961184 on the first fixed normal TRAIN, zero and all255 initialization inputs. The exact same BN-fitted graph on CPU4 matched Torch to within .000002683, but its complete profile reached only12.6763 calls/s. Neither result met the combined numeric/speed gates. All failed source/protocol/model/receipts remain unchanged.

## Existing graph, extra outputs only

The diagnostic adds outputs to the **existing retained ONNX**, rather than recreating or re-exporting the network. A bytewise core hash covers every existing node, initializer, sparse initializer, input and opset; the original score output is also retained byte-exact. Source and diagnostic cores both hash to `1a9484955129b2f539e702bff038b4e68fbeb4b751b71b60f1bba4f6f289dcb4`. The graph has253 original nodes and104 original initializers, with no new computation node or weight change.

The75 ordered outputs expose preprocessing, seven teacher/student block results, each tapped feature normalizer's scale/denominator/unit vectors, three disagreements, two bilinear interpolations, clamped maps/logs/log mean/exponential and the final score. Their exact names/types/shapes are frozen in the append manifest. The complete output footprint is134,420,480 bytes per case/backend; full intermediate arrays are used only during diagnosis. The saved evidence keeps per-stage metrics and four final-score stacks, not all intermediate arrays.

The first metadata-only preparation stopped before model-file writing because ONNX shape inference left eight Resize-derived tensors symbolic. Before any NN/protocol, the implementation resolves **only appended output ValueInfo** to the immutable known fine-map shape1×1×640×256, checks every already-known dimension and preserves all original nodes/weights/inputs. This static preparation failure is recorded in `metadata-preparation-first-failure.json`; it is not an inference retry or output-based selection.

The prepared diagnostic ONNX SHA is `b8538781f53b1e58cc6c45a19fac98326e014e2f80ccaaa66975e5cc88bca6c5`; source ONNX remains `3eb5f1cd0936342098658e4a808176e4b6bdb8bc1e2103dc3873e7542ed07ace`. The underlying saved L4 BN state and the same three Torch reference inputs/maps are pinned.

## Fixed twelve calls and strict context gate

The only proposed actual calls are: untouched serving graph onCPU4/f32 for the three inputs; untouched serving graph onGPU/f16/no-WinogradTrue for the same three; then diagnostic graph onCPU4/f32 andGPU/f16/no-WinogradTrue for three each. This totals exactly12 calls, with no model warmup, flags, thread sweep, crop, ROI, new normalization or changed data. Separate attempted/completed call journals preserve a failed/partial run and prevent silent repeats.

Actual compiled configuration must pass on all four graph/backend instances: declared CPU/GPU device, float32/float16 hint, one stream, latency mode, CPU4 threads or GPU no-WinogradTrue as applicable. Unsupported or unexpected properties fail closed. CPU instances do not query GPU-only properties. Diagnostic outputs are associated through **unique protected tensor names**, with shapes checked; positional association alone is rejected because many features share shapes.

Extra outputs can alter compiler fusion or precision context despite identical ONNX nodes. Therefore **every diagnostic final score must match the untouched serving graph on the same backend/input within1e-5**, for all threeCPU and all threeGPU cases. All diagnostic tensors must be finite; all final score maps bounded0–1; original CPU scores must remain within.01 of their corresponding Torch references; exact twelve-call order/configuration must pass. Only then may a descriptive first-stage difference be reported. Any context failure retains its report, suppresses attribution and permits no output-subset retry or relaxed tolerance.

Per-stage summaries compare CPU toGPU maximum/mean absolute error, RMS/relative RMS and value distributions: zeros, nonzero magnitudes below2^-14 or2^-24 and applicable fixed normalization/map floors. The first relative-RMS difference above1% is a predeclared descriptive flag, **not proof of a faulty kernel or causal explanation**. Even final-score context agreement does not prove the internal compiler/kernel inventory is identical. Numeric inputs contain no defects/GT and support no defect-recall claim.

Six pure fixtures cover append-only graph integrity and corruption, unique output-name association/aliases, both actual configuration guards and unsupported properties, exact call order/final-score context/finite/bounded checks, and tiny/floor metrics. They load ONNX metadata and fake compiled objects, with no Torch model, source-image decode or NN. Independent static review and the immutable protocol are required before root releases execution. Timing may overlap separately authorized wallCPU/media work and is diagnosis-only, never FPS capacity.

Root separately proposed L7 as a bounded CPU-FP32 training experiment from the exact saved L4 state, rather than allowing an untrained GPU mismatch to block all learning indefinitely. That is a distinct future protocol: fixed600 steps/save200/400/600, all outcomes retained, each numeric-failed learned checkpoint excluded without CAL, numeric-passing checkpoints evaluated under fixed CAL, and final fitted GPU numeric/speed/localization gates unchanged. L6 alone authorizes neither that optimizer nor a corrected architecture; diagnosis must precede causal claims or new architecture decisions.

## Immutable preparation receipts

Independent static review passed after checking tensor-name/semantic mapping against the retained ONNX and enforcing all compiled configurations, bounded scores and exact call order. Six pure tests pass in1.181 seconds. Preparation ran no Torch model, loaded no Torch weights, decoded no source image and executed no inference.

| Artifact under `output/optimization/factory-round3l6-stage-diagnosis` | SHA-256 |
|---|---|
| `protocol-before-twelve-NN-calls.json` | `a5815ccf7c1b6357770f59e723d165ff8d6e6ff2a2170af14bb0244e4957d31e` |
| `preparation-fixtures.json` | `ca29f3a1622e47a619a466150bc620d1c8b5d4460a4331aa1fcacb74f147055c` |
| `append-only-output-manifest.json` | `4df46f2bde8bb83a7a8475332a0c7686c3322bf3993e5ccbd07c5055ded54e6a` |
| `pre-NN-source-snapshot/snapshot-manifest.json` | `94bd16215f24bbe04ed2a67b19a2ba190921a471c3cb102acfd04a3f72ec7a2f` |

The fixed command below remains held until root reviews/releases the protocol; the flag is not itself authorization:

```powershell
.venv-roads-export\Scripts\python.exe -X utf8 scripts/diagnose_factory_stfpm_stages.py --stage run --root-released
```

No inference or optimizer has run under these preparation receipts. All later actual outcomes, including failures, must be appended without changing this protocol or its guards.

## Actual twelve-call result: compiler context changed; attribution rejected

After root released the protected protocol, **all12 calls were attempted and completed**, in the declared order. All four actual compiled configurations passed, every diagnostic tensor was finite, final scores were bounded0–1 and post-execution source pins stayed unchanged. The original serving GPU also reproduced the preserved L4 maximum errors exactly. Wall CPU1 work could overlap; none of these diagnostic durations is a throughput benchmark.

The strict final-score preservation gate nevertheless **failed on GPU**, so the process exited1 and no first-stage attribution is permitted. Every saved `first_observed_relative_RMS_above_1pct` field is null. Adding outputs preserves the ONNX computation/weights but materially changes the GPU compiler context/output. That is an observed behavior, not proof that any particular internal tensor, fusion, normalization or kernel caused the original error.

| Same fixed input | DiagnosticCPU vs servingCPU max error | DiagnosticGPU vs servingGPU max error |
|---|---:|---:|
| First normal TRAIN source | .0000017285 | **.08935547** |
| Zero initialization | .0000005662 | **.04199219** |
| All255 initialization | .0000007749 | **.05078125** |

All CPU differences satisfy1e-5; all GPU differences exceed it. `diagnosis.json` SHA is `03abd3c00d80852f9670ce8377469a273afcb9c6ce80dd71ddccb3ef4c3e3168`. All75 stage summaries remain retained with their context warning; they must not be used to assert an internal served-path failure location. The12-call journal and first-failure record remain preserved. There was no output-subset retry, flag change, training, CAL or exposed-comparison inference. All own compute exited and ownership returned to root.

## Cached final-score comparison requested after the context failure

Root then requested a **read-only comparison of already saved final score stacks**, without any new inference or stage attribution. The original three Torch reference maps remain unchanged. The diagnostic GPU's final output happens to match those references much better than the unchanged single-output serving graph did:

| Same fixed input | DiagnosticGPU vs Torch max error | DiagnosticGPU vs servingCPU max error |
|---|---:|---:|
| First normal TRAIN source | .009355843 | .009355426 |
| Zero initialization | .004387766 | .004388571 |
| All255 initialization | .003875732 | .003875077 |

All three diagnostic-GPU final errors are≤.01. This does **not** retroactively pass the1e-5 context guard or permit internal-stage causality claims. It identifies a concrete, separately testable possibility: the **exact full75-output graph** may provide a numerically acceptable final score on these fixed cases. No output subset or best input was selected.

`scoremaps.npz` SHA is `11fa668a59a72db1f7cf20edf484411156d2d001673f4697bdf09f941287fd8d`; the additive `cached-finalscore-comparison-only.json` SHA is `14d5d5ac463ecf2315aa14a60939dfd9cfd3ebf9662716b08541b35fd514bf63`. That artifact reports **zero new NN calls** and explicitly preserves failed context/no-attribution status. Numeric agreement on three normal/initialization inputs is neither defect accuracy nor15FPS qualification. A separate candidate must retain all75 outputs, define a truthful serving adapter and pass new numeric/capacity/accuracy gates before any adoption.
