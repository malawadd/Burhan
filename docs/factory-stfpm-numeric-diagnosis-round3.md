# L2 bounded numeric diagnosis: no passing deployment path

Initial L passed a fresh64-call mean-capacity preflight at18.38839/s but failed the protected .01 GPU-versus-Torch map-error guard (up to .0953511). It stopped before any600-step fit or CAL. [The failed L1 results](C:/Users/IAM/Documents/hackathon/opencv/project1/docs/factory-stfpm-mobile-round3.md) and [all13 executed source files](C:/Users/IAM/Documents/hackathon/opencv/project1/output/optimization/factory-round3l/L1-failed-executed-source-snapshot/snapshot-manifest.json) are preserved.

Root separately authorized this bounded four-path diagnosis. [L2 protocol](C:/Users/IAM/Documents/hackathon/opencv/project1/output/optimization/factory-round3l-numeric-diagnosis-v2/protocol-before-diagnostic-NN.json) SHA-256 **`ee7ce06a3f4eb3d3c3f9ceab4925ad0f59d0a22f373bbbd967fe2df4260d4af3`** preceded every L2 NN call. All paths used the exact three existing Torch-reference arrays: first pinned normal TRAIN image, zero initialization and all255 initialization. No new source images, masks, CAL or comparison data were opened; no optimizer or BN update occurred. CPU1 diagnostics overlapped other accuracy/metadata work and are not speed measurements.

## Actual four-path outcomes

| Path | Exact scope | TRAIN max error | Zero max error | All255 max error | Fixed ≤.01 guard |
|---|---|---:|---:|---:|---|
| A | Unchanged ONNX, ONNXRuntime CPU FP32 | .000001907 | .000001281 | .000000536 | pass |
| B | Unchanged ONNX, OpenVINO CPU FP32 hint | .000001848 | .000000328 | .000000954 | pass |
| C | Exact teacher/random-student initial states, Torch CPU FP16 encoder/preprocessing/normalization | .00758848 | .00115564 | .00312245 | pass, diagnostic only |
| C readout | Same C encoder features, only normalization/map arithmetic FP32; zero extra encoder passes | .00770590 | .00088084 | .00267941 | pass, diagnostic only |
| **D** | **Byte-identical ONNX, OpenVINO GPU.0 FP32 hint** | **.09572923** | **.05731881** | **.06044072** | **FAIL** |

All outputs were finite and bounded[0,1]. CPU FP32 confirms that export agrees with Torch closely on these cases. Ordinary CPU half execution produces a much smaller discrepancy than the installed GPU path; promoting only the same half features' normalization does not materially remove it. The random student's deepest normal-image features are very small:87.78% were nonzero below the FP16 normal-value boundary2^-14 in the CPU-half diagnostic. This is a sensitivity observation, **not proof that GPU flush-to-zero caused the failure**. CPU half and GPU kernels can behave differently, and no GPU intermediate-feature trace was measured here.

The sole deployment correction candidate was deterministically D, rather than whichever path matched labels. It uses the original ONNX bytes SHA `0383b508411a4a86c6f7d0cef17b348a6f17b27f3076e0cdb2f48bf66168566d`, copied to a separate filename with explicit `openvino_inference_precision=f32` metadata; sidecar SHA `0b4683a467e8489118c187932c431b3274ebe990702a411d8c1744c811cdc71b`. Changing that requested hint did **not** fix the mismatch. D exited1 at the numerical guard, leaving its raw score maps/report/failure receipt intact. Its64-call speed stage was therefore **not run**. All declared diagnostic paths are consumed; no new backend, tolerance or case was silently substituted. All compute was released; fixed600 fit, CAL, adoption and any15FPS final claim remain blocked.

## Optional runtime plumbing and validation

Root authorized an optional metadata-controlled inference-precision hint. `FactorySegmentation` now validates `f16`/`f32` and defaults to the identical previous `f16` configuration; `SurfaceDefectModel` passes through `openvino_inference_precision`, defaulting to `f16`. No weights, graph math, image preprocessing, mask postprocessing, thresholds or default model changed. Two pure mock fixtures passed in .335s (receipt wall .336s), verifying exact compile configuration, invalid-option rejection, old metadata defaults and explicit FP32 passthrough without compiling or running a neural model.

L2 pins actual runtime sources: encoder `37061ad8b88ce89f2000314715db8231f621e9455253d920ec647f1e85db3c8c`, defect wrapper `862df394b5fd75675872d6c2d89e790b9df359f718407b3c1432391bc47ebcd1`. Original L1 protocol verification intentionally rejects this later runtime drift, so the failed L1 receipts cannot accidentally authorize fitting after a future correction. A later passing version would require a new protected execution addendum and fresh numeric/capacity gates.

## Primary-source next lead, not yet executed

The [official OpenVINO2026 GPU documentation](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html) explicitly identifies possible accuracy/performance loss for fine-tuned models when Winograd convolutions are selected and documents `GPU_DISABLE_WINOGRAD_CONVOLUTION=True`. The [precision-control documentation](https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/precision-control.html) distinguishes an inference-precision hint from execution-mode optimization policy and describes `ExecutionMode.ACCURACY`.

These are concrete GPU-only correction leads, not established causes. Any test requires a separately authorized fixed L3 protocol and fresh same-case error≤.01 followed by mean≥15 complete-profile capacity. Nothing here authorizes another config search, relaxed limit, CAL fitting or original-comparison model selection.

## Separately authorized ONE L3 no-Winograd candidate: also failed

Root subsequently authorized exactly one new candidate: identical ONNX bytes and original `f16` precision, with only `GPU_DISABLE_WINOGRAD_CONVOLUTION=True`. The [L3 protocol](C:/Users/IAM/Documents/hackathon/opencv/project1/output/optimization/factory-round3l-no-winograd-v3/protocol-before-one-candidate-NN.json), SHA-256 **`db4893690582d9e90118b870a4f1a9d33e0512fa8e999b9ef0acb9012e238be1`**, preceded its NN calls. [Executed L2 sources](C:/Users/IAM/Documents/hackathon/opencv/project1/output/optimization/factory-round3l-numeric-diagnosis-v2/executed-source-snapshot/snapshot-manifest.json), all earlier outcomes and old sidecars remain preserved. The official2026 GPU documentation was saved as original HTML, SHA-256 `3cc2f30d26c357245516df106ada10b60bdbe0fa998099c5dc0268e69ddaf60d`; this pins the consulted source, not proof of the failure's cause. Python's local certificate-store fetch failed before protocol recording; native Windows verified HTTPS retrieval succeeded without disabling certificate checks.

The optional runtime flag requires an actual bool, defaults toFalse, and is **omitted from the compile configuration for old/default models**. Four pure mocks passed in .267s, covering prior precision metadata, strict bool rejection, exact opt-in compiler key and unchanged default compile arguments. Runtime sources are encoder `645ffea3fa26f417a23aa31e0cb39395f70ea32f02acd6bbb7e039a68dbf32d8` and defect wrapper `162e9acd74cead3d76af91752ab4afc6f0e14bece20b9493d24910ce084d7456`. No default factory artifact, weights, score math or postprocessing policy changed.

On the same fixed three numerical inputs, L3 maximum errors were **.09535110 / .05684519 / .05976671**, again above .01. All maps stayed finite and bounded. The three error maxima exactly equal the original f16 report; whole-map bitwise equality was not established because that first report did not cache actual GPU maps. There was **no improvement in the measured numerical guard**.

The report also queried actual compiled properties rather than relying only on requested hints:

| Queried property | Actual returned value |
|---|---|
| `GPU_DISABLE_WINOGRAD_CONVOLUTION` | True |
| `INFERENCE_PRECISION_HINT` | float16 |
| `EXECUTION_MODE_HINT` | PERFORMANCE |
| `EXECUTION_DEVICES` | GPU.0 |
| Runtime graph precision inventory | 114 f16 /2 f32 /1 u8 nodes |
| Runtime nodes with Winograd in reported `execType` | 0 |

The [complete L3 report](C:/Users/IAM/Documents/hackathon/opencv/project1/output/optimization/factory-round3l-no-winograd-v3/three-case-numeric.json) SHA-256 is **`845ef8fe5aaa370eca12c241ac9fd018b5b10511d73fb27f4a0a8ba13a72788d`**. It includes all117 runtime nodes' reported attributes, actual score maps and explicit config. The candidate's NN SHA stays `0383b508411a4a86c6f7d0cef17b348a6f17b27f3076e0cdb2f48bf66168566d`; separate L3 sidecar SHA is `343d385bfbe70250af1e29b421a749a5fcadfa9474310450d4085a5f3fa8be66`.

L3 exited1 at its first failed numerical guard. The conditional64-call speed stage was **not executed**, nor were600-step fitting, CAL or original-model scoring. All own compute was released. This one documented kernel correction did not resolve the GPU discrepancy; no new configuration was tested or error tolerance relaxed afterward.
