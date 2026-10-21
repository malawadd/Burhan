# L8 source preparation: full75-output graph and output-copy comparison

L6's compiler-context guard failed, so its internal stages cannot identify a cause in the original serving graph. Its **cached final scores** nevertheless show that the exact75-output graph agrees with Torch within.01 on all three declared numeric inputs. L8 tests that concrete graph as a separately named architecture probe; it is not an adopted factory model or an accuracy result.

The initial copied ONNX is byte-identical to L6's `b8538781f53b1e58cc6c45a19fac98326e014e2f80ccaaa66975e5cc88bca6c5`. All75 protected output names/types/shapes/order remain present. No computation node, initializer, input, source image, reference score, weight or BN state changes. The same full L4-fitted state/reference identities remain pinned. No output subset is selected.

## Isolated adapter and exactly two copy policies

`factory_tapped_segmentation.py` defines a standalone subclass which replaces only construction. Its `predict` and `warmup` are the **same inherited function objects** as the shared factory runtime. Construction validates the exact model hash, fixed native input, complete75-output schema, GPU/f16/no-WinogradTrue/one-stream/latency configuration and final `anomaly_score` selection. Each profile is instantiated under a CLI-process-local class override; the normal module binding is restored immediately. Shared factory source files, defaults and other processes remain unchanged. Future learned model hashes need their own root-reviewed binding/protocol.

Exactly two output-copy policies are declared before any L8 inference:

| Variant | Native request call | Final score ownership |
|---|---|---|
| `host_copy` baseline | Normal `infer(inputs)` with default `share_outputs=False` | Inherited prediction explicitly copies the selected score |
| `shared_views` optimized | Proxy calls the same native request with `share_outputs=True` | Inherited `np.array(...,copy=True)` owns the final score before another request |

The installed official `openvino/_ov_api.py`/`.pyi` document that default output handling copies outputs, whereas shared output handling returns views. The optimized path therefore avoids unnecessary copies of the74 auxiliary NumPy outputs while retaining all75 graph outputs/configuration/buffers. This is **not** a promise of zero device-to-host transfer or15FPS; framework execution/materialization costs still exist and complete-profile timing must measure them. Mock tests mutate the underlying request buffer after prediction and verify the returned score remains independently owned.

A device-USM auxiliary-output design was inspected from the installed official C++ header/Python stubs and drafted without execution. Before any L8 protocol/NN, root chose the simpler shared-output-view API. The unused draft and source-only decision are preserved; no remote context/tensor was allocated. This preparation decision used API/source reasoning, not performance or model outcomes. Remote buffers remain unimplemented research; there is no hidden fallback, device-memory grid or extra runtime variant.

Both named variants must receive their own first three-case numeric result. Both results are required before any capacity test. Each numerically passing variant then receives its first16-source/64-measured-call complete-profile run, in baseline-then-optimized order, retaining initialization and all tails. A failed baseline is retained and does not silently become the optimized run. Actual queried GPU configuration, finite/bounded/nonconstant output, unchanged maximum error≤.01, all measured calls inferred and mean≥15 remain mandatory. Explicit attempt/failure markers prohibit repeating partial runs. Root must release the quiet window before timing; no L8 NN has run during preparation.

If both fixed variants qualify, the predeclared future runtime choice is the higher measured mean throughput; an exact tie chooses baseline. Numeric/capacity failures cannot be selected. This performance-only choice is frozen before any learned CAL/original evaluation. Initial architecture capacity cannot substitute for final learned-model speed.

## Conditional all-three learned deployment-graph correction

Before **any** L7 GPU numeric or CAL result existed, the separate immutable file `conditional-learned75-proposal-before-L7-numeric.json` was recorded, SHA `d180fb78b4a8e9a653813849d55ad214193e0785b234b97f29b9f71b0eaca067`. It binds all actual L7 source checkpoint/PT/learned-reference/training-report hashes for200/400/600, not a chosen successful epoch. The later runtime study/protocol is honestly recorded after L7 single-output outcomes; its API-based copy-policy refinement does not change the earlier all-three graph plan or select by those outcomes.

Only if initial L8 numeric and fresh quiet≥15 capacity pass, and root separately releases it, the plan appends the **exact same75 existing outputs to ALL THREE L7 learned ONNX graphs once**, without changing any existing node, initializer, input, opset, original score or learned state. There is no new training, BN update, output subset, checkpoint subset, ROI or original-comparison feedback. Each new deployment graph/sidecar/append proof receives a separate identity. Reference inputs stay byte-identical to L4; expected scores must be the corresponding **learned L7 Torch reference scores**, not old BN-only targets.

All three transformed candidates require their own unchanged GPU numeric/configuration gate. A numeric failure is retained and excluded with zero CAL. Passing candidates use the exact protected80-view CAL set, fixed raw/growth grid/weakest-positive ranking, normal false alarms≤1/70 and pixel precision≥.65. All three terminal outcomes are required before global freezing; root alone evaluates the frozen winner on the exposed original comparison. L7's original single-output outcomes remain complete and unchanged. A transformed winner still needs fresh fitted≥15 complete-profile capacity and sustained16-input-FPS runtime qualification.

Eight pure/mock checks cover byte identity, complete schema/order/types, unchanged compile configuration, inherited prediction/warmup, shared-view final-copy lifetime, rejected subsets/aliases/wrong recipes, process-local override restoration, the pre-numeric all-three declaration and explicit numeric failure accounting. Only the exact adapter nonfinite-score error becomes a failed case with JSON null values while all three fixed cases are attempted; unrelated checksum/shape/compile errors stop. This preserves a failed baseline report so it cannot silently block or substitute the separately declared optimized study. The formal fixture/protocol/source snapshot and independent review are recorded before any L8 inference. No correction or integration is authorized solely by this document.

## Prepared receipts, before inference

The independent neural owner statically reviewed the adapter, fixed study order, exact type/schema/configuration, owned-final-copy lifetime and failure accounting and found no remaining material blocker. The formal preparation checker then passed all **8 mock tests in .0986 seconds**, with zero device compilation, source-image decoding, Torch weight loads or NN calls.

| Artifact | SHA-256 |
|---|---|
| Protected initial/conditional protocol | `c19090928ddb09701a807e7d0a5312f21004375a0de4c3616928dfd9fa3295ab` |
| Formal preparation fixture receipt | `3a47357aac8fe57ae10c6caf2e11de50f53387a771220a6b717fc8551448c065` |
| Pre-NN source/receipt snapshot manifest, 18 files | `569f359a866e7717942cc990df98ef69f1a0b09f1844cef61d1b0bf5fc529386` |
| Isolated adapter | `7cf75281090abc2cd2bb149a67a28dc333797124e3d33adc06461f33b4d3b6dd` |
| Standalone probe | `67556050870e6809ea4f03d07416084ebc0c012012ffb2c2346bcb46270f4784` |

Receipts are under `output/optimization/factory-round3l8-tapped75`; the snapshot is `pre-NN-executed-source-snapshot-v1`. The byte-identical initial copied model is `models/factory/spatial_round3l8_tapped75_bn64-step0.onnx`. These receipts qualify preparation only. Actual three-case calls and capacity remain unexecuted pending root's explicit release.

## Executed fixed comparison: numeric pass, both speed gates fail

Root subsequently released the exact protocol after confirming all other project compute was quiet, station stopped and browser paused. The fixed sequence ran once: baseline three-case numeric, shared-view three-case numeric, baseline first64 complete-profile calls, then shared-view first64 calls. Each capacity used the protected16 real normal TRAIN sources, three initialization calls plus four real warmup calls. Ordinary host services were uncontrolled. No graph/configuration/input changes, retries, output subsets or learned transformations followed a result.

Both variants passed actual GPU/f16/no-Winograd/one-stream/LATENCY configuration checks and the unchanged numeric gate. Their final scores were identical, with normal/zero/255 maximum Torch errors `.0093558431/.0043877661/.0038757324`, all≤.01. All75 outputs remained present. Numeric agreement is not defect localization accuracy.

| First64 complete-profile metric | `host_copy` | `shared_views` |
|---|---:|---:|
| Mean milliseconds | 130.647 | 84.675 |
| Serial calls/second | **7.654** | **11.810** |
| Median milliseconds | 130.033 | 83.265 |
| P95 milliseconds | 143.850 | 97.262 |
| P99 milliseconds | 147.162 | 99.919 |
| Maximum milliseconds | 148.881 | 100.349 |
| Initialization seconds | 11.502 | 9.421 |
| Actual calls inferred | 64/64 | 64/64 |
| Actual configuration/source pins | Pass | Pass |
| Mean≥15 gate | **Fail** | **Fail** |

The shared-view path reduced measured mean latency by about35.2%, supporting the API-based copy optimization. It still fails the requested15FPS requirement. No runtime variant qualifies; **the conditional learned75 transformations, CAL, new fit and integration remain blocked by the predeclared capacity gate**. No factory accuracy or production claim is made. All four commands exited; all own CPU/GPU compute was explicitly released to root.

| Preserved first receipt | SHA-256 |
|---|---|
| `initial-three-case-numeric-host_copy.json` | `cd9f92e50b6dcd5fe7764317b19a939bf601610df6d960d7e538a7401b041d65` |
| `initial-three-case-numeric-shared_views.json` | `3b47606d0dc17edb0db3b8cfa98efe1f0db03025201f406b8b83bf09dab5da3e` |
| `initial-fullprofile-capacity-host_copy.json` | `6ab73bd79fa365b690b36efd1e46569927ce110071307d2561ac2911a1f52dd4` |
| `initial-fullprofile-capacity-shared_views.json` | `c52ab3d766e5201981dbe227b09328d9966752ff27e096def4f5280df6f22ca3` |

All64 individual timings/statuses and first-failure markers are retained. The historical preparation statement above describes the state before release; this additive execution section records the subsequent outcomes without rewriting the protected protocol.
