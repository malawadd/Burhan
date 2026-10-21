# L9 preparation: GPU raw feature pyramid and CPU float32 score head

L8 showed that a75-output graph could pass the fixed Torch score check, but even shared-output views reached only11.81 complete-profile calls/second. L9 is a separately declared network partition: the GPU computes teacher/student raw features at blocks0,3,6; the CPU computes the original anomaly-score arithmetic in NumPy float32. The six outputs define a fixed architectural boundary, rather than a search through output subsets. No inference has been run during this preparation.

The graph source is the exact L4 BN-fitted ONNX, SHA `3eb5f1cd0936342098658e4a808176e4b6bdb8bc1e2103dc3873e7542ed07ace`, with the retained L4 state/reference identities. The extractor walks backward from the six predeclared tensor names, keeping every dependency in original graph order. It checks byte equality of every retained node, initializer, input and opset. Input BGR→RGB/ImageNet normalization remains in the original graph. No ONNX weights are loaded into a model, rebuilt, trained or exported through Torch.

Static extraction produced `models/factory/spatial_round3l9_partitioned_bn64-step0.onnx`, SHA `b8ecf60fa5ac104857d9c0713d44ea94c4c5c253be261f8d8df6e990da07018c`. It retains123 of253 original nodes, with unchanged retained bytes. The immutable manifest records the complete ordered output schema and extraction proof.

| Interface outputs, teacher and student each | Shape | Float32 output bytes, both |
|---|---|---:|
| Block0 raw features | 1×16×640×256 | 20,971,520 |
| Block3 raw features | 1×24×160×64 | 1,966,080 |
| Block6 raw features | 1×40×80×32 | 819,200 |
| Total | Six outputs | 23,756,800 |

The smaller interface does not prove speed. All raw features still reach the CPU. The NumPy head and inherited component postprocessing must be included in complete-profile timing; the previous graph's timing does not transfer.

## Fixed runtime and score semantics

The standalone `FactoryPartitionedSegmentation` subclasses the normal factory adapter, replacing only construction and the request proxy. The deployed `Base.predict` and `warmup` remain the same function objects. The GPU recipe is fixed GPU/f16/no-WinogradTrue/LATENCY/one stream. All six model and compiled output names, order, shapes and f32 interface types must match. The synchronous request uses `share_outputs=True`; feature views remain alive while the CPU head executes and are never written in place. The proxy returns an owned float32 score. Inherited prediction immediately makes its normal owned copy before any later request.

The CPU arithmetic retains the original formula: channel absolute maximum with floor2^-8, scaled-feature square-sum floor2^-12, unit-feature normalization, quarter squared L2 disagreement clipped0–1, bilinear half-pixel resizing with `align_corners=False` to640×256, then the left-associated three-log-map mean/exponential with map floor2^-14. Every operation uses float32. There are no fitted score statistics, probability calibration, new masks, ROI rules or defect-dependent branches. The bounded result is anomaly disagreement, **not a defect probability**.

This partition changes numerical execution and compiler context deliberately. Prior failed L6 context agreement does not identify a kernel cause, and L9 makes no such attribution. Its own fixed numeric test against the original Torch references is mandatory.

The full profile is instantiated using a CLI-process-local constructor binding restored immediately after construction; no shared factory file, default model or UI setting changes. Each measured call records host wall-time stages for GPU request/framework output handling, feature-view access, CPU float32 head and their total. These are not GPU-kernel event timings. No extra inference is used to obtain these stages.

## Gates before any learned graph correction

The initial architecture receives exactly the same three protected L4 numeric inputs: first fixed real normal TRAIN image, zero and255 initialization cases, with original Torch expected scores. The unchanged maximum absolute error≤.01, finite/bounded/nonconstant score and actual queried GPU configuration all must pass. Known nonfinite-output failures receive explicit JSON-null cases while all three fixed cases are attempted; unrelated checksum, geometry or compile failures stop. The first attempt/result is preserved without retries or configuration alternatives.

Only after numeric pass and root's quiet release does the initial model receive16 fixed real normal TRAIN sources, three initialization calls, four real warmup calls and its **first64 measured complete-profile calls**. All64 must infer and mean speed must reach15 calls/second, including the NumPy head and postprocessing. Every timing/tail is saved. No source, output count, precision, thread or input-geometry grid is authorized. Ordinary host services remain uncontrolled and actual workload is recorded.

Before any L9 numeric outcome, the protected protocol binds all three actual L7 learned checkpoint ONNX/PT/reference/training-report identities,200/400/600. If initial numeric and capacity gates pass and root separately releases the correction, the identical six-feature extraction applies to **all three** once, with no new optimizer, BN fitting, model fit or node/weight edits. Each uses the same CPU head/GPU recipe and corresponding learned Torch score references with byte-identical input cases. Numeric failures are explicit terminal outcomes with zero CAL; passing candidates use the same protected80 CAL views, fixed raw/growth grid, weakest-positive rank, normal false alarms≤1/70 and pixel precision≥.65. All three terminal outcomes are required before global freezing. Root alone tests the frozen winner on the exposed original comparison; fitted≥15 complete-profile speed and sustained16-input-FPS runtime remain mandatory.

The L7 single-output failures and L8 failed capacity results stay unchanged. L8's conditional75-output correction remains blocked; L9 does not silently substitute for that protocol. No CAL, original comparison or official SDD2 test input is read during L9 preparation. This document authorizes no NN; root reviews the pinned protocol and releases each execution phase separately.

## Preparation verification

Ten initial pure/mock/ONNX-metadata tests passed in .470 seconds, under ordinary project contention, with no neural model constructor, weight load, source-image decoding or forward. They cover deterministic ancestor extraction, source byte invariance, fixed six-output boundary, numeric floors/zero/tiny/half-maximum features, analytic disagreement, a scalar double-precision half-pixel oracle, declared NumPy arithmetic, score ownership, inherited methods, restored local constructor binding and explicit numeric failure accounting. This duration is test telemetry, not FPS. Independent tiny Torch arithmetic/interpolation fixtures and final source review are pending before the formal receipt/protocol is recorded.

The subsequent independent review found no remaining material blocker. Five additional tiny Torch-only arithmetic fixtures compare the actual original `unit_feature`/`disagreement` functions and `F.interpolate` on zero, subnormal, normalization-floor boundaries, high finite values, odd/up/down/degenerate resizing and the complete three-level score. They pass with2e-6 tolerance. They construct no network model, load no weights and perform no neural forward. Those arithmetic tolerances do not replace or relax the actual three-case runtime maximum-error≤.01 gate.

The combined formal checker passed **all15 tests in .9149 seconds**, excluding ordinary process startup/import costs, under project contention; this is not FPS. The exact sources, tests, extraction proof, sidecar and receipts were frozen before any L9 inference:

| Prepared artifact | SHA-256 |
|---|---|
| Protected initial and conditional-all3 protocol | `522a1f549ca08e964fa46c2787f32370da0b4787dba3fe1ab3809efc1f87b32c` |
| Formal15-test fixture receipt | `df864134111353b89f1ace00ddf1da8e71edb0ca620d33eab5a3bcbb39f1c1fe` |
| Pre-NN23-file source snapshot manifest | `bf4c90466de80cb083144f2a338155b7e41bc3ceefe407de7d5e18c3180a8fe9` |
| Initial six-feature ONNX | `b8ecf60fa5ac104857d9c0713d44ea94c4c5c253be261f8d8df6e990da07018c` |
| Initial sidecar | `d0c432bef2dd0ec52df96f5098f982e5fa3b7fa48d6c3293e1fe68cfb46d3693` |

Artifacts are under `output/optimization/factory-round3l9-partitioned`; the snapshot is `pre-NN-source-snapshot-v1`. Preparation and review are complete. **Actual numeric, capacity, learned transformations and CAL remain unexecuted and require root release.** The earlier pending statements above record the preparation sequence; this additive section preserves that history.

## Additive method clarification, before any L9 inference

After recording the v1 protocol but before any L9 numeric/capacity call, root changed one future investigation prerequisite. A separately root-released, deterministic transformation and numeric investigation of **all three already-trained L7 checkpoints** may proceed regardless of the BN-only initial prototype's numeric or capacity outcome. This costs no new model fit and avoids treating an initialization failure as proof that all learned states must fail. It is a newly declared method amendment, not a reinterpretation of an observed L9 result.

The v1 protocol/source/fixture/snapshot stay immutable. The additive `pre-NN-learned-investigation-amendment-v2.json` overrides only the dependence of future learned investigation on initial-prototype pass. Initial numeric/capacity recipes and first failures remain unchanged. All three source weights, exact six-feature interface, shared CPU score head/GPU recipe and no-new-fit rule remain fixed. Each actual learned candidate must still pass≤.01 numeric/configuration/finite/bounded checks before any CAL; failed candidates receive zero CAL. All three terminal outcomes are mandatory before freezing. Acceptance still requires fixed localization/false-alarm guards, fresh **actual fitted** mean≥15 and sustained16-input-FPS runtime; root may declare fitted capacity before original evaluation. Root's separate release is required before any learned transformation or inference.

The amendment helper and three pure fixtures validate the original v1/snapshot identities and all-three source bindings and reject relaxed numeric/speed/failed-CAL/root/all-three guards. The original L8 plan and its failed capacity are untouched. The amendment is recorded before any L9 inference and bound by the separately versioned learned pipeline.

Independent static review passed before recording the amendment. Its three pure fixtures passed. The additive amendment SHA is `343a0794f5ccb1bf504274a03691dc43a7d3b07fdf7d25c4180db76d1b4d1f38`, its fixture receipt SHA is `2e77d00ffd83d8319f83a2843fc1edd0e9d9407a0db16bffa1737b49708572db`, and its separate four-file source snapshot manifest SHA is `474b87445e1a8ea5d33961b8166f8f3d5129cfdcfdcc206a53526a1e352206e0`. No L9 NN had run when these were recorded. Initial probe execution remains separately released by root.

## Initial numeric execution: failed, no capacity calls

Root released the exact base protocol plus amendment in a project-quiet window, with all other project CPU/GPU jobs stopped, station stopped and browser paused. The first three-case command completed all three cases once. Actual queried GPU/f16/no-Winograd/one-stream/LATENCY configuration passed; all outputs were finite, bounded and nonconstant, and source pins were unchanged. The initial model nevertheless failed the unchanged maximum error≤.01 gate:

| Protected input | Maximum error versus original Torch score | Score range | Numeric gate |
|---|---:|---|---|
| First fixed real normal TRAIN image | .0540667772 | .2799028–.6749470 | **Fail** |
| Zero initialization | .0036967397 | .3328972–.4911092 | Pass |
| 255 initialization | .0031677783 | .3016735–.4748525 | Pass |

The overall initial result is **failed**, and the command exited1 after writing the result and first-failure receipt. No capacity attempt was created: there are zero64-call capacity measurements, so no capacity stage breakdown or FPS claim exists. No rerun, additional numeric input, configuration/head change, learned transform, model fit or CAL followed this result. All own CPU/GPU compute was released to root.

The numeric receipt SHA is `bc19743808863137c6a81ae5bcb4de8a43cad4b676ed4d79a43605ff0a81bf24`; the first-failure receipt SHA is `1bbe5a82658b5eaa4db3e9b5022b734f694bf017dd5a127407aeacce7a0f6503`. These results do not identify a GPU kernel cause. The pre-inference method amendment permits a separately root-released all-three learned investigation; it does not turn this initial model into a numeric or speed pass.
