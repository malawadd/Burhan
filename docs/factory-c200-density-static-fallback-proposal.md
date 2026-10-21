# C200 semantic model plus normal-density fallback: static proposal

This is a contingent, separately declared experiment **only if F2 fails its control-improvement or original-development recall gates**. No model, bank, training, calibration, comparison, defaults, or protocol was changed for this proposal. No neural inference or image/mask decoding was performed. There is no speed or accuracy claim.

## Source identity and exact semantic path

Use the already frozen C200 source, not another checkpoint selected after comparison:

- PT: `spatial_round3e_phase_max_refine-step200.pt`, expected SHA-256 `ff3b5571295e80978b4be873b6cce24b7cb947ae25415959a4121337dfe22112`.
- ONNX: `spatial_round3e_phase_max_refine-step200.onnx`, frozen SHA-256 `ca9b90e33f1cacc8cd83ccc8e406b32faa8f45947fd2d672dc48e678b20d7318`.
- Metadata SHA-256 from its freeze: `6d727b95adef756399dd72a9d0a6d814c4ce2a4b72c7cf4ee15abd4a3b702bdb`.
- Freeze: [winner-attempt3-phase-max-freeze-before-comparison.json](C:/Users/IAM/Documents/hackathon/opencv/project1/output/optimization/factory-round3/winner-attempt3-phase-max-freeze-before-comparison.json), expected SHA-256 `c3c2a6d86f86d625f9f6499c4d7477a25c2b9610b26a870be994a29ac969db40`, to be verified before any later execution.

The source class is [PackedPixelRefiner](C:/Users/IAM/Documents/hackathon/opencv/project1/scripts/train_factory_round3d.py:8), exported through [export_serving](C:/Users/IAM/Documents/hackathon/opencv/project1/scripts/train_factory_round3c.py:99) with `SERVING_POOL=2` in [round3e](C:/Users/IAM/Documents/hackathon/opencv/project1/scripts/train_factory_round3e.py:1). It is not the plain SpatialUNet used by F2.

Preserve this operation sequence exactly: BGR uint8 → float32 `/255` → RGB/ImageNet normalization → first seven encoder blocks; save first-block `fine` and fourth-block `middle`; execute the extra context blocks and context projection/addition; run the coarse/middle/fine decoder; bilinearly upsample semantic logits to 1280×512; compute raw-channel mean and 5×5 local contrast with `count_include_pad=False`; pixel-unshuffle these two channels into eight half-grid phase channels; concatenate them with the 12 decoder channels; apply the learned 20→8→8→8→4 refiner; pixel-shuffle its residual to full resolution; add to full-resolution semantic logits; then **MaxPool2d(2), default stride 2, no padding**. The resulting C logit map is 640×256.

## One shared encoder and aligned fusion

Implement a separately versioned paired forward returning `(pooled_C_logits, fine)` from that same encoder pass. Load the complete C state strictly, including context, refiner, normalization, and all batch-normalization buffers; freeze every parameter and use eval mode. Do not run `base(bgr)` and a second encoder for density, or replace C's forward with F2's simpler decoder.

The saved first-block activation has shape `1×16×640×256`, aligned with the 2×2 C output cells. Apply the F2 density head to this activation and fuse **after C's full-resolution addition and max-logit pool**:

`R = alpha * maxpool2(C_full_logits) + beta * signed_density(fine)`.

Keep the nine fixed coefficients: `(1,0)` control, and `alpha ∈ {.5,1}` crossed with `beta ∈ {1,2,4,8}`. One single-output GPU graph returns raw fused logits, followed by the unchanged serving sigmoid and component/growth policy. No mask OR, fallback, extra output transfer, ROI, phase rearrangement, or pooling of probabilities. Although adding a cell-constant density before pooling can be algebraically equivalent for positive alpha, changing operation order introduces unnecessary compiler/rounding differences and obscures source parity.

## Reuse the numeric method, refit the bank

Reuse F2's numeric fixes, **not its fitted bank or its feature graph**. C's learned early features and full compiler context differ. Fit exactly the same 208 GT-empty original TRAIN normals with their existing source identities, without SDD2, calibration, or comparison inputs. Extract features from the new **full C+density diagnostic graph retaining semantic, density, and fine outputs**, avoiding a pruned first-block-only graph.

Keep float64 full-context moments; half-representable deployed mean; residual covariance `E[ffᵀ] − mean_x(μ64 μqᵀ + μq μ64ᵀ − μq μqᵀ)`; the fixed shrinkage/eigenvalue floor; whitening rounded toward zero before the quantile pass; the same clipping limits and fixed TRAIN quantile samples. Store independently half-representable q95 and qspan, and use those exact constants in graph and reference. No numeric-guard relaxation or calibration-based bank adjustment.

## Mandatory gates and provenance

1. Record a distinct immutable protocol/prefix, full C freeze/PT/ONNX/metadata hashes, implementation hashes, exact 208-source receipt, all nine coefficients, and unchanged calibration search/rank/strict-control-improvement rule. Do not reuse F2 caches or source bindings.
2. After construction training ends and the project is quiet, run a new nonconstant dense-head full-profile preflight at 1280×512. This larger context/refiner graph cannot inherit F/F2 throughput. Abort the experiment if it cannot sustain at least 15 complete calls/s; no accuracy credit from dummy inputs.
3. Before CAL, verify the zero-density control against original C ONNX raw logits and frozen-policy masks on fixed TRAIN inputs, including source edges and nonconstant phases. Verify finite zero/TRAIN outputs; full-context diagnostic features; actual single-output candidate versus diagnostic fusion; and GPU versus exact-stored-constant NumPy evidence max error ≤.1. Adding a branch or pruning beta-zero density can alter FP16 compiler choices: compare outputs, not only Torch algebra.
4. Bind the refitted bank, all nine ONNX/metadata exports, parity receipts, and exact ordered 80 CAL source identities to their caches/reports. Run all nine actual single-output graphs. An eligible control winner or rank tie rejects the experiment; strict fusion improvement is required before any new exposed-development audit.
5. Freeze once before comparison. Preserve failures and assess original-development localization with the existing criteria. Final fitted-winner project-quiescent paired capacity/runtime must independently sustain ≥15 FPS. No official SDD2 test opening or widened admission gate follows automatically.

Source identities above were read from existing metadata and earlier pinned receipts; this static proposal did not rehash model binaries or execute their constructors.
