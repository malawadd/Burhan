# Separate deployed-precision anomaly-model hypothesis

This is a new conditional experiment, not a tolerance change to L4–L9. Torch parity≤.01 is necessary to claim equivalent export. It is not logically necessary to define and empirically evaluate a different, fixed deployed-precision anomaly function. All historical parity and speed failures stay failed. No deployed-precision inference, model copying, calibration or fitting is authorized by this document.

Root permits source-only preparation, conditional on all three learned L9 checkpoints completing their original strict parity tests and all three failing with zero CAL. The same fixed200/400/600 learned states, six-feature ONNX graphs, GPU/f16/no-Winograd/one-stream/LATENCY execution, input normalization and NumPy float32 score head define the new candidates. There is no new fit, graph/interface/weight/BN edit, configuration grid, output selection or ROI rule. The old Torch discrepancies remain reported but are not called equivalence passes.

The actual software, model/source bytes, hardware/backend metadata and requested/queried execution configuration form the deployed model definition. Actual score repeatability, finite/bounded/nonconstant behavior, real defect localization, false alarms and complete-profile speed determine whether this separate hypothesis is useful. A stable numerical artifact can still be a poor defect model; repeatability does not imply correctness.

## Fixed order and empirical gates

For each of the three candidates, the numeric phase makes exactly six inference calls: the same protected normal TRAIN/zero/255 inputs, then three additional repetitions of the same protected real normal TRAIN image. Every attempted call and actual score map is retained, including failures. All actual outputs must be finite and bounded; the real image must produce a nonconstant map. The maximum absolute difference among the original real-image map and all three repeats must be≤1e-6. Old Torch errors are diagnostic only. Input/weight/operator correctness and the actual unchanged execution-configuration guard remain mandatory.

Only a numerically valid candidate may receive its first64 complete-profile calls on the protected16 real normal TRAIN images, after three initialization calls and four real warmups. The NumPy head, score readback, component postprocessing and every timing tail are included. Mean speed must be≥15. A failed capacity run is preserved, never repeated, and opens zero CAL. All three empirical numeric/capacity outcomes must complete before any CAL is opened; failed candidates remain explicit terminal exclusions.

Passing candidates alone use the same protected80 CAL views, unchanged fixed raw/growth grid, normal false alarms≤1/70, pixel precision≥.65 and weakest-positive rank. Actual score maps come from the declared GPU-feature/CPU-head pipeline, not filtered alternative cutoffs or a Torch surrogate. All three numeric-failure, speed-failure or completed-CAL outcomes are mandatory before one global freeze. The selected fitted policy requires root's fresh≥15 complete-profile benchmark before original comparison and root's sustained16-input-FPS runtime before acceptance. No old capacity receipt transfers.

## Interpretation limits

CAL and the eleven-defect original comparison have been used repeatedly during development. A new fixed protocol can produce honest exploratory evidence on them; it cannot make them blind or erase selection bias. Even a strong result needs fresh source-disjoint validation and real operating-condition coverage before production reliability claims. The anomaly score is bounded disagreement, not calibrated defect probability. No physical site independence, production acceptance, universal factory coverage or automatic code-compliance claim follows from this experiment.

Concrete helper, pure fixtures, immutable source-only method plan and later actual-source protocol require neural-owner and root review before execution. Final protocol binding must preserve all three original failed-parity receipts and models, rather than choose a favorable checkpoint after seeing new results.

## Unexecuted and inapplicable in this phase

All three learned L9 checkpoints subsequently passed the existing≤.01 numeric gate. Therefore this draft's required **all-three old parity failures** condition is false. Root stopped L10 preparation. `scripts/factory_deployed_precision.py` remains an unexecuted source draft: no formal fixtures, method-plan JSON, new assets, model copies, NN, calibration or fit were run/recorded. It must not be treated as a qualified or approved execution path.
