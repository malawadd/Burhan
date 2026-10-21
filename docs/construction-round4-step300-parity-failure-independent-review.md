# Step 300 construction parity failure: independent review

The first step 300 CAL run failed closed at source index 21 after retaining 21 completed frame records. Its initial GPU numeric fixture passed with queried GPU/f32/one-stream execution. The original failed execution, signatures, raw cache and partial records remain intact. No guard was relaxed or CAL retried by this review; the fixed 900-step fit is unaffected.

## What the retained evidence establishes

`output/evaluation/construction-round4-mixed-v3/step0300/full-profile-parity-failure.json` records four crack candidates on each path. Maximum displayed score change is 0.003255; one box changes from `[112,107,190,131]` to `[111,107,194,131]`; three native mask hashes differ. This is not just rounding at six decimal places. A zero-logit mask threshold could amplify mask changes, but it cannot alone explain changed raw class scores or detector boxes.

Independent CPU-only decoding of `full-profile-parity-failure-raw-cache.npz`, using the unchanged native 260×346 letterbox arithmetic and fixed .25/.45 policy, exactly reproduced **all four first/cached signatures**, including every mask hash. It did not reproduce the second/profile signatures. This agrees with the owner's separate saved-output diagnosis. Therefore the retained first raw arrays were not overwritten by the second inference; cache aliasing does not explain this observed mismatch.

Source review found both paths use the same `CubitConstructionModel` delegate and compiled model, the full native RAW frame, shared gain once, same class order and same decoder cuts. `ConstructionProfile.process` passes the full-frame view when no ROI exists; the dispatcher does not apply gain a second time. Its subsequent tracking/annotation logic does not create the captured predictions. Decoder boxes/scores and native cropped boolean masks are owned values before the second inference.

The failed frame does **not** retain independently observed first/second incoming tensor hashes, second raw model outputs or second native mask bits. Its stored signatures are insufficient to establish whether the original input tensors differed or the same GPU request produced different outputs. The one initialization-fixture numeric pass does not prove repeatability on every real input. No kernel cause, precision-jitter cause or model/config mismatch is established.

## Bounded technical diagnosis proposed, not executed

A separate protocol could bind the same step 300 asset, failed real input and unchanged GPU/f32/one-stream recipe, then use exactly six alternating direct/profile calls A/B/A/B/A/B after three declared initialization calls. A process-local request proxy would record the actual incoming tensor bytes/hash/shape/dtype and immediately copy both raw output arrays for every call, plus source/gained hashes and actual native masks. No new cases, GT input, model/config search, policy change or accuracy selection is needed.

Within-route and cross-route comparisons would distinguish changed input tensors from changed raw outputs under instrumentation. All first outcomes must be kept, and even a passing instrumented diagnosis cannot erase or retroactively pass the original failed CAL. This proposal requires separate root authorization and does not alter any frozen source.

This independent review read source files, existing JSON and retained raw output arrays only. It performed no neural inference, image/private-video decoding, training or original-comparison access.
