# Independent static review: F shared-feature normal density

This review reads code/metadata only: no NN, fitting, data decoding, TRAIN/CAL/comparison pixels, official test access, benchmarks or active model edits. The accepted Dec31 construction artifacts remain unchanged. F is a proposed iterative development experiment, not an accuracy or speed result.

## Source features and numerical path

The draft [density graph/export code](C:/Users/IAM/Documents/hackathon/opencv/project1/scripts/prepare_factory_density.py) reconstructs the original frozen `SpatialUNet` with strict state-dict loading. Its `parts()` preprocessing and encoder/skip decoder/classifier operations textually reproduce the original `SpatialUNet.forward()`. Density uses the **16-channel first-encoder-block feature**, at native640x256 stride2 geometry; it does not accidentally use the12-channel later decoder features or a sigmoid score as a raw logit. Original logits and density share one encoder pass. The normal-feature extraction graph is offline fitting machinery and is not a second runtime inference path.

The [moment arithmetic](C:/Users/IAM/Documents/hackathon/opencv/project1/scripts/factory_density_math.py) is mathematically consistent for equal-size native feature grids: covariance is global `E[f fᵀ]` minus the average over columns of `mu_x mu_xᵀ`. It symmetrizes, applies90% covariance plus10% nonnegative diagonal, and floors eigenvalues at `max(1e-6,.001*median(diagonal))`. Its whitening orientation `diag(1/sqrt(eigenvalues)) @ eigenvectors.T` is correct for channel-column feature vectors; `WᵀW` yields the regularized precision matrix up to the explicitly recorded global numerical scale.

The graph bounds centered values to±32, rescales whitening row-L1 to<=8, bounds whitened values to±32, takes mean squares over16 channels, clamps distance before log, floors the train q95–q50 span at.001, and bounds signed normalized evidence to±8. These controls limit practical f16 overflow: squared terms<=1024 and their16-channel sum<=16384; the bounded log-distance range divided by.001 remains within ordinary f16 finite range. Global rescaling cancels in quantile normalization when clipping does not activate; clipping can affect ranks and therefore needs the retained actual clipping diagnostics.

Numerical caveat: `1e-6` is an f16 subnormal, so device/compiler subnormal handling can differ from NumPy float32 arithmetic. Post-log evidence clipping can keep final outputs finite even if a device flushes a zero-distance floor, but finite final output alone does not establish CPU/GPU parity. Actual deployed graph comparison is required before interpreting the fitted density.

The fixed fusion is **signed addition before sigmoid**: `alpha * original_raw_logit + beta * normalized_normal_evidence`, with a baseline control plus eight declared alpha/beta pairs. Sub-q95 normal evidence can suppress the score. There is no detected-code OR/union of masks, max-score recovery, original-miss location rule, evaluation ROI or full-frame fallback in this draft. All output pixels still pass through the ordinary frozen scalar score/mask decoder.

## TRAIN-only fitting and safeguards

The draft [normal fitter](C:/Users/IAM/Documents/hackathon/opencv/project1/scripts/fit_factory_normal_density.py) enumerates only the protected original30 TRAIN item folders, requires239 total views/31 positives/208 actual native-GT-empty normals, and excludes the31 positive TRAIN images from statistics. Neither SDD2 nor reserved/CAL/comparison images enter this head. Moment accumulation is float64; normal quantiles use a separately fixed second normal pass,2048 positions per normal with the declared RNG seed. This is an explicitly supervised TRAIN-normal selection, not an unlabeled or novel unseen-site validation claim.

The [preflight](C:/Users/IAM/Documents/hackathon/opencv/project1/scripts/probe_factory_density.py) proposes16 deterministic original TRAIN inputs/64 complete-profile calls, verifies the NN stage actually ran, and uses a nonconstant dense whitening probe. The eventual fitted graph still needs separate project-quiescent paired capacity/runtime qualification at>=15FPS. A preflight passing on an unfitted bank or prior base graph cannot substitute for that final measurement.

## Actionable checks sent to the owner before execution

1. The initial fitter draft checked only feature-extractor input/output shape, not a feature-extractor SHA pinned to a source/export manifest. Its source-PT hash in the report was copied from the protocol rather than reverified. Requested fix: export a provenance manifest linking actual probe/extractor hashes to exact protected PT/ONNX/metadata identities; require it and current source identities before fitting and fitted exports. No NN was run by this reviewer.
2. Torch identity assertions establish the copied code path, not identity of exported GPU candidate0 to the source ONNX. Require a fixed TRAIN-input raw-logit comparison on the actual exported/device path, with an explicit tolerance and retained failure.
3. Compare fitted graph density to independently computed arithmetic from the actual extracted TRAIN features, checking finite outputs and defined numeric error tolerance. Feature-only GPU compilation and fused graph compilation can introduce different rounding; fitting in NumPy float32 after actual f16 extraction does not by itself prove serving arithmetic parity.
4. Validate finite/shape-correct means, whitening and quantiles, q95>=q50, exact208 ordered normal source hashes, and bank/protocol identity. Hash/count/geometry guards must fail before any new statistics or calibration scoring when an existing artifact differs.

The owner is preparing calibration/freezing integration. Until those files are available and reviewed, this document does **not** claim a completed all-nine cache/freeze audit. Required contract: all nine actual exported graphs use exact ordered80 original CAL image/mask identities; unchanged FP<=1/70 and foreground precision>=65% guards/rank; the control is retained; no source/comparison mining; all nine finish before one global freeze; control winner/no strict rank improvement rejects without a new original audit. Original80 remains exposed development evidence and official SDD2 test remains sealed.

## Follow-up after owner hardening and coordinator preflight

The owner added the requested export manifest linking actual probe/feature/control/diagnostic graph hashes to protected source PT/ONNX/metadata and protocol. Static reread confirms the fitter verifies current source identities, manifest identity, probe identity and actual extractor SHA before feature inference, enforces239/31/208 TRAIN counts, and rechecks208 ordered image/mask hashes after both statistical passes. Bank construction now validates shapes, finite values, ordered quantiles and bounded whitening. The diagnostics API mismatch in the initial drafts is fixed. These changes address the original source/extractor provenance issue; this reviewer did not run their NN or hashing operations.

The coordinator reports an exclusive actual preflight of23.48 complete-profile calls/s, exact actual GPU source/control raw-logit and source-policy mask identity on the fixed TRAIN case, all outputs finite on TRAIN/zero inputs, dummy evidence max error.006278 versus NumPy, and standalone/diagnostic feature max difference.088867. These are owner/coordinator execution results, not independently repeated measurements. They do not qualify fitted-bank accuracy or final contended application throughput. The protected fitted-evidence error guard remains<=.1 before CAL.

Additional static safeguards requested while CAL/freeze code is forthcoming: persist a **fitted-export manifest** binding all nine actual candidate file hashes/coefficients to the exact bank, fit and protocol; pin those identities in fitted parity, CAL and global freeze. The current fitted diagnostic parity needs bank identity so an older pass cannot authorize a changed bank. Also compare the actual single-output alpha1/beta4 candidate to the diagnostic's fused output on the fixed TRAIN input: multi-output diagnostic compilation need not be bit-identical to the single-output served graph. The existing measured feature compilation difference makes this a concrete serving-contract check, without implying a new accuracy defect.

## Completed CAL/freeze code review and actual pre-CAL rejection

The owner implemented the fitted-export manifest and shared [identity validator](C:/Users/IAM/Documents/hackathon/opencv/project1/scripts/factory_density_identity.py), now called by fitted parity, CAL and [global freeze](C:/Users/IAM/Documents/hackathon/opencv/project1/scripts/freeze_factory_density.py). Static inspection confirms exact nine ordered IDs/alpha/beta declarations, standard model paths and actual model SHA; source PT/ONNX/metadata and initial exports; protected protocol/fit/bank/export manifests;208 unique TRAIN-only normal source records; finite bank geometry/quantiles/bounded whitening; and direct CAL report checkpoint/SHA to declared exported candidate binding. This closes the remaining stale/missing manifest combination issue raised in review. Existing source artifacts/protocol remain preserved.

The `--normal-density-complement` calibration branch uses actual exported one-output graphs, separate cache names, fixed native score geometry and exact ordered CAL image/mask/shape identities. The original FP<=1/70 and foreground precision>=65% guard, fixed heuristic shortlist and weakest-positive ranking remain unchanged. All nine planned reports are required. The freeze validates every model, including rejected candidates, and each selected point/metadata/policy/search/source identity. A control winner or a noncontrol tied with the eligible control is rejected; a noncontrol must have a **strictly greater declared rank**. If the control has no eligible operating point, a newly eligible noncontrol is explicitly treated as eligibility gain, not as an invented control score. No OR mask, alternative location rule or comparison-dependent candidate selection was found.

**F did not reach CAL.** The owner reports the fitted numeric guard failed: source/control GPU raw-logit error0 and source masks identical; actual served candidate7 versus diagnostic fused raw-logit error0; finite TRAIN/zero outputs; but actual GPU versus NumPy evidence max error **.79603**, exceeding the protected **.1** guard. Standalone versus full-graph feature max error remained.088867. The failed bank/exports/parity record are retained. No CAL threshold, original-comparison result or speed/accuracy promotion is authorized by these failed numerics. This review did not rerun NN or inspect data pixels.

Any subsequent context-matched feature fitting or precision adjustment requires a separately declared source/extractor/normal-stat experiment and its own TRAIN parity gates; it must not overwrite this failed F record or relax its declared tolerance after seeing the result. The current static code review is complete, with the failed numeric gate serving its intended purpose.

## Separately authorized F2: static review before bank fitting

The coordinator subsequently authorized F2 under distinct prefix `spatial_round3i_normal_density_f16`, keeping failed F artifacts/guard results. Static reread of preparation, normal fitting, shared identity validation, actual GPU parity, CAL and freeze code found no material implementation blocker before its own bank fit. This reviewer ran no NN, tests, data decoding or source-image access.

The offline extractor is a byte-identical copy of the pinned **three-output full pre-fit diagnostic graph**, and fitting selects its unique16x640x256 feature output. Keeping semantic/density outputs alive prevents the fine-only ONNX pruning/compiler-context mistake. The complete source graph SHA is checked before use. The reused23.48/s F receipt is hash-pinned and explicitly an identical-topology architecture gate, not a new F2 speed measurement; final actual fitted graph paired throughput remains mandatory after construction training is quiet.

For raw column mean `mu64=E[f|x]` and deployed nearest-half mean `muq`, the implemented residual second moment is

`E[f fᵀ] - mean_x(mu64 muqᵀ + muq mu64ᵀ - muq muqᵀ)`.

This correctly expands `E[(f-muq)(f-muq)ᵀ]`; simply subtracting `muq muqᵀ` would be wrong. Passing this residual matrix to the existing whitening helper with zero mean retains the same symmetrization/shrink/floor/row-L1 policy. Whitening magnitudes are rounded toward zero to representable half values, preserving the proved row-L1 bound. The quantile pass uses deployed `muq/Wq`. Raw normal q50/q95 are retained; deployed q95 and **independently stored qspan** are quantized afterward. Both graph and NumPy reference read stored qspan directly rather than reconstructing it from differently rounded q50/q95. The validator requires finite correctly shaped exactly-half-representable deployed constants.

F2 still uses the same208 normal sources, same nine coefficients, fixed RNG/statistical sample counts, unchanged scalar fusion, existing mask decoder/grid/guards/rank and protected evidence error<=.1. Variant flags select distinct banks/manifests/parity/CAL caches/reports and attempt7 freeze, with exact source/fit/export/candidate/row bindings. All nine must finish before freeze; an eligible control win or rank tie rejects the fused candidate. No comparison output selects coefficients, normalization, numeric tolerance or source pixels. An F2 fitted numerical failure must remain a failure before CAL; this static review does not predict its outcome or confer accuracy credit.
# Completed timing supplement for globally rejected F2 ID1

The coordinator subsequently measured the lowest-ID eligible noncontrol (ID1) chosen before timing, after all fits/calibration/bounded probes exited and the station stopped. This is timing of a **globally rejected** model, not replacement of the control-gate outcome. No original comparison was opened for F2; its known CAL result remains7/10 meaningful with one normal alarm.

| Project-quiescent paired240 complete-profile CAL calls/model | Baseline | Rejected F2 ID1 |
|---|---:|---:|
| Capacity | 28.9362 calls/s | 26.4618 calls/s |
| Candidate mean/P95/P99/max | — | 37.7903/46.1942/53.1325/57.4604ms |

| Sequential60s CAL replay at16inputFPS | Baseline | Rejected F2 ID1 |
|---|---:|---:|
| Captured /processed | 960 /960 | 960 /960 |
| Drop /late /error | 0 /0 /0 | 0 /0 /0 |
| Decision P50/P95/P99/max | 30.0/39.1/45.4/72.3ms | 34.0/42.6/54.5/82.9ms |

Late means decision age above100ms; source skips/inspection errors are zero. All first tails are retained. These are software replays of real public CAL stills, not moving production-line or physical-item coverage evidence. Repetitions add no accuracy samples. Whole-host exclusivity is not claimed: ordinary desktop services and an unrelated Python process26076 at102.5/105.5% of one logical core remain uncontrolled. The paired curvature baseline's separate drop/late run is retained separately and not substituted here. [Coordinator methods/trace details](C:/Users/IAM/Documents/hackathon/opencv/project1/docs/inspection-improvement-round3.md:368) describe the exact workload. Speed does not repair the strict F2 control-gate rejection or authorize adoption/official SDD2 test opening.

