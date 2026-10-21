# L7 independent static review

The six separately versioned L7 common/preparation/trainer/probe/calibration/freezer sources were reviewed without model execution, tests, new source-image decoding or heldout access. The source review found no remaining material blocker after the numerical-failure terminal case was hardened.

L7 explicitly declares a **new experiment**, rather than rewriting L4/L5's failed initialization gates. It permits a fixed CPU-FP32 optimization run from the exact saved L4 BN-fitted state despite failed untrained GPU parity/CPU capacity. Initial agreement or earlier GPU timing is not deployment qualification. The hypothesis that learning changes GPU numerical behavior remains unproven.

Verified contracts include exact original208 normal TRAIN sources/seed, strict saved-state loading before AdamW, source/export/teacher/student hash checks, original sampler/matched crop/loss, student BN epsilon.001/momentum.01, frozen teacher/eval, source archive before optimizer and complete fixed600 updates/save200/400/600 without numeric/CAL feedback. No positive/CAL/comparison pixels enter fitting. Proposal approval now binds both the proposal SHA and exact frozen protocol content, preventing later source drift from silently inheriting approval.

Every learned checkpoint separately receives the fixed three-input GPU/f16/no-Winograd numerical test, reference input identity, queried configuration, finite/bounded/nonconstant output and unchanged maximum error≤.01. A failing checkpoint is excluded with **zero CAL inputs**, while passing checkpoints use the same protected80-view CAL identity, exact raw/growth search/rank and false-alarm/precision guards. All600 training steps and all three terminal outcomes—numeric failure or full CAL—are mandatory before global freezing. Only the one global winner may reach root's exposed comparison; fitted≥15FPS and sustained16-input-FPS qualification remain independent.

One material edge found during review was corrected before receipts/optimization: the serving adapter raises on nonfinite actual GPU scores before a normal parity JSON can be written. Without narrow handling, that would leave a partial checkpoint instead of the intended explicit numeric-failure terminal outcome. The probe now catches only the adapter's exact nonfinite-output error, preserves the failing case/error with JSONnull score/range, attempts the remaining fixed cases and recomputes numeric failure safely. It propagates unrelated checksum, source, geometry and compile errors. A pure fixture covers the valid failed/no-CAL outcome, strict JSON serialization, wrong case count and unrelated-error propagation.

All failures remain failures; numeric-failed checkpoints cannot have CAL artifacts or be ranked. This review does not authorize the optimizer. The owner records formal fixtures/proposal and root separately reviews/releases the exact600-step run. No architecture/default/runtime source was changed by this reviewer.

## Subsequent producer outcomes

The owner reports the root-authorized fixed600-step run completed, preserving all200/400/600 checkpoints. All three then failed the unchanged GPU≤.01 numerical gate; finite/bounded/nonconstant output and queried configuration passed. Maximum errors for the fixed normal/zero/255 cases were:

| Step | Normal | Zero | 255 | CAL inputs |
|---|---:|---:|---:|---:|
| 200 | .068823 | .037044 | .060239 | 0 |
| 400 | .073975 | .036003 | .056443 | 0 |
| 600 | .078888 | .047850 | .059522 | 0 |

All three terminal outcomes were completed and the global winner is null. The hypothesis that this bounded CPU training would resolve the single-output GPU disagreement failed. These numbers are owner-reported outcomes, not independent reruns by this reviewer. The separately predeclared exact75-output deployment-graph experiment retains its own numeric, capacity and root-release gates; it does not rewrite these failed results.
