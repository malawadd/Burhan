# L7 proposal: learn in CPU FP32, then apply unchanged GPU gates

This is a distinct proposed experiment, not a correction of the preserved failures. L4's BN-fitted untrained student fails GPU numerical parity; the same exact graph/state agrees numerically on CPU FP32, but CPU4 capacity is only12.6763calls/s. L6's added outputs also perturb GPU compilation, so its stage statistics cannot identify the served-path cause. None of those candidates qualifies for deployment.

The new hypothesis is that normal-only learning can change the student's activation distributions and subsequent GPU numerical behavior. It may fail. A separately approved fixed CPU FP32 fit can test that hypothesis without calling the random student's GPU failure a passing preflight. The initial failure and CPU capacity failure remain recorded. Prior GPU speed establishes only architectural context and gives this experiment no fitted capacity credit.

The proposed fixed recipe preserves the exact L4 BN64 state, ImageNet teacher,208 original GT-empty TRAIN views, original seed20261030, identical shuffle/crop plan, normalization, three-level loss, BN momentum.01 and CPU1 optimizer execution. It uses600 updates and saves200/400/600 with no early stopping, extension or feedback. Teacher/normalization invariants, finite student parameters and BN buffers, and counter64+step checks remain. No graph, target grid, precision choice, threshold grid, TRAIN source or augmentation changes are introduced. No optimizer is currently authorized by this document.

All600 updates and exports must finish before any learned GPU numerical/CAL evaluation. Each checkpoint then receives the same byte-identical first-normal/zero/all255 input arrays with its new finite bounded FP32 reference maps. Actual GPU/f16/no-Winograd/one-stream/LATENCY configuration, nonconstant first TRAIN map, finite bounded output and maximum absolute error≤.01 are required. A failed checkpoint gets an explicit numeric-failed terminal outcome with zero CAL inputs and is excluded. A passing checkpoint receives the unchanged ordered80 CAL source/GT identity, guarded threshold search and ranking. All three terminal outcomes are mandatory before global freeze; a mixture of failed numeric outcomes and completed CAL is explicitly accounted for, never silently dropping a checkpoint.

Only a global CAL winner can reach the coordinator's exposed original-mask/bbox audit. The winner still requires fresh fitted-policy≥15 complete-profile performance and sustained16FPS end-to-end replay with all processed/dropped/late/queue counters before acceptance. This is still a replay comparison rather than measured production-line coverage. Official SDD2 test remains sealed; normal-view alarm rates do not establish good-product rejection rates.

Implementation is isolated under `*_factory_stfpm_learn.py` and `factory_stfpm_learn_common.py`, prefix `spatial_round3l7_learned_stfpm`, output `output/optimization/factory-round3l7-learned`. Pinned immutable prior receipts, exact approved-proposal/protocol equality, executed-source archiving before the optimizer, per-checkpoint raw-map/parity/source hashes and all-three freeze checks preserve lineage. Shared runtime, defaults and construction experiments are unchanged. Initial pure preparation checks pass; independent review and a separately hash-bound parent decision precede recording the fit protocol or launching any optimizer.

The completed independent source review passes after a terminal-policy hardening: a narrowly identified adapter nonfinite-score exception is preserved as a JSON-null numeric failure for that fixed case, all three numerical cases are still attempted, and the checkpoint receives numeric-failed/no-CAL status. Unrelated source, compilation and geometry errors propagate as hard stops. The validator recomputes eligibility from actual errors/ranges/reference/configuration evidence; it does not trust an acceptance flag alone.

Formal preparation receipt:9 pure checks pass in.692s, with no model loading, NN, optimizer or image decoding. Receipt SHA `214cc519827ddacb3f7cde5cb4fd4eba0671dee5553194862a63bbc32a25a80b`. Proposed protocol SHA `f360eda9a28200bfd72e315a3a0933555aece2c83a68375a0687a3e584bcddf0` at `output/optimization/factory-round3l7-learned/proposed-protocol-before-fit.json`. The fit remains unlaunched pending the coordinator's exact proposal-bound approval/release. Any source change requires a preserved, separately versioned proposal rather than silently replacing these bytes.

## Exact fixed fit completed

The coordinator approves the exact proposal and releases the fit. The protected protocol is byte-identical to proposal SHA `f360eda9a28200bfd72e315a3a0933555aece2c83a68375a0687a3e584bcddf0`. CPU1 training completes600/600 with exit0, no extension or numeric/CAL feedback. All200/400/600 model, optimizer, ONNX, three-reference and training receipts are saved. Teacher state remains unchanged, student/BN state stays finite, counters are264/464/664, and all208 source hashes are rechecked. Completion receipt SHA `07d46f9cc215c73ef100c67c0d61cda865105298be857cb27c5bc1eb57310c5f`; source archive manifest SHA `eb048effbdcd1e014c2a44e9d9e28220e1b22cd2483e2e87ea6fa60aae01e6cb`.

| Step | Normal TRAIN loss | Recorded elapsed seconds |
|---:|---:|---:|
|50|.543806|25.26|
|100|.439665|50.82|
|150|.417022|76.03|
|200|.329077|101.10|
|250|.296694|129.52|
|300|.272354|154.67|
|350|.249535|169.65|
|400|.219385|184.30|
|450|.161111|199.55|
|500|.141840|214.00|
|550|.122616|227.52|
|600|.118597|241.80|

Total optimizer/export completion is243.97seconds, under overlapping wall-v3 CPU1 training and ordinary host work. These are contended TRAIN/export timings, not inference FPS or accuracy. Decreasing normal training loss does not establish defect recall. At fit completion **no learned GPU numeric case, CAL or original comparison has run**; all-three outcomes and global selection remain pending a coordinated GPU release.

Learned ONNX identities:200=`ef97e25489833a8619c9f2f63a1808f9044767ccf202e1a30ec7e5ab9fbae906`;400=`51f341c506de404bcb69f65c696c870ed45f9e09813a089627a0c090bdf85809`;600=`19f6286ceefeddb3fa76637df25eaed5827bab0b8e65fbe32606403320bbdae0`.

## All learned GPU numeric outcomes: rejected before CAL

After completed600 training and an explicit coordinated GPU release, all three fixed learned checkpoints run all three protected numerical cases. Actual GPU/f16/no-Winograd/one-stream/LATENCY configuration, finite/bounded output and nonconstant first-normal maps pass for every checkpoint. The unchanged maximum absolute error≤.01 criterion fails for every checkpoint:

| Learned checkpoint | First normal TRAIN error | Zero error | All255 error | Outcome | CAL inputs opened |
|---:|---:|---:|---:|---|---:|
|200|.0688227713|.0370437503|.0602385402|Numeric failed, excluded|0|
|400|.0739750266|.0360029042|.0564430952|Numeric failed, excluded|0|
|600|.0788882673|.0478497148|.0595220327|Numeric failed, excluded|0|

The all-three global freeze completes with **accepted_for_independent_audit=false, winner=null**. Learning under this fixed normal-only recipe does not resolve the numerical mismatch. This run opens no CAL or exposed original comparison and runs no fresh75 diagnostic or fitted performance evaluation; no recall/precision improvement is established. Any separately predeclared diagnostic remains the coordinator's decision and cannot promote this rejection. No tolerance, graph, configuration or schedule is changed and no checkpoint is promoted. Official SDD2 test remains sealed.

Global rejection SHA `b2409626143f6857962d6a3e2a8bf1e38b1a60db6d6006689c2aeb98c83801f8`. A completed39-artifact manifest pins all three model/PT/optimizer/ONNX/sidecar/reference/export/parity/terminal/log assets, completion/approval/protocol/fixture receipts and the source archive. Its SHA is `62dd1d55ec89bbaf23362d1e05076be557caa758843b73de585406a677ef145f`. All executed source pins still match the pre-optimizer archive at closure. All worker sessions exit0 for their explicit numeric-failed terminal outcomes and global accounting; an exit0 does **not** mean model qualification. CPU/GPU ownership is released to the coordinator.
