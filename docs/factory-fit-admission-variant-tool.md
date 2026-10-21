# Additive factory admission variant tool

`scripts/create_factory_fit_admission_variant.py` is prepared but **has not been executed on a real candidate**. It requires a future frozen attempt4 D winner and coordinator authorization before use. It performs no inference, fitting, calibration, test opening, source image decode or image/mask reading.

Its only runtime-policy change is the distinct TRAIN-derived height/width gate **[2.4206,3.4757608695652173]**. It copies the parent's ONNX bytes exactly, retains every original metadata field and selected seed/area/growth setting, changes `supported_height_width_ratio`, and adds an explicit `admission_variant` provenance object. Model name/version, score semantics, input/output geometry, quality settings, manual-review safeguards and training lineage remain untouched. The new model has the same NN hash, a distinct metadata hash, a distinct path, and its own immutable freeze report. Parent artifacts remain preserved.

Required inputs are the parent freeze path **and exact SHA256**, and the [FIT header proposal](C:/Users/IAM/Documents/hackathon/opencv/project1/output/optimization/factory-round3/fit-admission-geometry-proposal.json) path **and exact SHA256**. The tool fails on changed input identities, changed selected NN/metadata/policy, an unfrozen/non-D parent, nonplanned D checkpoint membership, recursive admission derivation, or an already-modified parent aspect gate. It verifies the proposal's1,865 inner-FIT/239 original TRAIN membership, source-hash success, fitting-item scope, protected SDD2 split identity, recorded source geometry and exact fixed min/max×0.98/1.02 envelope. It does not infer physical-domain validation from those dimensions.

All three destination paths must be new and distinct from source/proposal/freeze identities. Exclusive file creation prevents overwriting existing outputs. A failed write removes only this invocation's partial newly created artifacts; original sources remain untouched. JSON serialization has stable sorted keys and no fresh timestamps, so the same pinned inputs/destinations produce identical bytes. The freeze is written last and links parent/proposal identities, the copied NN and new sidecar. Its inherited calibration metrics are explicitly labelled **parent policy provenance**, not new admission-variant evaluation.

Only after the future D freeze exists and the coordinator authorizes this separate experiment, use explicit new destinations:

```powershell
.venv\Scripts\python.exe -X utf8 scripts/create_factory_fit_admission_variant.py --freeze <FINAL-D-FREEZE.json> --freeze-sha256 <EXACT-FREEZE-SHA256> --fit-proposal output/optimization/factory-round3/fit-admission-geometry-proposal.json --fit-proposal-sha256 <EXACT-PROPOSAL-SHA256> --output-model models/factory/<NEW-D-ADMISSION-NAME>.onnx --output-freeze output/optimization/factory-round3/<NEW-ADMISSION-FREEZE>.json
```

This command is a prepared workflow, not an execution record. It does not update application defaults, registered profiles, thresholds, original freeze reports or old result artifacts. A later optional official-SDD2 audit still requires a separate pre-opening declaration pinned to this exact new freeze; the preserved original declaration cannot authorize it. The original D policy and the admission variant must be reported separately. Broadening admission cannot be credited as improved learned localization, guaranteed throughput, manufacturing acceptance or factory generalization.

Thirteen meaningful fixtures passed in2.20s using only temporary **nonexecutable fake NN bytes**. They verify exact NN identity, gate-only metadata differences, unchanged selected policy, original-file preservation, wrong source/model/metadata/proposal identities, missing/non-D freezes, wrong source domains/range, no-overwrite rules, deterministic output and cleanup after a simulated failed final write. No real candidate was copied and no model was run.

## Later coordinator execution record

The preceding sections preserve the original preparation/testing state. Subsequently, the coordinator explicitly authorized and executed the additive tool after the D winner freeze existed, **before any original D comparison result or official SDD2 opening**. This execution copied model bytes and wrote distinct metadata/freeze files; it ran no model, calibration or test input.

- Parent D freeze SHA256: `69b93c8f313c5e9b9342b45aa70fc8f6785c3b6d2ceedcfc576e51f26fc946c0`.
- FIT proposal SHA256: `b73b3a56934b52697ac159bebebfac3642efb335f2f05f4a3680d54d3146759e`.
- New model: [spatial_round3f_orientation_multiscale-step600-fit-admission-v1.onnx](C:/Users/IAM/Documents/hackathon/opencv/project1/models/factory/spatial_round3f_orientation_multiscale-step600-fit-admission-v1.onnx), identical NN SHA256 `93e836b11b2b2bfdb2696b557afa1e4a0e35cb31399101ebe3e396eaf5dfb916`.
- New sidecar SHA256: `839fea3143142dbef389108adbc82acd210e959059de1c98bdc3af1bf812ccd4`.
- Separate [admission freeze](C:/Users/IAM/Documents/hackathon/opencv/project1/output/optimization/factory-round3/winner-attempt4-fit-admission-v1-freeze.json) SHA256: `050fcceca218451312858c56261b95f3ab4abcaec8f71a79ab13f39e20affc60`.

The coordinator reports unchanged defaults, thresholds and parent artifacts. The new gate remains a separately named unvalidated admission policy, without learned-localization or independent-test credit. This record does not imply that its results selected the parent model or that an official test was opened.
