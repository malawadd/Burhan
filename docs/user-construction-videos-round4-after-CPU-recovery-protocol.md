# Private AFTER CPU recovery: source-only preparation

The original GPU private AFTER helper and its preparation receipts remain unchanged. The separate `scripts/evaluate_user_construction_after_round4_cpu.py` is inert until its explicit CLI stages. No CPU recovery plan, private-video decode or inference has been run during this preparation.

The new helper uses the unchanged model files and .25/.45/native-gain/mask policy. It requires the separate approved CPU recovery protocol, completed 900-step fit, one complete original-author CPU122 baseline, all three complete CPU numeric/CAL outcomes and the CPU global checkpoint choice. An eligible globally chosen model is used for all four private videos; if none qualifies, exact900 remains visibly **FAILED research diagnostic**. Private footage never influences checkpoint or policy selection.

The full original 878-frame implementation is reused through explicit bindings confined to this standalone Python process: new output/plan paths, CPU metadata validator, actual CPU configuration guard and explanatory scope. Original source files, installed registries and runtime defaults are not edited. The actual core adapter must report only `CPUExecutionProvider`, intra-op1 and inter-op1. Full-profile prediction parity is checked separately on public CPU calibration; private processing calls the real core adapter and provides no FPS qualification.

RAW, cached original-author BEFORE, actual new AFTER native prototype masks and information panes retain the original 1400×1240 layout. BEFORE is the actual historical **GPU** result; AFTER is the new **CPU** result. Both training and backend changed, so this private comparison cannot isolate improvement caused by training. This limitation appears in the frozen scope, model-role text shown in the video and final report. It is not replaced with an invented CPU private baseline.

The original file/native-frame hashes, all rational timestamps, contiguous source indices, complete frame count, cached BEFORE NPZ bits and actual AFTER packed masks retain the original integrity checks. Source metadata retains the original inventory and adds the explicit camera-frame count/path required by the UI's held-timestamp mapping. All predictions remain unscored, GT arrays empty and accuracy fields null. Moisture measurement, fixing adequacy and code acceptance remain unassessed.

Future commands, only after root reviews the CPU recovery and all downstream source bindings:

```powershell
.venv-roads-export\Scripts\python.exe -X utf8 scripts/evaluate_user_construction_after_round4_cpu.py --stage freeze
.venv-roads-export\Scripts\python.exe -X utf8 scripts/evaluate_user_construction_after_round4_cpu.py --stage run --root-network-release
```

Freeze is metadata/hash-only after the required CPU public global choice exists. The run needs a separate explicit root compute release and executes exactly 3 initialization calls plus 878 private-source calls for one candidate, with zero old-model inference. First attempts and failures remain preserved under `after-public-cal-choice-cpu-recovery-v1/`.

Construction-owner source review passed. After root released the factory quiet window, four tiny CPU-provider/local-binding/disclosure/source-preservation fixtures passed in 0.002s (0.329s including imports/discovery). The receipt is `after-CPU-source-preparation-v1.json`, SHA `902169263400723d6a5c9c865fe9255373ade5bd0fdfff0251a0e1a88fa8c0ef`. The CPU helper SHA is `da61889638d43a64ac013b186160b2fe17da1641edeff232d015e50e4aa514cb`; the pinned original GPU helper remains `9859a61412192fe2252fc3bad7b83aefe98d054fee207b2f22588eba249cbca4`. No private pixels, model assets, NN, execution-plan freeze or public global choice were opened for these checks. Actual private freeze still waits for all required CPU public outcomes.
