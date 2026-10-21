# Private-video AFTER diagnostic: prepared, not executed

`scripts/evaluate_user_construction_after_round4.py` is a new standalone helper. No plan freeze, private-video decode or model inference has been run during its preparation. It does not change baseline artifacts, installed model registries, runtime defaults or shared model code.

## Root execution sequence

Only after construction finishes all 900 updates, all three 300/600/900 export/numeric/CAL outcomes and `global-three-checkpoint-freeze.json`:

```powershell
.venv-roads-export\Scripts\python.exe -X utf8 scripts/evaluate_user_construction_after_round4.py --stage freeze
```

Freeze performs metadata/hash checks and saves the exact model, policy, original inventory, baseline reports, original executed-source archive, software versions and new source archive. It imports no network, opens no source pixels and executes no model. The chosen model is the globally ranked public-CAL winner; if none qualifies, it uses **exact 900 FAILED research diagnostic**, with this status visible in every output. It cannot choose another checkpoint from private-video results.

After root reviews that frozen plan and explicitly allocates compute, following the factory quiet benchmark or another explicit root release:

```powershell
.venv-roads-export\Scripts\python.exe -X utf8 scripts/evaluate_user_construction_after_round4.py --stage run --root-network-release
```

Actual run uses one local GPU/f32/one-stream model, the unchanged 640/.25/.45/shared-native-gain decoder and both author classes `[crack, spalling]`. It makes exactly 3 zero-frame initialization calls and 878 actual private-source calls across all four complete videos. It makes **zero BEFORE model calls**. These are offline diagnostics, not FPS measurements.

The AFTER helper calls the actual core `CubitConstructionModel.infer` adapter, **not** `ConstructionProfile.process`. Full-profile prediction parity is checked separately in the required public CAL reports. Private AFTER timings exclude profile tracking/findings and the edge runtime queue/evidence path, and provide no full-profile FPS or deployment qualification.

## Integrity and presentation

- The immutable baseline plan, original inventory and independently completed baseline-audit identities are pinned explicitly. Current full public experiment identity, model/export metadata and numeric receipts, all three complete 122-record CAL reports and global ranking are checked before private inference.
- Every original file SHA is verified. PyAV checks full original rational PTS/count; OpenCV supplies each native BGR exposure. Every native hash, source index, dimensions and original timestamp must match the audited baseline before that exposure reaches the new model.
- Original-author BEFORE masks are reconstructed from cached NPZ bits, verifying file SHA, packed SHA, native dimensions, foreground count and padding. They are never recomputed or fabricated from boxes.
- The 1400×1240 output shows separate RAW, cached actual BEFORE masks, actual new AFTER native prototype masks and an information pane. Both model panels use the same native gained display; its gain must match the original and new inference policies. RAW remains separate. No GT pane exists for these unannotated videos.
- Every original exposure is encoded once with its exact relative rational PTS, checked by decoding the final MP4. Per-frame JSON, actual AFTER packed-mask NPZ files, full unscored report, class counts and prediction-present frame counts are retained.
- Report `source` retains the complete original inventory body and adds `original_camera_frames=frames.length` and `original_path=file`. Together with contiguous source indices, this lets the UI use the original held-PTS mapping rather than nearest-frame mapping between camera timestamps. The original inventory file is unchanged.
- All predictions remain `unscored`; `evaluated=false`, `expected=[]` and accuracy/match/false-alarm fields are null. Candidate presence is not defect accuracy. Humidity, fixing adequacy, structural severity and Saudi code acceptance remain unassessed.
- An exclusive global attempt marker, exclusive output folders and exclusive JSON artifacts preserve partial failures. A failed run is not silently retried or overwritten. Revalidation after all four videos detects source or policy drift during execution.

Outputs stay under `output/evaluation/user-construction-videos-round4/after-public-cal-choice-mixed-v3/`. Original private videos remain local and are never used for training, selection or upload.

## Preparation checks

Five tiny metadata/cache fixtures passed in 0.022s in the final formal run (0.682s including fixture imports/discovery). The receipt is `output/evaluation/user-construction-videos-round4/after-source-preparation-v1.json`; source SHA is `50737a32d23dcf38967d5abb47e706b1d7caa27c0d700e56d6481b1751bbea45`. They cover global-rank/exact 900 fallback, duplicate or wrong winners, queried actual backend configuration, packed cached-mask identity/count and inert import/full-call scope. Fixtures use tiny synthetic bit arrays only; no video, photo, weights or network inference.

Construction-owner source review passed after the per-frame gained-hash write order was corrected. The historical baseline has no saved gained-pixel hash; the helper checks the original shared gain source identity and original recorded gain instead of inventing one, then saves the actual new gained hash. No inference speed or accuracy outcome is claimed by this preparation document.

Root's subsequent UI review added the explicit original-camera-frame count/path report contract before freeze. Six pure fixtures now pass in 0.020s; the new fixture verifies irregular timestamp inventory preservation, required added keys, no inventory mutation and rejection of truncated frame counts. Additive receipt: `after-source-preparation-v2-UI-PTS-contract.json` (SHA `9f708704567ea68f6f6488bea239c3d0e45c0a9b272f23c1434649acfccabfe7`). Current helper SHA: `9859a61412192fe2252fc3bad7b83aefe98d054fee207b2f22588eba249cbca4`. The earlier five-fixture receipt remains preserved. No private pixels, NN or plan freeze were executed for this update.
