# Synthetic profile evaluation and timing

Run from the project root:

```powershell
.venv/Scripts/python.exe scripts/evaluate_profiles.py --frames 240 --output output/evaluation/profile-baselines.json
```

`--width`, `--height`, `--frames`, `--warmup` and `--source-fps` are configurable. Frames must be at least 64 so every scheduled synthetic condition has enough observations for finite confirmation; dimensions must be at least 240. Smaller dimensions still may fail a profile's fixed demonstration thresholds. No network, paid model, camera, external dataset or AWS calls occur.

The script first runs behavioral checks on known self-authored pixel scenes, then runs the three profiles sequentially using their actual OpenCV pixel processing. Factory checks include a clean board, omitted body, foreign scratch, displaced component and uncalibrated refusal. Freight checks capture each fictional marked page over 12 stable frames and verify the two-page checklist, plus blank-image refusal. Roads checks cover a moving compact dark region and a scene where it is outside the view. Scenario labels are used only by the evaluator; the profiles receive BGR images and `FrameContext`, never scenario labels or indices. These checks are functional regression checks, not accuracy estimates.

Factory timing alternates clean observations with each fault so continuous episode alarms can rearm. Freight timing supplies the first-page sequence beginning at demo index 0, then the second beginning at 120; each settles on a stable view. Roads timing follows the actual moving synthetic scene generator. The script records every status and event, rather than conflating a persistent alert's many frames with unique faults. Event numbers have different meanings across profiles and must not be pooled into a common score.

`captured_monotonic` and `source_time_ms` follow a synthetic source clock at the selected source FPS. **Replay is unpaced**: no sleeps occur. Actual processing duration comes from `perf_counter_ns`, separately from source time. Selecting 30 source FPS therefore does not claim that the machine sustained 30 FPS. The timing interval covers `profile.process`, including annotation rendering, but excludes frame generation, capture, buffering, API transfer, disk evidence, AWS synchronization and physical action. A separate warm-up instance prevents warm-up frames from completing the measured freight checklist or confirming road tracks.

The JSON contains p50/p95/p99, maximum and mean processing time, sum of process durations, process-only FPS, whole-loop wall time, status counts, event keys, final domain state and environment details. Optional `psutil` sampling reports maximum observed RSS of the entire process after frames. This is not a native-memory peak, incremental profile footprint or guarantee against future memory growth. Missing `psutil` leaves RSS null.

## Interpretation boundaries

- All scenes and faults are self-authored synthetic examples. No independent dataset, plant footage, freight rejection corpus or road survey is included.
- Functional passing does not establish recall, precision, calibrated confidence, market benefit or field readiness. Factory fault decisions and road dark-region candidates have different semantics.
- This is serial profile timing on a shared development machine. It does not establish production-line trigger deadlines, vehicle coverage, camera-driver reliability, concurrent throughput or thermal stability.
- Inspect the JSON's environment, sample count, input dimensions and event policy before comparing runs. Host load, power mode, native threads and OpenCV changes affect timing.
- Never present the synthetic source duration as measured time, or process-only FPS as end-to-end camera throughput. Production every-item coverage requires trigger identifiers, physical deadlines and explicit missed/late outcomes outside this script.

The saved JSON is the authoritative observation from a particular run. Repeat the command to produce a fresh baseline; do not compare it to unrelated domain accuracy scores.

## Recorded development-machine run

On 20 October 2026, the command above passed all 10 functional checks and processed 240 frames per profile at 960×540. Python 3.13.5, OpenCV 5.0.0 and NumPy 2.2.6 ran on Windows 11 / Intel64 Family 6 Model 142 Stepping 10, with eight logical CPUs and eight OpenCV threads. Synthetic source timestamps used 30 FPS; the replay was unpaced. RSS was unavailable because optional `psutil` was not installed.

| Profile | Process p50 ms | p95 ms | p99 ms | Maximum ms | Emitted events |
|---|---:|---:|---:|---:|---:|
| Factory | 50.194 | 90.476 | 105.255 | 116.463 | 3 |
| Freight | 10.093 | 17.758 | 20.255 | 22.681 | 2 |
| Roads | 14.195 | 32.369 | 37.747 | 41.403 | 5 |

Factory returned 137 clear, six pending and 97 fault frames. Freight returned 14 needs-evidence, 14 observing, two accepted and 210 duplicate frames; this represents two captured fictional page types. Roads returned 10 pending, 134 review and 96 clear frames. Its five events are heuristic track events, **not five verified defects**: overlap tracks can fragment and the generated scene loops. Factory process-only throughput averaged about 16.95 frames/second, so this run does not support a 30 FPS factory claim. End-to-end performance and physical deadlines remain untested.
