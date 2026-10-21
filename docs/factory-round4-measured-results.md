# Round4 actual factory before/after benchmarks

20 October2026. **No accuracy upgrade and no strict sustained-runtime pass.** All three new policies remain at10/11 meaningful localizations; no model is promoted. These results precede the separate Round5 thin-structure experiment.

## Same exposed original80-view localization comparison

Meaningful localization requires predicted native mask IoU≥.1 with the publisher mask. All11 defective views and every unaffected-view alarm have exact PNG prediction/GT masks and a four-panel visualization. The80 views represent ten defective physical items;69 unaffected views are not a wholly-good-product test population. This repeatedly exposed development set is not a blind holdout.

| Model | Localized defect views | Unaffected-view alarms | ≥50% GT coverage | Pixel recall | Pixel precision | Mask IoU≥.5 region hits |
|---|---:|---:|---:|---:|---:|---:|
|Retained v10h|10/11 (90.9%)|5/69|7/11|54.44%|68.73%|5/11|
|Prior C200|10/11 (90.9%)|1/69|9/11|75.40%|61.54%|4/11|
|Round4 step600 PRIMARY|10/11 (90.9%)|2/69|8/11|71.57%|61.41%|5/11|
|Round4 step600 SECONDARY|10/11 (90.9%)|2/69|8/11|71.57%|61.41%|5/11|
|Round4 step300 POST-CAL exploratory|10/11 (90.9%)|1/69|8/11|70.62%|61.06%|4/11|

All three new policies miss `kos12/Part5.jpg` completely (pixel recall0 and IoU0). The larger real-data transfer did not solve this failure. PRIMARY/SECONDARY worsen C200 alarms1→2; the exploratory step300 matches1 alarm while reducing aggregate pixel recall75.40→70.62%. No result exceeds90.9% meaningful recall.

## Fresh paired complete-profile speed

All project fits/exports/NN/encoding/tests had exited; inference station and browser video playback were stopped. Ordinary host services were retained. Same80 predecoded CAL views×3 repeats/model; model order rotates by view/repeat. Every first timing and outlier remains included. These are serial full-profile calls, not camera/runtime throughput.

| Model | Calls | FPS | Mean ms | P95 ms | P99 ms | Max ms | First64 candidate capacity FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
|Retained v10h|240|24.841|40.26|54.91|63.28|80.36|Historical comparator|
|Prior C200|240|19.344|51.70|68.73|78.67|82.80|Historical comparator|
|Round4 step600 PRIMARY|240|19.193|52.10|64.74|84.71|151.05|19.783|
|Round4 step600 SECONDARY|240|19.249|51.95|65.48|75.13|106.16|17.543|
|Round4 step300 POST-CAL exploratory|240|20.426|48.96|60.47|79.58|186.02|20.477|

## First actual16FPS/60second replay attempts

Fixed order: v10h→C200→PRIMARY→SECONDARY→exploratory. Expected960 submitted/processed observations with exact sequence1…960, zero drops/skips/errors/gaps/late decisions and clean EOF. Deadline100ms. No retries, favorable tail exclusion, clock correction or process suspension. Engineering replay cycles real CAL stills; it is not a physical moving production-line video.

| Model | Captured | Processed | Drops | Late >100ms | Official latency P95 / P99 / max ms | Strict gate |
|---|---:|---:|---:|---:|---|---|
|Retained v10h|960|960|0|2|43.1 / 59.0 / 110.3|FAIL|
|Prior C200|959|950|9|99|129.2 / 151.4 / 267.7|FAIL|
|Round4 step600 PRIMARY|960|954|6|28|67.4 / 142.3 / 183.6|FAIL|
|Round4 step600 SECONDARY|960|950|10|92|134.7 / 157.6 / 179.4|FAIL|
|Round4 step300 POST-CAL exploratory|959|955|4|50|96.2 / 130.7 / 160.8|FAIL|

Every run had0 source skips, inspection errors and reported coverage gaps, and clean EOF. Those counters do not erase missing trace sequences or drops. Two runs captured959 rather than960;959 is retained. v10h had no drops but2 late decisions. C200’s previous960/960/no-late run is historical; its fresh950 processed/9drops/99late result must also be reported. No current run satisfies the strict zero-loss/deadline requirement.

## Artifacts and next experiment

- [Frozen benchmark plan](../output/optimization/factory-round4-root-benchmark/plan-before-any-benchmark.json)
- [Host snapshot](../output/optimization/factory-round4-root-benchmark/host-before-first-benchmark.json)
- [Balanced paired raw calls](../output/optimization/factory-round4-root-benchmark/paired-CAL80.json)
- [Every localized defect and false alarm, including exact masks](factory-round4-all-localization-evidence.md)
- [PRIMARY all localization evidence](../output/optimization/factory-round4-root-benchmark/primary-ALL-original-localization-visuals/)
- [SECONDARY all localization evidence](../output/optimization/factory-round4-root-benchmark/secondary-ALL-original-localization-visuals/)
- [Exploratory all localization evidence](../output/optimization/factory-round4-root-benchmark/lower_alarm_exploratory-ALL-original-localization-visuals/)
- [Next training hypothesis](factory-round5-cldice.md)

The metadata-only host capture initially failed at import because the serving environment lacks `psutil`; no benchmark/model call occurred. It then completed once in the export environment, whose Python/Numpy versions are explicitly recorded in that receipt. Actual capacity, accuracy, paired and runtime commands use the serving `.venv` environment. The fixed benchmark plan/source/model pins remained unchanged after every stage.
