# Round5 measured results

**The requested joint improvement was not achieved.** R5 still meaningfully localized10/11 defect views and raised one alarm among69 unaffected views, matching C200 on those counts. It increased foreground pixel recall but reduced precision and strict region overlap. It passed average complete-profile capacity; the fresh full-runtime trial dropped five frames and recorded25 late events. No promotion or production qualification.

## Accuracy on the same original80 views

The original80 comparison is repeatedly exposed development data. Baseline accuracy below comes from retained audits; their exact ordered source-image and GT hashes match the new R5 audit. Baseline weights/metadata are the same hashes used in the fresh timing trials.

| Model | Meaningful view recall (IoU≥.1) | Normal-view alarms | ≥50% GT coverage | Pixel recall | Pixel precision | Micro IoU | Mask-region matches IoU≥.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v10h | 10/11 | 5/69 | 7/11 | 54.435% | 68.728% | 0.436291 | 5/11 |
| C200 | 10/11 | 1/69 | 9/11 | 75.404% | 61.536% | 0.512494 | 4/11 |
| Round5 | 10/11 | 1/69 | 9/11 | 79.448% | 58.754% | 0.510021 | 3/11 |

R5 has0 abstentions and still completely misses kos12/Part5. Against C200, pixel recall rises75.404%→79.448%, precision falls61.536%→58.754%, micro IoU falls.512494→.510021 and strict mask-region matches fall4/11→3/11. Supplementary actual runtime-box matches stay9/11 at IoU≥.5; boxes do not reverse the pixel result.

No completely normal products occur in this comparison:69 unaffected views belong to ten defective items. Good-product false-rejection rate is unmeasured. The zero-alarm CAL80 result did not transfer to the exposed original80.

## Fresh speed and full runtime

The first64 complete-profile normal-TRAIN calls passed at19.645FPS, mean50.903ms/P9566.983ms/max77.606ms. This excludes capture, queueing, preview and storage and is not the sustained runtime result.

Paired CAL80 used three balanced repetitions:240 timings per model, not240 new accuracy samples.

| Model | Serial full-profile FPS | Mean ms | P95 ms | P99 ms | Max ms |
|---|---:|---:|---:|---:|---:|
| v10h | 23.339 | 42.847 | 59.879 | 70.193 | 90.145 |
| C200 | 18.063 | 55.362 | 74.557 | 93.212 | 126.728 |
| R5 | 17.978 | 55.622 | 77.217 | 90.023 | 142.048 |

Fresh runtime order was v10h→C200→R5. Each replay requested16FPS for60 seconds on the same varied CAL80 sequence, preview6FPS, deadline100ms. All first results/tails were retained.

| Model | Captured / processed | Dropped | Official late events | Processed / full uptime FPS | Host P95 ms | Host P99 ms | Host max ms | Strict gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| v10h | 960 / 960 | 0 | 6 | 15.947 | 42.606 | 87.368 | 131.609 | FAIL |
| C200 | 960 / 956 | 4 | 27 | 15.880 | 61.605 | 135.659 | 164.310 | FAIL |
| R5 | 960 / 955 | 5 | 25 | 15.864 | 63.028 | 120.899 | 201.802 | FAIL |

All three fresh runtime trials failed the predeclared no-drop/no-late gate. Average processed rate exceeds15FPS, but this does not guarantee inspection of every captured exposure or a production deadline. All three report0 source-skipped/inspection-error/missed-unit/coverage-gap counters; those counters do not erase the explicit dropped-frame counts. Official late counters and traced over100ms rows are both retained in the raw summary; they are not silently substituted for each other.

Project fits, inference, encoding and tests were stopped for these measurements; ordinary host services were uncontrolled. The host snapshot is pinned. Queue/profile/GPU-stage traces are descriptive and do not prove a low-level cause. This is clocked real-still replay, not physical moving-line validation.

## Reproducible artifacts

[Training, sources, all three candidates and calibration](C:/Users/IAM/Documents/hackathon/opencv/project1/docs/factory-round5-cldice.md)

[All11 defects and the normal false alarm](C:/Users/IAM/Documents/hackathon/opencv/project1/docs/factory-round5-localization-evidence.md)

[Measured summary and raw-report/source/model SHA256 pins](C:/Users/IAM/Documents/hackathon/opencv/project1/output/optimization/factory-round5-root-benchmark/measured-summary-and-evidence-pins.json)

[Actual mask audit](C:/Users/IAM/Documents/hackathon/opencv/project1/output/optimization/factory-round5-root-benchmark/winner-original.json)

[Fresh paired timing](C:/Users/IAM/Documents/hackathon/opencv/project1/output/optimization/factory-round5-root-benchmark/paired-CAL80.json)

[Actual R5 runtime and every trace row](C:/Users/IAM/Documents/hackathon/opencv/project1/output/optimization/factory-round5-root-benchmark/winner-runtime.json)

Summary SHA256: `75e2f68158ca5af120f5c34a5ba0c1fd177ba9f88767f5d20b10171b551e8379`. No new neural forward, policy selection, retry or model edit was performed by this summary.
