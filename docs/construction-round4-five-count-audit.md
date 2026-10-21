# Independent construction recording count audit

Independent read-only JSON count/actual-box-IoU audit. No NN, media pixels, model assets or active report/registry edits.

Class-aware greedy in recorded candidate order, strict one-to-one at IoU0.50; evaluated=false frames and unscored predictions excluded from accuracy; unsupported labels counted separately. FP means unmatched model boxes under publisher annotation scope, not independently verified material false alarms.

**PASS**. This verifies saved counts and JSON timeline consistency, not detector quality, real-time speed, pixel segmentation, physical-site independence or Saudi code compliance.

| Evidence | Result | Frames | TP | FN | Unmatched boxes (FP) | Unsupported |
|---|---|---:|---:|---:|---:|---:|
| [construction-round4-cpu-recovery-v1-crack_89.json](C:/Users/IAM/Documents/hackathon/opencv/project1/output/construction-round4-cpu-recovery-v1-crack_89.json) | PASS | 24 | 0 | 8 | 5 | 0 |
| [construction-round4-cpu-recovery-v1-crack_129.json](C:/Users/IAM/Documents/hackathon/opencv/project1/output/construction-round4-cpu-recovery-v1-crack_129.json) | PASS | 104 | 12 | 59 | 43 | 0 |
| [construction-round4-cpu-recovery-v1-spalling_75.json](C:/Users/IAM/Documents/hackathon/opencv/project1/output/construction-round4-cpu-recovery-v1-spalling_75.json) | PASS | 84 | 0 | 77 | 17 | 0 |
| [construction-round4-cpu-recovery-v1-ros_crack_21.json](C:/Users/IAM/Documents/hackathon/opencv/project1/output/construction-round4-cpu-recovery-v1-ros_crack_21.json) | PASS | 222 | 53 | 389 | 607 | 0 |
| [construction-round4-cpu-recovery-v1-spalling_76.json](C:/Users/IAM/Documents/hackathon/opencv/project1/output/construction-round4-cpu-recovery-v1-spalling_76.json) | PASS | 249 | 47 | 160 | 231 | 0 |

Recordings without external evidence JSON were skipped, without granting any accuracy credit: .

[Full machine-readable audit](C:/Users/IAM/Documents/hackathon/opencv/project1/output/evaluation/construction-round4-cpu-recovery-v1/videos/root-independent-five-count-audit.json)

Unmatched boxes include partial coverage that fails whole-instance IoU and predictions outside the available publisher labels. Counts are unchanged; they are not verified material false alarms. Class-negative false-positive rates require a separate completely annotated negative dataset, such as the reserved DACL polygon-negative scope.

Actual encoded container frame counts/timestamps and source pixels are not independently re-decoded by this audit; only their saved declarations are checked against frame records.
