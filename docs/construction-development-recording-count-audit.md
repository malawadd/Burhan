# Independent construction recording count audit

Independent read-only JSON count/actual-box-IoU audit. No NN, media pixels, model assets or active report/registry edits.

Class-aware greedy in recorded candidate order, strict one-to-one at IoU0.50; evaluated=false frames and unscored predictions excluded from accuracy; unsupported labels counted separately. FP means unmatched model boxes under publisher annotation scope, not independently verified material false alarms.

**PASS**. This verifies saved counts and JSON timeline consistency, not detector quality, real-time speed, pixel segmentation, physical-site independence or Saudi code compliance.

| Evidence | Result | Frames | TP | FN | Unmatched boxes (FP) | Unsupported |
|---|---|---:|---:|---:|---:|---:|
| [construction-crack_10-joint-candidate.json](C:/Users/IAM/Documents/hackathon/opencv/project1/output/construction-crack_10-joint-candidate.json) | PASS | 162 | 136 | 26 | 87 | 0 |
| [construction-crack_89-joint-candidate.json](C:/Users/IAM/Documents/hackathon/opencv/project1/output/construction-crack_89-joint-candidate.json) | PASS | 24 | 0 | 8 | 9 | 0 |
| [construction-spalling_35-baseline.json](C:/Users/IAM/Documents/hackathon/opencv/project1/output/construction-spalling_35-baseline.json) | PASS | 72 | 0 | 0 | 0 | 72 |
| [construction-crack_10-baseline.json](C:/Users/IAM/Documents/hackathon/opencv/project1/output/construction-crack_10-baseline.json) | PASS | 162 | 14 | 148 | 157 | 0 |
| [construction-crack_89-baseline.json](C:/Users/IAM/Documents/hackathon/opencv/project1/output/construction-crack_89-baseline.json) | PASS | 24 | 0 | 8 | 8 | 0 |
| [construction-round3-diagnostic-evcivil-yolov8n-r3-spalling_54.json](C:/Users/IAM/Documents/hackathon/opencv/project1/output/construction-round3-diagnostic-evcivil-yolov8n-r3-spalling_54.json) | PASS | 341 | 65 | 185 | 93 | 0 |
| [construction-round3-diagnostic-evcivil-yolov8n-r3-spalling_75.json](C:/Users/IAM/Documents/hackathon/opencv/project1/output/construction-round3-diagnostic-evcivil-yolov8n-r3-spalling_75.json) | PASS | 84 | 0 | 77 | 3 | 0 |
| [construction-round3-diagnostic-evcivil-yolov8n-r3-crack_129.json](C:/Users/IAM/Documents/hackathon/opencv/project1/output/construction-round3-diagnostic-evcivil-yolov8n-r3-crack_129.json) | PASS | 104 | 4 | 67 | 15 | 0 |
| [construction-round3-diagnostic-spalling_54.json](C:/Users/IAM/Documents/hackathon/opencv/project1/output/construction-round3-diagnostic-spalling_54.json) | PASS | 341 | 0 | 250 | 127 | 0 |
| [construction-round3-diagnostic-spalling_75.json](C:/Users/IAM/Documents/hackathon/opencv/project1/output/construction-round3-diagnostic-spalling_75.json) | PASS | 84 | 0 | 77 | 0 | 0 |
| [construction-round3-diagnostic-crack_129.json](C:/Users/IAM/Documents/hackathon/opencv/project1/output/construction-round3-diagnostic-crack_129.json) | PASS | 104 | 17 | 54 | 16 | 0 |
| [construction-development-evcivil-yolov8n-r4-head-640-gain-crack_89.json](C:/Users/IAM/Documents/hackathon/opencv/project1/output/construction-development-evcivil-yolov8n-r4-head-640-gain-crack_89.json) | PASS | 24 | 0 | 8 | 4 | 0 |
| [construction-development-evcivil-yolov8n-r4-head-640-gain-crack_129.json](C:/Users/IAM/Documents/hackathon/opencv/project1/output/construction-development-evcivil-yolov8n-r4-head-640-gain-crack_129.json) | PASS | 104 | 0 | 71 | 49 | 0 |
| [construction-development-evcivil-yolov8n-r4-head-640-gain-spalling_75.json](C:/Users/IAM/Documents/hackathon/opencv/project1/output/construction-development-evcivil-yolov8n-r4-head-640-gain-spalling_75.json) | PASS | 84 | 0 | 77 | 1 | 0 |
| [construction-development-evcivil-yolov8n-r4-head-640-gain-ros_crack_21.json](C:/Users/IAM/Documents/hackathon/opencv/project1/output/construction-development-evcivil-yolov8n-r4-head-640-gain-ros_crack_21.json) | PASS | 222 | 0 | 442 | 45 | 0 |
| [construction-development-evcivil-yolov8n-r4-real-lowlight-640-gain-crack_89.json](C:/Users/IAM/Documents/hackathon/opencv/project1/output/construction-development-evcivil-yolov8n-r4-real-lowlight-640-gain-crack_89.json) | PASS | 24 | 0 | 8 | 8 | 0 |
| [construction-development-evcivil-yolov8n-r4-real-lowlight-640-gain-crack_129.json](C:/Users/IAM/Documents/hackathon/opencv/project1/output/construction-development-evcivil-yolov8n-r4-real-lowlight-640-gain-crack_129.json) | PASS | 104 | 1 | 70 | 1 | 0 |
| [construction-development-evcivil-yolov8n-r4-real-lowlight-640-gain-spalling_75.json](C:/Users/IAM/Documents/hackathon/opencv/project1/output/construction-development-evcivil-yolov8n-r4-real-lowlight-640-gain-spalling_75.json) | PASS | 84 | 0 | 77 | 0 | 0 |
| [construction-development-evcivil-yolov8n-r4-real-lowlight-640-gain-ros_crack_21.json](C:/Users/IAM/Documents/hackathon/opencv/project1/output/construction-development-evcivil-yolov8n-r4-real-lowlight-640-gain-ros_crack_21.json) | PASS | 222 | 10 | 432 | 412 | 0 |

Recordings without external evidence JSON were skipped, without granting any accuracy credit: niosh-ppe, bridge-crack, bridge-roi, site-ppe.

[Full machine-readable audit](C:/Users/IAM/Documents/hackathon/opencv/project1/output/evaluation/construction-development-recording-count-audit.json)

Unmatched boxes include partial coverage that fails whole-instance IoU and predictions outside the available publisher labels. Counts are unchanged; they are not verified material false alarms. Class-negative false-positive rates require a separate completely annotated negative dataset, such as the reserved DACL polygon-negative scope.

Actual encoded container frame counts/timestamps and source pixels are not independently re-decoded by this audit; only their saved declarations are checked against frame records.
