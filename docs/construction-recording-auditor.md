# Independent recording-count auditor

`scripts/audit_construction_recording_counts.py` reads registered `config/construction-videos.json` evidence and optional `--evidence-paths`. It uses only Python's standard library: no inference frameworks, video/image pixels, downloaded models, or producer matching functions. Reports and registry inputs are never edited; output paths are rejected if they could overwrite those inputs or each other.

The auditor independently assigns recorded predictions in saved candidate order to same-class original publisher boxes, at actual whole-instance box IoU0.50. Each annotation can match once. It verifies recorded match IDs, exact matched IoU, best unmatched same-class IoU, supported/unsupported scope, scored/unscored state, per-frame counts, declared per-class/global totals, both historical/current unsupported aliases, and precision/recall denominators. Explicit `evaluated:false` frames and `unscored` predictions receive no TP/FN/FP credit. Unsupported annotations remain a separate category. Publisher boundary projection roundoff within1e-6 pixels is tolerated without changing geometry.

Recorded source indices/timestamps must increase. Saved video PTS intervals must match original timestamp intervals. A declared complete original sequence must retain contiguous original indices/counts; encoding and lossless-replay declarations must agree with saved frame counts. **Container bytes and pixels are not independently decoded**: a passing JSON audit establishes arithmetic/geometric and declaration consistency, not verified source authenticity or successful damage detection.

FP means **unmatched model boxes under publisher annotation scope**, including partial coverage that fails full-instance IoU and locations without a matching publisher annotation. These counts are preserved exactly and are not independently confirmed material false alarms. Negative false-positive rates need separately complete annotated negatives. Pixel-mask accuracy is unavailable from box annotations.

Current [audit results](C:/Users/IAM/Documents/hackathon/opencv/project1/docs/construction-recording-count-audit.md) cover five registered damage-evidence reports and all six new historical/adapted whole-video diagnostics: **11 passed, zero count/geometry/timeline inconsistencies found**. This includes severe model failures; passing arithmetic does not convert them into successful inspection. Raw JSON stores every audited input SHA256 and recomputed per-class totals.

Run from the workspace root:

```powershell
.venv\Scripts\python.exe -X utf8 scripts/audit_construction_recording_counts.py --evidence-paths output/construction-round3-diagnostic-spalling_54.json output/construction-round3-diagnostic-spalling_75.json output/construction-round3-diagnostic-crack_129.json output/construction-round3-diagnostic-evcivil-yolov8n-r3-spalling_54.json output/construction-round3-diagnostic-evcivil-yolov8n-r3-spalling_75.json output/construction-round3-diagnostic-evcivil-yolov8n-r3-crack_129.json
```

Exit0 means all supplied evidence passed; exit1 preserves JSON/Markdown failure details when counts/geometry/timeline assertions fail. Context recordings without an evidence JSON are explicitly skipped, with no accuracy credit. Default outputs are `output/evaluation/construction-recording-count-audit.json` and `docs/construction-recording-count-audit.md`; `--output`/`--markdown` can select distinct audit destinations.

Meaningful fixtures in `tests/test_construction_recording_counts.py` cover corrupted global/per-class summaries, false claimed geometry, incorrect reported matched IoU, duplicate matching, wrong classes, unsupported-only cases, unscored negatives/unevaluated frames, missing/malformed summaries, complete-sequence index/count errors, PTS drift, floating boundary roundoff and input-overwrite prevention. **17 tests passed in0.34s**; no NN or media access occurred.
