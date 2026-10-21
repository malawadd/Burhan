# Shared media worker integration verification

Verified 2026-10-20 with stub source/profile frames; no camera, trained-model inference or performance timing was involved in this regression. These tests validate concurrency and lifecycle behavior rather than camera or detector accuracy.

The runtime uses the actual `MediaWorker` and real JPEG encoder. The test holds the encoder behind a synchronization event while capture and inspection continue. At least 8 additional inspection frames complete while the encoder remains blocked. The pending image byte budget and evidence queue remain bounded; rejected event images retain their finding metadata through the normal storage path.

While the encoder is blocked, both stop and station switching report `Media worker is still closing`. The runtime keeps the original profile/session/media owner and creates no replacement profile. After release, queued images drain with their original factory/session attribution, stop succeeds, and a new freight session starts normally.

Three concrete lifecycle corrections accompany the regression:

- A successful stop retry clears resolved source/document/media closing warnings. Unrelated source and inspection errors remain available for diagnosis.
- Once inspection workers have stopped, pending accepted factory triggers become `inspection_missed` before waiting for auxiliary image work. A blocked encoder cannot delay their terminal outcome.
- An explicit stop no longer enters the EOF finalization path when the closed capture buffer wakes inference. This keeps the trigger outcome reason `station_stopped` instead of incorrectly calling it source EOF.

Focused verification command:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_runtime.py -q -k 'blocked_media or missing_file or background_profile'
```

Recorded result: **4 passed, 8 deselected**, one existing third-party Starlette/AnyIO deprecation warning. The shortened join timeout in the tests affects only test duration; production keeps its 3-second media close bound. The native encoding work is released in a `finally` block so test failures do not leave a worker behind.
