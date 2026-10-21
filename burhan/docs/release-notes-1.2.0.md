# Burhan1.2.0 — release and verification

Package `com.burhan.site`, version code **4**. Same signing certificate as1.1.1; install as an update to retain the workspace.

## Changes

Explore/selected-check state now controls the actual detector, selection indicator, title and instructions together. Deselect, Stop, Complete and Skip return to Explore. AI proposals require a tap. One subtle hint supports swipe and accessible dismissal, scene suppression and task-bound guidance. Camera & Capture groups16 independent per-device preferences. Android includes hardware capture tools, higher-detail photo capture, Quick Review, Undo, transactional Retake and original-preserving markup/sharing. The browser supports shared selection/settings/image review and labels hardware-only features.

## Executed checks

- 32 app/domain/backup/provider-contract tests passed, including Retake targeting the latest photo of the current check even when another check has a newer image.
- 20 backend tests passed: authentication, idempotent events, provider normalization, cancellation, quotas and persisted history.
- 25 native JVM/Robolectric tests passed, including14 actual Material camera layouts/interactions, three preference tests covering every switch, scene filtering, Room replacement rollback/ownership and markup baseline preservation.
- Android lint: **0 errors /38 warnings**. Instrumentation APK compiled; it was not executed on a phone.
- APK v2 signature and16KB ZIP/ELF alignment passed. Existing signing identity retained.
- Final packaged shared settings JSON and web bundle compared byte-for-byte with the source/build output.
- Real Codex two-turn checks passed through both an isolated backend evaluation and actual Edge/public-domain workflow. A proposal stayed inactive until the tap; subsequent Stop and swipe events were recorded. Actual local WASM detection ran only after selection; deselect cleared its overlay.
- Real browser import, Undo, markup, share download, Settings persistence and round trips, Retake success and injected storage-failure recovery were exercised. A rapid first-import regression was found and fixed; save confirmation now follows durable storage. Header targets were checked at390px; modal sheets render above Undo.

## Limits of this record

No physical phone was attached. CameraX still/analysis combinations, device rotation, volume keys, actual pinch/focus, sensor level, haptics, thermal behavior, native share receivers and real camera performance need the [physical-device checklist](device-checklist.md). Native screenshots use a blank camera fixture. Browser image tests and numerical fixtures do not establish defect accuracy. Live AWS model access/deployment still needs configured credentials; SDK contract tests are not live Bedrock inference.

## Files

- APK: `delivery/Burhan-v1.2.0-debug.apk`
- SHA-256: `2ac237dbe7cf00ff93e265a6a9e08376b0cdd8f3398f909782e4231de0798ab1`
- Bytes: `207745653`
- Final bundled web entry: `assets/public/assets/index-BGxo76zd.js`
- Machine-readable record: `delivery/verification.json`
- Actual provider/session traces, browser screenshots and native View renders: `delivery/evaluation/`
- Settings and tools: [Camera & Capture](camera-settings.md)
- [Architecture](architecture-v12.md), [English field guide](user-guide-en.md), [دليل الاستخدام](user-guide-ar.md), [backend/provider setup](smart-inspection.md)

Build record: 2026-10-21T05:05:13.790Z.
