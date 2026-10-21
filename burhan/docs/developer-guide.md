# Burhan developer guide

## Architecture and ownership

All implementation lives in `burhan`. React19/TypeScript/Vite6 implement the shared workspace; Capacitor8 hosts it on Android. `src/native.ts` registers the single native bridge. The native Java Activity uses CameraX1.5.1, ONNX Runtime Android1.23.2, OpenCV4.12 and Room2.8.3. Browser inference runs in a dedicated worker using the lockfile-pinned ONNX Runtime Web (initial build1.30.0). No inference API/service is called.

Shared types are in `src/types.ts`. `InspectionEngine` behaviour lives in `src/engine`; persistence/report/backup adapters live in `src/lib`. Native camera frames stay native; compact results and file references return to JavaScript. Original/annotated evidence is written and fsynced before the capture journal row is inserted. Workspace JSON is chunked in a Room transaction to avoid Android CursorWindow limits. Persisting the shared workspace happens before journal acknowledgement.

Native APIs: scan, loadState, saveState, getPendingCaptures, acknowledgeCaptures, printReport, beginBackup, appendBackup, finishBackup, cancelBackup. Scan receives inspectionId, mode, autoSave, language, followUp and optional findingId. Returned captures preserve evidence IDs and canonical observation metadata. Saved image file:// URLs are converted only for display; backup/report export embeds image contents.

Native CPU baseline remains unaccelerated. Existing desktop GPU results do not qualify an Android backend. Models are SHA256 checked on load. Session loading verifies exact manifest identity, and output shapes are checked while decoding. Failed/diagnostic checkpoints are not shipped.

## First build on Windows

Install Node24+, then run npm ci. Run npm run setup:android to download a private JDK21 and SDK36/build-tools/platform-tools under .tools. It can use an existing sdkmanager as a launcher without changing that installation. SDK licences require review/acceptance. This machine's existing JDK17/SDK34/KetoCut setup stays unchanged.

The Gradle8.14.3 wrapper and AGP8.13 configuration come from Capacitor8. Android local.properties is generated with the private SDK path. All tool environment overrides are process-scoped. The first build downloads Maven dependencies and can take several minutes. Run npm run build:apk; the output is delivery/Burhan-v1.0.0-debug.apk plus SHA256/build metadata.

The debug APK is ARM64 only, minAPI31/targetAPI36. It is signed with the local Android debug identity (normally %USERPROFILE%/.android/debug.keystore). Preserve that file for compatible prototype updates. This key is for testing, not a public commercial release. Do not commit signing keys. Model redistribution/commercial clearance remains unresolved for some included assets; see model-notes.

## Commands

| Command | Purpose |
|---|---|
| npm run dev | Browser development at127.0.0.1:5173; excludes tool/build trees from watching |
| npm run build:web | Typecheck and create dist with local assets |
| npm run preview | Serve production dist, default4173 |
| npm run verify | Types, domain/decoder tests and bundled asset hashes |
| node --experimental-strip-types tests/inference-parity.mjs | Golden Python-output decoder and preprocessing checks |
| node --experimental-strip-types tests/inference-parity.mjs --wasm | Also execute all four models through Web WASM |
| npm run build:apk | Build/sync Android, assemble APK, unit-test task, verify signature/ZIP alignment |
| npm run package:delivery | Create APK+guides field-kit ZIP |

For native instrumentation compilation, set JAVA_HOME/ANDROID_HOME/GRADLE_USER_HOME to the same private tool paths and run android/gradlew.bat -p android :app:assembleDebugAndroidTest. Run connectedDebugAndroidTest only with a physical ARM64 phone attached. No emulator is needed for browser work or APK assembly.

## Model assets

The four ONNX files, manifest, fonts, labels and catalogue are inside assets and copied to dist/native assets. `scripts/prepare-assets.py` refreshes them from the parent project and validates source hashes. It requires the parent Python environment only for this developer refresh; the app itself is independent.

`scripts/sync-runtime.mjs` copies matching offline WASM runtime files after npm installation. Keep runtime JS/WASM versions together. `scripts/verify-assets.mjs` checks all model hashes, catalogue counts, fonts and runtime files. The Arabic title dictionary covers every task/stage. Original upstream catalogue evidence terms and regulatory references are retained as source material, not translated acceptance rules.

## Data, reports and backup

State schemaVersion1 contains projects, inspections, findings, immutable report snapshots and settings. Finding review and issue status are distinct. Reports carry copies of the project/inspection/findings; later edits do not change earlier reports. Evidence preserves per-frame observations, including model SHA256, input size, preprocessing, thresholds and runtime. Merging findings retains this provenance. Domain validation checks required timestamps, nested report snapshots, settings, decisions and references before replacement. UI saving stays disabled after load failure.

Web uses IndexedDB. Android uses Room plus files/burhan-evidence. Model caches are installed under noBackupFilesDir. Backups contain versioned JSON and content-addressed evidence in a ZIP, excluding model weights. Import validates IDs/references, schema and SHA256 images before state replacement. Native ZIP bridge streams512KB chunks; JS compression still uses memory. Compressed import/export maximum200MB; expanded maximum350MB, per-entry100MB. Use smaller workspaces for constrained phones.

Report HTML escapes user input, uses local fonts and a light A4 layout. Native export embeds images/fonts and opens Android PrintManager's Save as PDF dialog; browser export uses a temporary print frame. The archive stores the report snapshot, not a copy of every externally saved PDF. System print completion/cancellation is user-controlled and not claimed as a successful PDF save.

## Verification limits

Numerical fixtures use an original procedural textured image, not a field-accuracy dataset. Golden output decoding is exact; JS/OpenCV resize differences are explicitly reported. Native test source verifies all models execute and Room/journal roundtrips on a real device. Building its test APK only checks compilation. Native camera, thermal, sharing and printing remain pending until actual device testing.

Use docs/device-checklist.md and record actual device/OS, timing, battery, thermals and errors. Do not infer Android speed or reliability from desktop/browser tests.
