# Burhan · برهان

An offline construction inspection workspace with native Android live camera analysis, reviewed findings, corrective actions, inspection history and English/Arabic reports. Dark neobrutalist interface with accessible field controls.

## Start here

- **Install:** copy `delivery/Burhan-v1.0.0-debug.apk` to an ARM64 Android 12+ phone, open it and permit installation from the app you used to open the file. Burhan requests camera permission when you scan.
- **Use:** create a project → new inspection → select stage/task and analysis mode → start live inspection → collect evidence → review findings → complete checks → generate report.
- **Reports:** saved inside Burhan as immutable versions. **Export PDF** opens Android's print dialog: choose **Save as PDF**, A4, and save to your chosen folder.
- **Back up:** Settings → Export backup → save the ZIP outside the app. Uninstalling or clearing app storage removes local inspections and evidence.

Read [the English field guide](docs/user-guide-en.md) or [دليل الاستخدام العربي](docs/user-guide-ar.md). See [build instructions](docs/developer-guide.md), [models and limitations](docs/model-notes.md), and [physical-device checks](docs/device-checklist.md).

## Browser preview — no emulator

Requires Node.js 24 (installed on this machine). From this folder:

```powershell
npm ci
npm run dev
```

Open http://127.0.0.1:5173. The browser runs real ONNX inference locally using WASM; model loading can take time. Camera access requires localhost or HTTPS. Image import works without a camera. Browser and Android workspaces are separate; use backup/restore to transfer them. A production browser build caches assets/models after use for offline access, subject to browser storage limits.

For a lightweight production preview:

```powershell
npm run build:web
npm run preview
```

## Verify and build the APK

```powershell
npm run setup:android
npm run verify
node --experimental-strip-types tests/inference-parity.mjs --wasm
npm run build:apk
npm run package:delivery
```

The Android setup uses a private `.tools` folder. Review SDK licences when prompted. The build uses JDK 21, SDK 36 and the checked-in Gradle wrapper. It generates a signed **debug prototype APK**, verifies its signature/alignment and records its SHA-256 checksum. Keep the debug signing identity for updates; see the developer guide.

## What is included

- Native CameraX camera + local ONNX Runtime/OpenCV inference; image import is secondary.
- PPE, joint-aware crack segmentation and DACL surface-condition models (320 live / 512 detailed imported image).
- 89 original inspection topics plus three direct visual scan tasks, across 16 stages, with Arabic task titles.
- Automatically saved stable candidates, same-frame original/annotated evidence, inspector review and manual findings.
- Room/SQLite native storage, interrupted-camera evidence recovery, IndexedDB browser storage, ZIP backup/restore.
- Open/in-progress/resolved issues, inspector priority, corrective actions and follow-up evidence.
- Versioned English, Arabic and bilingual report records; PDF through platform/browser printing.

## Measured limits

This is a personal field prototype using existing experimental models. It is not a detector for every construction defect. Thin cracks, spalling and several surface conditions have documented misses and false alerts. Tasks without a supported model use manual/instrument evidence. Missing detections never establish a pass; model scores do not establish structural severity or code compliance.

The Python-output numerical decoder checks and all four Web WASM model executions pass. Browser resizing introduces explicitly measured pixel differences; see `tests/fixtures/README.md`. No Android phone was connected during initial development: hardware camera, native inference, thermal endurance, Android PDF/share and native data recovery require the physical-device checklist. Consult `delivery/verification.json` for the exact delivered checks.

Source, assets, Android project, build scripts and guides are all inside this folder. No Python service or internet is needed by the installed app. Build dependency downloads require internet on a fresh development machine.
