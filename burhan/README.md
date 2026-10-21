# Burhan · برهان

Construction inspection with offline Android detection, an adaptive camera guide, reviewed findings, corrective actions and English/Arabic reports. Smart Inspection uses a paired backend to interpret scenes and decide which evidence to collect next. Dark neobrutalist interface with accessible field controls.

## Start here

- **Install:** copy `delivery/Burhan-v1.3.0-debug.apk` to an ARM64 Android 12+ phone. Keep the previous installation to preserve data; this update uses the same package/signing identity. Export a backup first.
- **Smart Inspection:** start `backend/start.ps1 -Install`, obtain the pairing code from its terminal, then pair in Burhan Settings using `https://merhaba-api.kiyan.finance`. Tap **Start Smart Inspection**, start the assistant and point the camera. See [the Smart Inspection guide](docs/smart-inspection.md).
- **Manual/offline use:** create a project → new inspection → select stage/task and analysis mode → collect evidence → review findings → complete checks → generate report. This remains available without a backend.
- **Reports:** saved inside Burhan as immutable versions. **Export PDF** opens Android's print dialog: choose **Save as PDF**, A4, and save to your chosen folder.
- **Back up:** Settings → Export backup → save the ZIP outside the app. Uninstalling or clearing app storage removes local inspections and evidence.

Read [the English field guide](docs/user-guide-en.md) or [دليل الاستخدام العربي](docs/user-guide-ar.md). See [build instructions](docs/developer-guide.md), [models and limitations](docs/model-notes.md), and [physical-device checks](docs/device-checklist.md).

Version 1.2.0 adds explicit Explore/check selection, tappable and swipeable AI hints, and independent Camera & Capture preferences. [Settings and camera tools](docs/camera-settings.md). The native camera keeps one current check, a short next instruction, yellow Capture evidence, and a saved-evidence Review count. Full guidance, alternative/installed checks and assistant controls open in a Material bottom sheet. Camera pause and guide pause are labelled separately. Large text uses stacked actions; landscape places controls beside the camera. [Native camera layout and verification](docs/camera-interface.md).

## One backend, three modes

Edit `backend/.env` and restart the backend:

```env
SMART_INSPECTION_PROVIDER=codex
```

Use `codex` for the installed, signed-in CLI, `aws` for Bedrock Nova 2 Lite, or `off` to disable remote guidance. The phone keeps the same backend URL; no APK rebuild is needed. Codex runs on the developer PC but uses online inference and the signed-in account's usage limits. AWS needs backend credentials/model access; it is not silently used as a fallback. See [backend setup and API documentation](backend/README.md).

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

Source, assets, Android project, backend, build scripts and guides are all inside this folder. Offline detection requires neither Python nor internet. Scene-aware Smart Inspection requires the paired backend and online provider; snapshots are transmitted only while the assistant is enabled. Build dependency downloads require internet on a fresh development machine.


## Optional expansion in 1.3

The original models remain the default. Enable **Settings → Camera & Capture → Expanded inspection models** for five new experimental model checks and 25 manual checks in three families. The three extra ONNX files are bundled for offline use. [Field workflow and limitations](docs/expanded-inspections.md) · [Release notes](docs/release-notes-1.3.0.md) · [Standalone design guide](design/Burhan-Design-Language.html).
