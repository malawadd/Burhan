# Expanded inspections — Burhan 1.3

The app starts with the existing PPE, crack/joint and surface models. **Settings → Camera & Capture → Assistance → Expanded inspection models** is off on a new device and after upgrading a device without that preference. It uses the same preference store as the native camera options screen.

Enable it to add three inspection families: Walls & finishes (12 checks), Brackets & railings (5), and Tile installation (13). In Smart Inspection open Details, choose a check, then close Details. An AI hint is only a proposal: tap it to select, or swipe to dismiss. The selected check determines its detector and instructions. Stop, complete or skip returns to Explore. The switch does not start a detector by itself.

| Bundled model | Selected checks | Status |
|---|---|---|
| PHELE SSDLite320 | Peeling paint; mold-like appearance | Experimental candidates |
| SDNET v2 YOLOv8n512 | Visibly unseated bolt/nut; generic fastener-absence appearance | Experimental candidates |
| Wall-stain SSDLite320 | Damp/stained appearance | Experimental candidates |
| No new detector | Remaining 25 checks, including all 13 installed-tile checks | Manual evidence and physical inspection |

The three new weights add 30,776,221 bytes before APK compression. All seven model files are bundled and checksum verified: installation does not require a model download. Only one neural network is loaded at a time. Switching to Explore or disabling the expansion releases it. The camera preview is independent of inference; result age and analysis rate describe inspected frames, not preview FPS.

## Field workflow

1. Enable the switch in Settings, then return to the camera. Select one check in Details or accept its hint.
2. Follow the short framing instruction. Look for the Experimental label. Retain a wider location photo and a close-up when appropriate.
3. Save evidence and review the original/annotated pair. Detection on a saved capture uses the actual retained photograph, rather than live-frame boxes.
4. Confirm or dismiss each candidate. Empty detection is not a pass. If a detector fails, the app states the failure and keeps manual capture available; it does not silently run a different model.
5. For tile measurements, create a manual inspection in the Tile installation family and optionally enter the specified width, tolerance, measured width and instrument/reference notes. Measurements can also be recorded from a saved tile photo's Review screen in All findings. These are inspector-entered physical readings, not camera measurements or automatic acceptance.
6. Generate a new report version. The record includes the task/category, experimental status, successfully executed model identity/checksum and operating thresholds. Old report versions remain unchanged after later edits.

Turning the switch off cancels expanded guidance, clears an active expanded check to Explore and preserves evidence. Reopening an old expanded inspection while the switch is off permits manual evidence; it does not reactivate the model.

## Interpretation

Stains do not measure moisture or prove hidden leakage. Mold-like appearance does not identify a species. Publisher-labelled absent fastener appearance does not establish a required missing bolt/nut. Visible unseated appearance does not measure torque, embedment or anchorage. All new models retain documented false alerts, misses and independent-site/video qualification gaps. No expanded category has passed the desktop project's release gates.

The existing legacy checks remain available separately. Adding a guided category is not training a detector. Tile geometry remains a desktop research tool and is not shipped as phone-based physical measurement in this release.

## Backend compatibility

The app keeps `https://merhaba-api.kiyan.finance`, device pairing and the `SMART_INSPECTION_PROVIDER=codex|aws|off` environment switch. New session/turn context adds `expandedInspectionModels` and `installedModelIds`. Old clients omit them and retain the original catalogue. The backend uses registered task/model bindings and rejects disabled expanded proposals. Model files and camera evidence stay local; optional smart planning still sends bounded thumbnails online.

## Verification boundaries

See delivery/evaluation/v13-* and delivery/verification.json for executed results. Retained TRAIN photographs and desktop/browser timing are numerical diagnostics, not independent accuracy or phone qualification. Physical camera performance, thermal behaviour, JNI inference and hardware lifecycle checks must be recorded on an actual phone. Live AWS verification requires configured AWS credentials/model access; transport-contract tests alone do not establish a real Bedrock invocation.
