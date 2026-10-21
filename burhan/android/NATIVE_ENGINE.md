# Burhan native Android engine

The Android build targets API 36, supports API 31 and newer, and packages ARM64 libraries only. Camera operation and inference do not need a server. Native model execution and sustained camera performance require verification on a physical phone; compilation is not evidence of field accuracy or frame rate.

## Bridge contract

`MainActivity` registers `BurhanNativePlugin` as `BurhanNative` before loading the shared interface.

| Method | Purpose |
|---|---|
| `scan({inspectionId, mode, autoSave, language, followUp?, findingId?})` | Opens CameraX inspection; resolves with durable pending captures when closed |
| `getPendingCaptures({inspectionId?})` | Recovers the capture journal after interrupted sessions |
| `acknowledgeCaptures({ids})` | Deletes journal records only after the shared repository commits them |
| `loadState()` / `saveState({json})` | Reads/writes the shared version-1 application state in Room |
| `printReport({html, jobName})` | Opens Android printing; choose **Save as PDF** |
| `beginBackup({filename})` | Creates a private temporary ZIP and returns `{id}` |
| `appendBackup({id, base64})` | Appends a bounded chunk (512 KiB binary recommended) |
| `finishBackup({id})` | Flushes the ZIP and opens Android sharing |
| `cancelBackup({id})` | Removes an unfinished ZIP |

Capture records contain `inspectionId`, `followUp`, optional `findingId`, `evidence`, and `observations`. Evidence paths use `file://` and remain private to the app. Use `Capacitor.convertFileSrc` when displaying them in the WebView. Backups must embed the bytes rather than preserve device-specific file paths.

State is split into 128 Ki-character chunks in one Room transaction, preserving UTF-16 surrogate pairs. This avoids Android cursor-window limits for image-heavy restored backups. Capture journal rows are separate from state rows. Saving state does not acknowledge captures.

## Camera and inference

`ScanActivity` uses CameraX `Preview` and `ImageAnalysis` in one viewport, a rear camera, RGBA analysis and `STRATEGY_KEEP_ONLY_LATEST`. It respects the analysis crop rectangle and physically rotates pixels before preprocessing. Analysis requests 640×480; device negotiation and viewport cropping determine final evidence dimensions. Preview and annotations fit the same source geometry.

Frames remain native. There is one analysis worker, a latest-frame reference and an overlay reference. Old frames become garbage-collectable; they are not recycled while readers could still be drawing or saving them. Captures keep files and small observation records, not Java bitmap objects.

The CPU engine verifies the model SHA-256 at session load. Packaged assets are copied into private model storage for ORT. Expected manifest IDs are `ppe`, `crack`, `surface320`, and `surface512`.

- PPE/crack: RGB float32 NCHW, bilinear letterbox with fill 114, per-class NMS. Joint predictions are not treated as crack findings.
- Crack masks: coefficient/prototype logits, bilinear resize/crop/projection, zero-logit threshold, clipped to the detector box. The bitmap is cropped to its bounding box to bound memory.
- DACL: BGR uint8 NCHW; normalization remains inside ONNX. Each independent sigmoid channel is thresholded separately. Components use 8-connectivity, a four-grid-pixel minimum and the existing damage-class subset. Retain up to twenty components per class and one hundred results overall.
- DACL mask rendering: preserve compact component grids and draw nearest-projected pixel runs using integer ceil boundaries. This matches the OpenCV projection without allocating a full source-size bitmap for every component.
- Surface 320 is used live. The detailed 512 checkbox applies to imported photos.

The optional observation wire mask is omitted from native journal JSON. Actual masks are present in the saved annotated evidence. Canonical observation labels remain unchanged; Arabic overlays use translated display labels.

Automatic capture requires three associated observations within two seconds, class equality and IoU above 0.35. One evidence capture is allowed per second; the short-lived track is saved once. This is approximate deduplication, not persistent identity tracking. Slow inference can prevent stable automatic capture; manual Capture remains available.

Both original and annotated evidence come from the exact same analysed source bitmap. The journal is committed after both JPEG files are written. Manual, imported and follow-up captures use the same journal. Follow-up scanning disables automatic evidence capture.

Analysis pauses in the background. CameraX follows the activity lifecycle, ORT sessions close on stop, and sessions reload when needed. A severe Android thermal signal pauses analysis until the phone cools. Failed analysis can still preserve a requested frame as manual evidence. Results older than three seconds are cleared from the live overlay. Paused/imported views display the corresponding frozen source frame.

## Physical-device checks

Connect a real ARM64 phone with USB debugging enabled. From this folder, using the bundled JDK/SDK environment documented in the main build guide:

```powershell
.\gradlew.bat connectedDebugAndroidTest
```

`NativeStateTest` checks large Arabic/emoji state round trips, capture acknowledgement, and bundled-model execution contracts on the device. Its neutral-frame model check does not evaluate real construction accuracy. Also perform the main physical-device checklist: rotations and overlay alignment, permissions, backgrounding, process termination, original/annotated correspondence, airplane mode, PDF export, backup restore, and a 20-minute camera session with thermal and latency observations.

No physical-device result should be marked passed until it has actually been observed.
