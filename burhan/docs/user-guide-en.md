# Burhan field guide

## Install and start

1. Transfer `Burhan-v1.2.0-debug.apk` from the delivery folder to an ARM64 Android phone running Android 12 or newer. A 2025 midrange phone or newer is the intended class; actual speed varies.
2. Open the APK. Android may ask you to allow installation from the file manager/browser you used. Install Burhan.
3. Open Burhan. Choose English or العربية. Set your inspector name in Settings.
4. Offline detection needs no account or laptop connection. Models are bundled. Smart Inspection requires a paired backend running on your PC or a deployed host; its agent uses online inference. Camera permission is requested when opening a live inspection.

Updating with the same package and signing key preserves data. Export a backup first. Uninstalling, clearing app storage or switching signing keys can require reinstalling and deletes local data.

## Smart Inspection — start with the camera

1. Start the backend using `backend/start.ps1 -Install`. Keep its computer and tunnel running. Obtain the pairing code from its terminal or `backend/data/pairing.json` (valid one hour after startup).
2. In **Settings → Smart Inspection**, keep `https://merhaba-api.kiyan.finance`, enter the code, and tap **Pair device**. You should see the selected provider and connection status.
3. Tap **Start Smart Inspection** on the home screen. A site walkthrough is created automatically. On Android the paired guide begins in the camera screen; the top guide indicator shows its state. In the browser, start the assistant after reading its snapshot-transmission notice.
4. Point the camera at the area. The assistant proposes a scene and relevant checks, selects supported detectors, and asks for useful views or manual checks. Live detection continues while it thinks.
5. On Android, the main screen shows one **Current check** and a short next instruction. Tap **Details**, **Change**, the guide indicator, or **⋮** for full instructions, alternative checks, **Complete current check**, **Skip current check**, correction, Stop check and guide pause. Browser Details offers the same shared check-selection controls. Pausing the guide stops new remote snapshots while local detection continues.
6. Save evidence and review findings as usual. The recorded guidance is included in the inspection's report; it is advice rather than a verified defect or acceptance decision.

Without a camera, start the assistant and **Import image** in the browser. Imported-image processing can exercise the same backend and installed detectors. Online planning may take tens of seconds; moving away or correcting the scene invalidates obsolete guidance. See [connection and provider troubleshooting](smart-inspection.md).

## Create a project and inspection manually

Select **New project**, enter the site name and address, and save. Select **New inspection**, choose the project and enter a title, room/floor/element and inspector. Select a construction stage and task.

The catalogue covers 89 construction topics plus three direct visual tasks. It does not contain 89 working detectors. Read the mode and automatic/manual indication:

| Mode | What it observes |
|---|---|
| Cracks | Experimental visible crack segmentation, with a model trained to distinguish joints |
| Concrete & surfaces | Registered surface-condition classes, including spalling, rust, exposed rebar, wet spots, efflorescence and honeycomb appearance; measured quality varies |
| PPE observations | The installed PPE model's classes; insufficient for dependable safety monitoring |
| Manual evidence | Camera/photos, notes and inspector decisions without automatic analysis |

In manual inspections the mode is your explicit choice. In Smart Inspection the assistant proposes an available check; only your selection activates its detector. Stop or deselect returns to Explore. Selecting a checklist topic does not create a detector for unsupported work.

## Live camera

Tap **Start live inspection** and allow camera access. Android opens the native camera screen. Point the rear camera at the element and let the model load. Keep the phone steady, move slowly and use suitable lighting. The camera is the main view. Tap the preview to request focus and exposure at that point. The current check identifies whether you are recording manual evidence or using an on-device detector. Measured analysis frequency and technical status are available in **Details**.

Use the yellow **Capture evidence** button to save evidence. A confirmation and the **Review** count acknowledge a successful save. **Pause camera / Resume camera** freezes or resumes the analysed view; it is separate from pausing the guide. Flash and zoom controls sit on the camera view. Native source evidence comes from the cropped/rotated analysis frame (normally 640×480), not a full-resolution sensor photograph.

With automatic saving enabled, a candidate must match across three analysed observations within two seconds. Saving is limited to one evidence capture per second. Slow inference may never satisfy the stability rule: use Capture. Tracking only suppresses short-term duplicates; review and merge duplicates yourself.

Original and annotated images are saved from the same analysed frame. Results belong to their recorded capture time; they can lag the moving live view. Pausing freezes the analysed view. Severe thermal status pauses analysis until the phone cools. Backgrounding stops processing; resume explicitly when returning.

Tap **Review** (or Back) to return directly from the native camera to the inspection and its saved findings. This exits capture; it does not complete or approve the inspection. Evidence is journalled before returning to the shared interface; interrupted sessions recover saved captures when the app reloads. An interrupted follow-up capture keeps its issue association when available.

## Import an image

Use **Import** inside the native scanner or **Import image** in the browser scanner. Surface mode offers the slower 512px detailed model for imported images (Android: **Details → Detailed surface photo analysis**); live surface analysis uses 320px. Imported native images may be downsampled to bound memory; retained original evidence means the actual analysis input, not the untouched original phone file.

In the finding editor, **Add photos** attaches supporting evidence without automatically analysing it. Use the scanner's image import when analysis is required.

## Review findings and corrective work

Each AI suggestion starts as **Needs review**. Confirm or dismiss it. Set inspector priority, notes, corrective action and issue status. A model score is not the probability of structural failure or a measurement of severity.

Add manual findings for anything the model missed. Select related findings and **Merge selected** to preserve their evidence in one issue. Finding edits save automatically. Check for the save-error message if storage fails.

For repair follow-up, reopen the finding, add photos or select **Capture follow-up**, record work and mark the issue In progress or Resolved. Existing reports remain unchanged; generate a new report version after edits.

Complete each checklist item as Not checked, Acceptable, Issue or Not applicable based on your own evidence. The app never automatically accepts work. Acknowledge outstanding checks/unreviewed candidates if completing an incomplete review; they remain visible in the report.

## Reports and history

Choose English, Arabic or bilingual report output, then **Generate report**. The report record is saved in Reports and can be reopened without repeating inference. Versions preserve the findings and evidence as they stood when generated.

Select **Export PDF**. On Android, choose **Save as PDF** in the print destination list, choose A4 and save. In the browser, choose Save as PDF in its print dialog. The app considers the report record saved separately from whether you finish or cancel the PDF dialog. Exported files are stored where you choose; use your file manager to share them.

Report content includes site, element, inspector, scope, reviewed/unreviewed/dismissed findings, notes, priorities, corrective actions, original/annotated/follow-up images and method limits. Reports support observations; they do not certify compliance.

## Backups and device transfer

Settings → **Export backup** produces a portable ZIP containing projects, inspections, findings, report snapshots and evidence. Save/share the ZIP somewhere outside Burhan. Native export streams chunks through the Android bridge; ZIP preparation itself still uses memory. Keep archives below 200MB. Very large archives may exceed available phone memory.

Settings → **Restore backup** → select the ZIP → confirm replacement. Restore replaces the current workspace rather than merging it. Back up the current workspace first. The app validates the schema and evidence hashes before replacement. Browser and Android do not synchronise automatically; backup/restore is the transfer mechanism.

If data cannot be read, Burhan displays a persistent error and does not overwrite the saved workspace. Retry without clearing storage. Keep the last working backup.

## Troubleshooting

- **Camera denied:** allow Camera in Android Settings → Apps → Burhan → Permissions. Image import remains available.
- **Slow analysis:** wait for the initial model load; keep still; capture manually. Detailed surface512 is deliberately slower. No model is guaranteed to run at a fixed rate.
- **Phone hot:** stop or pause and let it cool. The native engine pauses at severe thermal status.
- **No detection:** this is not a pass. Use close review and manual evidence; consult model limitations.
- **Save failure/low storage:** stop collecting evidence, free space outside the app and export existing reports/backup. Never clear Burhan's data to recover space without a backup.
- **Browser camera unavailable:** use localhost/HTTPS and grant permission. A plain LAN HTTP address is insufficient.
- **Android PDF unavailable:** confirm a system print service exists and use Save as PDF. Cancelling printing does not delete the report record.
- **Model checksum error:** do not use the affected mode; preserve data and reinstall a verified APK only after backup.


## Camera & Capture in1.2

Start in Explore, then tap a check or the small AI hint to inspect it. Tap the active check again or choose Stop check to return to Explore. Complete and Skip also return there. The assistant proposes checks and never starts another detector without your choice. Swipe a suggestion away to dismiss; use Details → Suggest again to bring suggestions back.

Open Settings → Camera & Capture or camera options for independent tools. Quick Review lets you continue the camera, retake the latest current-check photo, or open all findings. Undo lasts five seconds for the newly saved photo. Higher-detail photos, arrows/circles and photo sharing are optional and initially off. [Full settings and review guide](camera-settings.md).


## Optional expanded inspections (1.3)

Settings → Camera & Capture → Assistance → Expanded inspection models is OFF by default. Enable it to show Walls & finishes, Brackets & railings and Tile installation. Five checks have experimental local detectors; the remaining 25 are guided manual checks. Choose a check in camera Details; a hint never starts detection until you tap it. Turn the switch off to return an active expanded check to Explore while keeping evidence.

Tile width/tolerance fields record your physical readings, not camera measurements. Enter them when creating a tile inspection, or open a saved tile photo in All findings → Review → Record physical measurements. Reports retain experimental model identity and entered values. Review all candidates; no detection is not a pass.
