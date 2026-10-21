# Physical Android verification — pending until a phone is attached

Record device model, RAM, Android version, WebView version, available storage, Burhan version/hash and ambient conditions. Target at least one2025 midrange ARM64 phone, plus a newer phone if available.

- Check the redesigned native camera at normal and 200% text: camera, yellow capture, Import and Review remain reachable; Arabic and landscape do not clip labels. Open Details for full guidance and alternatives. Guide pause must not pause the camera; camera pause must not disable the guide control. Tap preview to request focus. Verify count/confirmation only follows a successful durable save.
- Install the delivered APK; verify package com.burhan.site and version. Relaunch offline.
- Pair through the public HTTPS backend, start Smart Inspection, then test all provider settings without rebuilding the APK. Record which providers were actually exercised.
- Walk indoor/outdoor/transitional scenes; correct a wrong suggestion, select a check, complete/skip, and confirm evidence leads to a useful next instruction. Record wrong suggestions and misses.
- While a request is running, change scene, correct context, pause, rotate and background. Old guidance must never change the current detector or attach to new-scene evidence. Verify local detection continues during slow/no network.
- Import a photo, tap an assistant proposal to choose a detector, then capture: original/annotated evidence and model metadata must come from the reanalysed frame. Verify guidance history survives exit, backup/restore and immutable report generation.
- Deny/allow camera permission; verify image import remains available.
- Create project, location and inspection. Scan each included mode; compare a fixed permitted still with the stored Python/browser fixture and inspect source-aligned masks.
- Verify portrait/landscape, crop, zoom and torch. Move the camera and confirm displayed results/evidence timestamps refer to the analysed frame.
- Save manually, trigger stable automatic candidates and verify original/annotated image correspondence. Review duplicates and merge while retaining evidence.
- Pause/resume, open/return from photo picker, lock/unlock phone and background/resume. Verify camera/torch release and no unintended automatic pass.
- Interrupt after a capture before Done. Relaunch; recover exactly once, including follow-up finding association. Repeat after normal state save to verify acknowledged captures do not return.
- Complete a manual and AI-assisted inspection, generate English/Arabic/bilingual reports, save PDFs using system printing and inspect Arabic shaping, images and page breaks.
- Review report versions after reopening/resolving a finding; earlier versions stay unchanged.
- Backup/restore between browser and Android with evidence. Verify replacing rather than merging, corrupted ZIP rejection and missing-media handling.
- Install a same-signature update and verify data retained. Export backup before any uninstall test.
- Run20minutes live analysis per mode: record measured analysis latency/rate, CPU/memory, battery, thermal level, capture counts, save failures and dropped/late analysis. No fixed FPS guarantee is established until measured.
- Verify font scaling200%, TalkBack,48dp targets, RTL, system back, edge-to-edge insets and low-storage handling.
- Run NativeStateTest instrumentation on the phone: all4 models shape/finite smoke, large Arabic/emoji Room state chunk roundtrip, capture journal acknowledgement. Use a test workspace and back up first.

Do not mark this checklist passed from APK compilation, browser screenshots or model structure checks. Attach actual observations and failures.


## Burhan 1.2 capture and selection checks — physical results required

Record each row as Pass / Fail / Not supported, including the effective mode and actual image dimensions. This list is not a verification result.

- Select cracks → select cracks again → Explore: title, selection indicator, detector and overlay all clear together. Repeat Stop, Complete and Skip; saved photos remain.
- During a delayed agent response, switch checks and change scenes. A response for the previous task cannot change the selected check or its instruction.
- Tap a hint, swipe in each direction and use TalkBack Dismiss. Dismissal leaves the selected check untouched, persists in the same scene, survives a brief scene change for at least60seconds, and clears with Suggest again. Verify system animations off and no timed hint disappearance.
- Toggle every Camera & Capture setting in main Settings and native camera settings; relaunch and compare. Open settings while a request/capture is pending and return without losing selection or evidence. Hints stay hidden in Settings, Review and critical warning states.
- Press and hold both volume keys: exactly one capture per distinct key-down, no repeated shutter. Outside scanning, normal volume behavior returns.
- Pinch zoom then lift fingers: no accidental focus or capture. Long-press with focus lock off must not lock. With it on, lock, tap to release, disable to release, then leave/re-enter: torch and focus lock remain off.
- Check supported zoom clamps, remembered zoom on/off, and restoration after changing camera availability.
- Screen stays awake only while scanning with the preference enabled. Pause, Review, Settings, background and exit restore normal timeout. Resume respects current preference.
- With extra shutter cooldown on/off, double-tap and press shutter while storage is busy. Mandatory single-save protection always remains. Haptic confirmation occurs only after storage succeeds and only when enabled.
- Undo within5seconds removes exactly its capture from journal/count, not an earlier photo. Timeout, rotate, background and interrupt immediately after Undo; recovery must not resurrect the removed capture.
- Retake latest current-check capture successfully, with low storage/failure, and with interrupted replacement. Successful replacement updates count once; failure leaves the old photo. Retake must not replace a capture belonging to another check.
- Compare standard and higher-detail capture in portrait/landscape and on a device rejecting the3-use-case combination. Verify orientation, longest edge≤2048, source/dimensions, clear fallback indication, and overlays computed from that exact stored image. Confirm remote image bounds did not increase.
- Toggle live overlays, grid and level independently. Level follows phone orientation and never claims building alignment. Low-storage warnings hide suggestions and keep essential actions reachable.
- Mark arrows/ellipses in both image orientations; compare original, baseline and derivative. Generate report, edit again, and confirm old report images remain unchanged. Disable markup: existing annotations remain.
- Share selected original/annotated image to a real receiving app; verify URI grant and MIME. Browser download fallback saves the selected variant.
- Export/import a workspace with captures from both modes, markup and multiple reports; compare originals/baselines/annotations/capture metadata and all versions. Restore an older1.1 backup too.
