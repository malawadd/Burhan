# Physical Android verification — pending until a phone is attached

Record device model, RAM, Android version, WebView version, available storage, Burhan version/hash and ambient conditions. Target at least one2025 midrange ARM64 phone, plus a newer phone if available.

- Install the delivered APK; verify package com.burhan.site and version. Relaunch offline.
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
