# Camera & Capture — Burhan 1.2.0

Open **Settings → Camera & Capture**, or use the camera’s options menu. Camera settings are saved on this device, separately from inspection backups and pairing credentials. Opening the native settings screen pauses camera analysis and remote snapshots. Return with Done to retain your check and saved evidence. Preference definitions and English/Arabic descriptions live in `public/camera-settings.json`; native and browser settings use that same file.

## Explore, inspect and suggestions

Explore has no selected check or running defect detector. Choose a check to update its indicator, detector, title and instruction together. Choosing the active check again, Stop check, Complete or Skip returns to Explore and clears transient overlays and pending guidance. Saved evidence stays available.

The AI proposes checks; it does not choose one for you. A small **✦ Try: … ›** hint appears within the preview above the camera controls. Tap to select it. Swipe horizontally either way to dismiss without changing the current check. An accessible Dismiss action is available. Explanations are in Details. Hints wait for your interaction and have no reading timeout. Motion follows the system animation preference.

A dismissed check stays suppressed in the current stable scene and for at least 60 seconds. Changing the selected check alone is not a new physical scene. Complete and Skip also filter that check for the current scene. **Suggest again** in Details explicitly clears suppression. Settings, Review and critical warnings hide hints. Returning from a sheet must not display an obsolete scene suggestion.

## Switches

| Preference | Initial value | Effect |
| --- | --- | --- |
| AI suggestion hints | On | One compact hint; Details remains available |
| Volume-button capture | On | One photo per distinct hardware key press |
| Pinch to zoom | On | Camera-supported zoom, separate from tap focus |
| Quick capture review | On | Review recent evidence and continue the camera |
| Undo last capture | On | Five-second Undo for the identified newly saved photo |
| Keep screen awake | On | During active scanning only |
| Accidental double-tap protection | On | Extra shutter cooldown; atomic save and one capture in flight are always enforced |
| Capture haptics | On | Successful storage confirmation |
| Remember last zoom | On | Supported zoom restored; torch and focus lock are never restored |
| Show live detection overlays | On | Detector marks on preview; saved evidence unaffected |
| Long-press focus lock | Off | Hold to lock focus, tap or disable to release |
| Framing grid | Off | Optional thirds guide |
| Camera level | Off | Phone orientation; not a building-level measurement |
| Higher-detail photos | Off | Dedicated CameraX photo capture when supported |
| Photo markup | Off | Arrows and circles in Review |
| Individual photo sharing | Off | Explicit original/annotated sharing |

Automatic saving remains a separate existing workspace preference. Disabling any optional feature does not delete evidence or existing annotations. Changes apply on return to the camera; quality changes apply to subsequent captures.

## Review, Undo and Retake

With Quick Review enabled, Review opens recent current-session evidence. Choose Original or Annotated, Continue camera, Retake latest capture, or All findings. Retake applies to the latest current-check capture, even if another check has a newer photo, and replaces it only after the new photo has saved successfully. A failed replacement retains the previous photo. With Quick Review disabled, Review returns directly to the inspection findings.

Undo is tied to one capture ID and lasts five seconds. It cannot delete earlier evidence. The save journal and displayed count are updated together. Evidence remaining in the native journal is recovered into the workspace on the next app load if capture is interrupted.

Markup starts from the immutable detector baseline and writes a new annotated derivative. The original and detector results remain available. Reports use saved snapshots, so editing current evidence does not alter a report already generated. Annotations store normalized arrow/ellipse coordinates. Backups include all referenced originals, baselines and derivatives, including older report snapshots.

Sharing is deliberate: select the original or annotated version, then Share. Android opens the platform share sheet. Browsers use file sharing when available and otherwise download the selected image.

## Capture modes and browser differences

Standard mode saves the analysed frame. Higher-detail mode binds CameraX ImageCapture alongside Preview and analysis. It normalizes orientation, bounds the longest edge to 2048 pixels, and runs the detector against that retained still image before creating its overlay. It never copies detections from another live frame. Unsupported camera combinations fall back to standard capture and display the effective mode. Stored dimensions and `captureSource` identify what was actually saved (`analysis`, `photo` or `import`). Remote snapshots retain their existing thumbnail bound; turning on higher detail does not increase cloud upload size.

Android supplies the full hardware camera workflow. The browser supports shared check selection, suggestions, preferences, imports, image review, Undo, Retake, markup, grid, overlay visibility and sharing/download. Unsupported volume, native pinch/focus controls, sensor level, dedicated photo capture and native haptics are identified in browser Settings. Browser camera access and wake lock depend on device permissions and browser support. Use image import when a desktop camera is unavailable.

## العربية

افتح **الإعدادات ← الكاميرا والالتقاط** أو قائمة خيارات الكاميرا. تُحفظ المفاتيح على الجهاز، وتبقى أدلة الفحص والاتصال بالخادم منفصلة عنها. يؤدي فتح الإعدادات إلى إيقاف الكاميرا وإرسال الصور مؤقتاً، ثم تُستأنف مع الاحتفاظ ببند الفحص والأدلة عند العودة.

في وضع **الاستكشاف** لا يوجد بند محدد ولا كاشف عيوب نشط. اختر بنداً لتغيير الكاشف والعنوان والتعليمات معاً. إلغاء اختيار البند أو إيقافه أو إكماله أو تخطيه يعيد الاستكشاف دون حذف الصور. اقتراح المساعد مجرد اقتراح: انقر التلميح الصغير لاختياره أو اسحبه أفقياً لإخفائه دون تغيير الفحص الحالي. الشرح في التفاصيل، وخيار «اقترح مجدداً» يعيد إتاحة الاقتراحات المستبعدة.

تتيح المراجعة السريعة عرض الأصل أو الصورة المعلّق عليها، ومتابعة الكاميرا، وإعادة أحدث صورة للبند الحالي، والانتقال إلى جميع الملاحظات. لا تُحذف الصورة السابقة عند فشل إعادة الالتقاط. يخص التراجع الصورة الجديدة وحدها خلال خمس ثوانٍ. تحفظ الأسهم والدوائر نسخة جديدة مع بقاء الأصل وخط أساس الكاشف، ولا تتغير التقارير المحفوظة سابقاً. مشاركة الصور تتطلب اختياراً صريحاً.

الصور الأعلى تفصيلاً والتعليق والمشاركة وقفل التركيز والشبكة والميزان مغلقة افتراضياً. بقية أدوات الالتقاط المساعدة مفعلة افتراضياً، ويمكن تغيير كل أداة بشكل مستقل. يصف ميزان الكاميرا اتجاه الهاتف فقط. تظهر قيود ميزات العتاد بوضوح في المتصفح. راجع قائمة فحص الهاتف قبل الاستخدام الميداني.


## Expanded inspection models (v1.3)

Default: **off**, independent of other capture settings. Enables three families/30 checks, including five model-backed experimental appearance checks. Switching it off cancels expanded guidance and returns the selected expanded check to Explore; evidence and saved reports remain. See [expanded-inspections.md](expanded-inspections.md).
