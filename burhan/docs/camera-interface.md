# Native camera interface — 1.2.0

The native camera now gives the live view priority. Its persistent controls are Back, the guide-state indicator, options, camera flash/pause/zoom, Import, Capture evidence and Review with the number of durably saved captures. With Quick Review enabled, Review opens session evidence while preserving the camera. All findings returns to the workspace. With it disabled, Review returns directly to findings; neither action completes the inspection.

The main panel names one current check and identifies manual evidence versus an installed detector. It shows a short next instruction. Details opens the complete instruction, supporting explanation and capture quality, plus technical diagnostics. Change opens the available checks. The available manual/crack/surface/PPE choices remain usable without pairing or network access; additional agent suggestions appear separately.

The Material bottom sheet also contains completion, skip, Stop check, Suggest again, guide pause/resume, Camera & Capture settings and context correction. Choosing the active check again returns to Explore, closes its detector and removes transient overlays. Pausing the guide stops new cloud snapshots while the camera and local detector can continue. Pausing the camera freezes its analysed view and is labelled separately. New context clears obsolete instructions and alternative suggestions before a new plan arrives.

Successful evidence storage updates the Review count and produces a confirmation and accessibility announcement. Haptics and five-second Undo follow their independent preferences. A tap on the preview requests CameraX autofocus and exposure metering; the transient focus ring acknowledges a request rather than certifying focus success.

The dark surfaces, yellow primary action and strong headings preserve Burhan's visual identity. Material 3 components provide buttons, text input and the bottom sheet. Controls use at least 48dp touch targets. Large text stacks capture above Import/Review, while compact landscape uses a supporting control pane and scrollable guidance. System bars and cutouts are inset.

## Verification

`npm run build:apk` runs the native JVM tests as part of the build. Robolectric renders actual Android/Material Views on SDK35 without an emulator. Layout cases include 320×640 and 360×800 portrait, 640×360 landscape, English/Arabic, 100/160/200% text, and a long storage warning. Assertions check essential control bounds, touch targets, text clipping, mid-word button wrapping, camera space, readiness, independent camera/guide states, full instructions and offline choices.

Native View PNG fixtures and their exact test results are included under `delivery/evaluation/native-ui`. The camera area is deliberately blank in these fixtures: the tests do not start CameraX, run a detector or contact an agent. They establish layout and interaction behavior, not phone camera performance. Verify tap focus, flash, zoom, capture durability, system insets and TalkBack on a physical phone using the device checklist.

Design references: [Android layout basics](https://developer.android.com/design/ui/mobile/guides/layout-and-content/layout-basics), [Android accessibility defaults](https://developer.android.com/develop/ui/compose/accessibility/api-defaults), and [Material bottom sheets](https://m3.material.io/components/bottom-sheets/overview).


##1.2 tools and selection

AI results are proposals only. A small single-line hint overlays the preview above existing camera controls; tap to select, swipe either way or use accessible Dismiss to hide it. Details retains the reason. Stable-scene suppression is distinct from request-version invalidation, and dismissal lasts at least60seconds. Hints hide in sheets, Settings, Review and warning states.

[Camera & Capture preferences](camera-settings.md) document all16 independent switches, higher-detail capture, review/markup/share, supported browser behavior and defaults. [Architecture](architecture-v12.md) shows local capture and remote proposal boundaries. Hardware-specific results are recorded separately from JVM renders.
