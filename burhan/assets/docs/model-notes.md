# Included models and supported observations

Four SHA256-pinned ONNX files are bundled: PPE YOLOv8n (~11.7MiB), joint-aware crack YOLOv8n segmentation (~12.7MiB), DACL SegFormer B1 surface320 and surface512 (~52.3MiB each). See assets/models/manifest.json for exact hashes, upstream source, preprocessing, ordered labels and licences. Existing failed research checkpoints are excluded.

The surface model evaluates independent sigmoid channels, preserving overlapping labels rather than selecting a single global class. Conditions include crack, alligator crack, wet spot, efflorescence, rust, rockpocket, inspector-marked hollow-area boundary, cavity, spalling, graffiti, weathering, residual formwork, exposed reinforcement and surface-loss appearance. A hollow-area class observes chalk boundaries drawn by an inspector; it does not see hidden hollow material. Wet spots do not measure moisture or prove a hidden leak. PPE absence is not inferred merely because an item was not detected.

The 320 surface model has weaker recorded diagnostics than512; live processing has not been field-qualified. The existing project records weak/uneven precision and recall, missed thin cracks/spalling and failed newer training candidates. Native packaging does not improve accuracy. There is no independent Saudi-site qualification.

Concrete strength, reinforcement cover, waterproofing performance, humidity, calibrated dimensions/flatness, torque/anchor engagement and code acceptance remain outside automatic support. The original89-topic catalogue records these as manual/instrument/document checks. The UI's mode controls the analysed class group.

Desktop source references: ../docs/current-model-status.md, ../docs/construction-round4-measured-results.md and ../config/construction-capabilities.json (relative to the parent project root, use parent docs/config folders). Burhan assets retain full source-catalogue.json to preserve the original research references.

## Preprocessing and runtime

PPE/crack use640px RGB float32 letterboxing with114 padding, class-wise NMS and source-image coordinate projection. Crack masks use actual prototype coefficients, not filled boxes. DACL takes stretched BGR uint8 NCHW; its graph contains normalization. It uses independent sigmoid classes,8-connected components and the original minimum component policy.

Browser JavaScript interpolation differs slightly from OpenCV. On the recorded original procedural fixture, decoded golden outputs match Python exactly; actual WASM raw drift is small. Full JS-input masks differ by19 pixels for crack,6 for surface320 and18 for surface512; the smallest component's IoU is0.875. These are disclosed numerical comparisons, not measures of crack detection accuracy. See tests/fixtures/README.md.

## Licence/provenance notice

PPE metadata includes Ultralytics AGPL3 lineage and an author CC BY4 weight declaration. Joint-aware crack weights include AGPL3 lineage and merged source data whose redistribution clearance is unresolved. DACL photos are CC BY-NC4; the upstream author's checkpoint licence is undeclared, and a mirror label does not establish permission. Personal prototype packaging does not resolve commercial/public redistribution rights. Preserve notices and obtain appropriate rights before distributing beyond the chosen personal use.

OpenCV is Apache2, ONNX Runtime MIT, CameraX/Room AndroidX Apache2, React/Capacitor/Vite MIT and the bundled Noto fonts SIL Open Font Licence. Font licence texts are bundled alongside the fonts. Runtime/package licences remain in locked dependencies and native AARs.
