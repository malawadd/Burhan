# Fixed construction pixel-video exporter: independent static review

Reviewed `scripts/compare_construction_pixel_videos.py` and its source, scoring, adapter, preprocessing and all-three binding helpers. This was code/metadata inspection only: no camera pixels, reserved photos, NN inference or model loading was performed.

## Pre-freeze blocker reported

The exporter verifies the prior four-video protocol's known SHA, but recomputes source file hashes through `selected_sources()` without comparing the selected three source records to the prior protocol's immutable `sources`. Consequently a changed camera/label file could receive a new hash and be accepted by the first new protocol freeze, despite the old protocol still being hash-correct. The same issue applies to the original source-group protocol unless its hash is checked against the prior receipt.

Required fix before freeze/inference: require the exact three selected source rows, in declared sequence order, to equal the corresponding prior pinned rows, and assert the current original source protocol SHA equals the prior `source_protocol_sha256`. The exporter should bind actual dependency implementation hashes (loader/association, scoring/matching, decoder and shared gain) alongside its own source hash. Root owns the exporter and was sent this finding; this review did not edit it.

## Contracts that appear coherent

- `binding_value()` requires complete300 lineage, all100/200/300 CAL reports, the global freeze even if rejected, fitted numerical parity and all122 serving-mask/box/score parity. The fixed AFTER hash is checked against final300 rather than substituting the CAL winner. Original TUT identity is independently pinned.
- `predict_maps(raw)` receives only unchanged camera pixels. It applies the shared whole-native gain once, then384 linear resize. Truth annotations enter scoring and external-GT rendering after prediction; no ROI, annotation mask or GT-centered crop enters inference.
- The native raw trace is the actual384 probability map resized bilinearly and thresholded strictly above .50. The separate served trace unions actual area4/top100 eight-connected decoder masks. Neither trace is presented as pixel GT or material certainty; no morphology fills a crack.
- All source camera frames and timestamps are loaded through the fixed source loader, timestamp-to-label association remains the shared publisher forward convention, lossless replay is checked pixel-by-pixel/PTS against raw, and output MP4 decoded PTS/count must equal every original exposure exactly.
- Masks are bit-packed at native geometry with original source indices/timestamps and per-frame raw/input/map/mask hashes. Publisher boxes remain a distinct pane and box-IoU≥.50 is the only video localization metric. Unsupported categories and unlabelled exposures retain their distinct scope.
- Existing complete or partial video/mask/report outputs fail closed rather than being silently overwritten. These reused exposed/calibration recordings receive no independent accuracy credit and no real-time claim.

Minor robustness recommendation: derive `frame_size_wh` from actual raw shape or assert all selected source geometries are exactly346×260; the current hardcoded shape matches the declared sources but should fail if that contract changes. Actual rendered/raw bitmask/PTS verification remains necessary after authorized export; static review cannot establish runtime output correctness.
