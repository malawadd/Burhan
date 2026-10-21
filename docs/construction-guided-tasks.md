# Guided construction task runtime

Implemented in `edge_runtime/profiles/construction_tasks.py`. These are real-input research workflows, not newly qualified defect detectors. Catalog entries do not make categories operational. The public catalog exposes availability/reason per category and currently releases **zero** categories.

## Task selection and boundaries

`inspection_task` selects exactly one family: `walls_finishes`, `brackets_railings`, or `tile_installation`. The host routes guided configuration to `GuidedConstructionProfile`; the legacy construction profile remains separate. This module never opens a camera, uploads footage, downloads weights or creates an unbounded work queue. It consumes the host's actual frame and returns an annotation of that same frame.

Walls include damp/stained appearance, peeling, blisters, efflorescence, cracks, spalling, paint runs, patchy coverage, damaged/missing sealant and uneven reveals. Brackets/railings include missing bolts/nuts, visibly unseated connections and corrosion appearance. Tile installation includes joint dimensions/variation, alignment, missing grout, tile damage, missing tiles, open perimeter joints and skirting alignment. All categories except experimental edge-spacing geometry are unavailable until suitable real data and independent evaluation exist.

No RGB task measures humidity percentage, torque, concealed anchorage, tile hollowness/adhesion, floor slope or lippage. A generic `wetspot`, `leakage`, `bulge`, `abscission` or manufacturing anomaly is not automatically renamed a precise new wall/installation fault.

## Configuration

```json
{
  "inspection_task": "tile_installation",
  "roi": [0.1, 0.2, 0.9, 0.9],
  "nominal_joint_mm": 3,
  "tolerance_mm": 0.5,
  "confirm_frames": 2,
  "calibration_reference": {
    "marker_size_mm": 50,
    "marker_id": 0,
    "same_plane": true,
    "image_size": [640, 480],
    "camera_matrix": [[800, 0, 320], [0, 800, 240], [0, 0, 1]],
    "distortion_coefficients": [0, 0, 0, 0, 0]
  },
  "expected_fasteners": [
    {"id": "bolt-1", "bbox": [0.1, 0.2, 0.2, 0.3]}
  ]
}
```

**The example intrinsic matrix is illustrative; it is not a calibration of this laptop or camera.** `image_size` must match the actual analysis frame, including rotation/resizing. Supply intrinsics calibrated for that exact image geometry. Marker dictionary is OpenCV `DICT_4X4_50`; exactly one chosen ID must be detected. `same_plane=true` is the operator's explicit physical reference assertion, not an automatic plane verification. Calibration is optional; absent/missing/small/unstable markers and image-size mismatch fall back to pixels. Invalid numerical/reference configuration is rejected.

The repository includes a [printable ID0 reference](../assets/construction-reference/aruco-4x4-50-id0-50mm.svg). Its black outer square is specified as 50mm, surrounded by a 10mm white quiet border. Print the SVG at actual size/100%, disable fit-to-page, and independently measure the printed black square before entering `marker_size_mm`. The PNG is a screen preview, not a physically calibrated print file. `scripts/create_construction_reference.py` reproduces both files and their hash receipt; its software decode check is target-generation verification, not a measurement-accuracy test. None of the supplied owner videos contains this reference or a verified camera calibration, so their results cannot acquire physical scale from this asset.

`roi` and every expected-fastener `bbox` are normalized full-frame **x1,y1,x2,y2**, not x,y,width,height. Returned finding boxes use analysis-frame x,y,width,height. The fastener reference is an expected visible layout. It does not prove a bolt absent.

`min_sharpness` optionally adjusts the Laplacian-variance capture gate (default 35). It is a capture heuristic, not a learned quality score. Minimum material ROI is 96 pixels per side; heavily clipped/dark or unsharp crops do not run inference. Visibility/occlusion cannot generally be established from image statistics; the output explicitly records that limitation.

## Research model adapter contract

The constructor accepts `detector=local_adapter` for a compatible local trained model. The adapter implements `infer(uint8_BGR_crop)` returning detections with:

```python
{"label": "peeling_paint", "bbox": (x, y, width, height), "score": 0.8}
```

Label semantics must match the catalog exactly; detection geometry is crop-local. Cross-task, unknown, malformed and explicitly occluded detections are ignored. The adapter must apply its independently selected calibration threshold; a score is not universal probability. Optional `describe()` is exposed verbatim under `model_provenance`. Optional `prepare_worker()` is delegated; otherwise three stateless inference initialization calls at `input_size` are discarded before source capture. Initialization arrays are never evidence or accuracy tests.

An optional `mask` must be a boolean NumPy array matching the native crop height/width. True mask pixels are clipped to the detection region, rendered in place, and recorded as actual mask area plus external-contour polygons in original analysis coordinates (largest 32 contours at most; truncation is flagged). Empty valid masks abstain; malformed masks are not stretched or painted across the frame. Box-only predictions remain explicitly identified as box localization. These are model predictions, not ground-truth masks.

An explicit `research_model` configuration object can wrap an existing `ConstructionModel` variant **only for `walls_finishes`**. Tile and bracket selections reject this option before a model is loaded; the unrelated crack/bridge network cannot consume those tasks' initialization/inference budget. Their models must use a task-matched `research_model_manifest`. Regression tests verify both rejected families never call the legacy constructor and the explicit wall path remains available. No failed model is implicitly selected by default. This wrapper does not grant new class semantics, quality or release status.

New `research_model_manifest` assets are confined to the local construction-model directory and SHA-bound. The common adapter now rechecks and supplies the exact verified ONNX bytes to ORT, rather than letting ORT reopen a possibly replaced path or load unhashed external-weight sidecars. Models must be self-contained. Malformed preprocessing/threshold-object types and invalid ORT graphs produce configuration errors. Regression tests cover the verified-byte binding, file-change race, actual external-data graph rejection and invalid configuration. These loader checks do not change decoder thresholds, model weights or measured accuracy.

For `missing_bolt`/`missing_nut`, all three are mandatory: positive missing-class prediction, an overlapping expected slot (IoU at least 0.2), and adapter-provided `seat_visible=true` based on actual visible evidence. Empty predictions are never converted into missing fasteners. Existing generic detectors that cannot establish seat visibility will abstain from this decision.

## Tile geometry and output

The geometry path measures spacing between sufficiently long, near-parallel Canny/Hough edge pairs. It does **not** recognize grout, distinguish decorative patterns, validate tile material, count missing tiles, or declare faulty workmanship. Output label is `parallel_edge_spacing` and status is `research_measurement`.

With valid intrinsics, a visible marker and explicit common plane, the selected crop is undistorted and rectified using the marker plane. Crop boundaries constrain extrapolation; marker sides below 30 pixels or unstable/extreme perspective abstain from physical scale. Observation polygons map back through the inverse homography and camera distortion to the original inspected frame. Widths may be reported in **experimental mm**, clearly marked `not_independently_validated`; otherwise units are image pixels. Entered nominal/tolerance values are preserved for research comparison, but no threshold crossing generates an installation alarm. Measurement uncertainty is `not_estimated`, not an invented accuracy bound.

The algorithm can pair shadows, texture or marker edges. This is why it remains research-only pending independently measured real installations and task-specific positive/negative validation. It must not be marketed as an operational gap-width classifier on the basis of these mechanics tests.

Every observation includes task, frame sequence, native video timestamp when supplied, analysis-frame localization, status, explanation and optional measurements. Research candidates require repeated overlapping observations for a review event; emitted tracks suppress repeats for their visible lifetime. Session/sequence/time regression resets tracking. Geometry observations never emit fault events. No observations means unassessed/unavailable, never installation approval.

## Verification

`tests/test_construction_guided_tasks.py` checks task/catalog availability, malformed configuration, simulation rejection, bad timestamps, capture-gated inference, exact label semantics, ROI coordinate mapping, repeated-event suppression, task isolation, missing-fastener visibility prerequisites, calibration dimension/marker fallback, marker projection and non-alarming geometry behavior.

Generated arrays and printed-marker arrays in unit tests test software mechanics only. They provide **zero model accuracy, real tile measurement accuracy or real-video success evidence**. Accuracy/speed release requires the separate frozen real-data evaluation protocol, including at least 50 independent positive and 50 negative examples per released category, 95% precision and 90% recall targets, and sustained runtime measurements.
