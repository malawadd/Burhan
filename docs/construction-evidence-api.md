# Recorded construction evidence API

The recording list is a fixed local registry in `config/construction-videos.json`. It exposes public artifact IDs and HTTP URLs, omitting `path` and `evidence_path`. The MP4 and JSON routes independently resolve allowlisted files under the workspace `output` directory; unregistered files, path escapes, missing files and wrong suffixes are unavailable. File names supplied in a request are not arbitrary filesystem access.

`GET /api/construction/videos/{id}/evidence` returns a registered evaluation JSON. Files above 4 MiB, invalid UTF-8, invalid/overly nested JSON and malformed evidence return an explicit 409. Unknown or unavailable artifacts return 404. Video playback retains its separate MP4 route, byte-range seeking and HEAD support.

The evidence validator checks frame arrays, timestamps, annotation/prediction arrays, ID references, nonempty labels, finite scores and IoUs, positive bounding-box geometry, and frame bounds where dimensions are supplied. A 0.000001-pixel tolerance preserves the publisher's floating-point rounding around image edges. A claimed localized match additionally requires an evaluated frame, a supported annotation, matching class labels and a unique annotation assignment. The backend recomputes the actual box IoU, checks agreement with the stored IoU and requires the declared threshold (default 0.50). Merely adding `matched_annotation_id` to an unrelated prediction cannot establish localization.

This validation establishes file and geometry consistency. It cannot establish that publisher labels are correct, that a dataset was independent of model training, that a surface condition is structurally dangerous or that Saudi code requirements passed. Pixel masks are not inferred from bounding-box annotations. A schema-valid unmatched box is still a model candidate, and unscored/unsupported frames remain outside the declared detection accuracy scope.

`tests/test_construction_evidence_api.py` uses HTTP byte and geometry fixtures with a stub runtime. It starts no camera, model inference or database. The tests cover valid evidence, hidden registry paths, allowlisting, traversal, wrong/missing files, malformed and oversized JSON, unsupported damage, false match claims, one-to-one matching, and unchanged MP4 Range/HEAD delivery. These fixtures are not real-video or model-accuracy evidence.

Verification on 20 October 2026: **33 passed in 3.78 seconds**. The sole warning is an upstream Starlette TestClient / AnyIO deprecation. Reproduce with `.venv\Scripts\python.exe -X utf8 -m pytest tests/test_construction_evidence_api.py -q`.

## Visible-condition capabilities and model selection

`GET /api/construction/capabilities` returns `config/construction-capabilities.json` with resolved availability. `model_status` is the declared research/evaluation state; `effective_status` can be `supported_experimental` only if the referenced registered model file has the expected SHA256 and its capability list contains this category. Missing or modified weights resolve to `unavailable`. Merely listing a dataset or an intended feature does not establish a working detector.

The `models` array contains registered damage-variant IDs, readable labels, installation state, input size and capability IDs. The dashboard derives its damage-model selector from this inventory, including separately named resolution variants when registered. This endpoint checks local bytes; it neither loads a neural model nor opens a camera. It does not establish local precision/recall or independent field validation; those measurements and their scopes remain attached to the category study.

`effective_status` also preserves `measured_weak` for installed models with documented weak localization. This is distinct from a model that is absent or unimplemented. The optional operator `damage_classes` task filter must name supported conditions; it cannot invent a detector, change a recorded benchmark or establish that other conditions were inspected. Result model metadata retains `damage_variant`, `class_confidence`, `selected_damage_classes`, selected stage and confirmation count.
