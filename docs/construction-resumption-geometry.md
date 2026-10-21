# Real installed-floor diagnostics and tile geometry repair

21 October 2026. Real public floor inputs exposed a runtime crash that mocked geometry tests had missed. The tile path now executes those inputs, reports readable image-space edge-band widths, and retains its research status. A separate unchanged crack/spalling model probe demonstrates that its transfer to installed ceramic floors is poor. No owner footage was trained on or used to tune this work.

## Actual inputs and rights

Two original photographs were acquired through Wikimedia Commons' official `imageinfo` API. Publisher SHA1 and byte lengths were checked, then local SHA256s recorded in `datasets/construction/installed-floor-diagnostics-v1/asset-manifest.json`. These are real installed floors, not factory tiles, rendered scenes, thumbnails or videos.

| Photograph | Author / reuse | Local SHA256 |
|---|---|---|
| [Cracked tile flooring](https://commons.wikimedia.org/wiki/File:Cracked_tile_flooring.jpg), 2096×1179, 741,464 bytes | Malcolm Koo / Mk2010, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); file page also offers a separate CC BY4 permission | `c944810d04acaca9a38faad91e99ba863ab74e8338188b60002ca40666379a8b` |
| [Tiled Floor](https://commons.wikimedia.org/wiki/File:Tiled_Floor.jpg), 5312×2988, 5,129,094 bytes | Deavmi, [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) | `472ec98fe5df3edad74c2abf5cd9959553c047194685410694d7aa0eca25b2c9` |

Derived images preserve prominent author/license credits; modifications are resizing, geometry overlays or model predictions. Reuse each derivative under its original photograph's listed ShareAlike license. The photographs have no independent defect masks, joint-width measurements or complete healthy/defective labels. They are development diagnostics, not a held-out accuracy evaluation. In particular, the second photograph is not a certified fault-free installation.

## Confirmed before failure and implemented repair

The installed OpenCV binding returns `HoughLinesP` segments as **N×4**, while the old code iterated `raw[:100,0]` as though it were N×1×4. All four photo/resolution combinations failed with `TypeError: cannot unpack non-iterable numpy.int32 object`. The original source and failed inputs/protocol/results are preserved in `output/evaluation/construction-resumption-geometry/before-v2/`; the initial interrupted `before-v1` also remains. New regression tests reproduced the failure before the fix.

Only the tile geometry path in `edge_runtime/profiles/construction_tasks.py` changed:

- Normalize both OpenCV result shapes. Canonicalize segment endpoints along with their direction, fixing overlap measurements when an endpoint order is reversed.
- Calculate possible parallel pairs in bulk instead of thousands of Python pairwise operations. Suppress surface grain before line voting with a 3×3 Gaussian blur and use 0.5-degree Hough bins. Actual band support still samples the original grayscale pixels.
- Sample 24 cross-sections along a candidate. At least 60% must show an interior consistently brighter or darker than both flanks by at least eight grayscale levels. This rejects arbitrary parallel lines on a flat image; it does not distinguish grout, a shadow, a crack or decoration.
- Report median perpendicular edge spacing, spatial p10/p90 line-fit widths, supported sample count/fraction and intensity contrast. Spatial variation is **not measurement uncertainty or confidence**. Prioritize sustained spans and suppress overlapping duplicate pieces.
- Render numeric width labels inside the image bounds and include the measurement/support fields in actual `Finding.measurements`. Explanations say “visible edge band,” with installation condition unassessed.

The 2.5-pixel minimum width and nearly parallel/long-span requirements mean thin or oblique joints may be missed. At 1280 pixels the ordinary-floor example yields only one accepted band; this is a documented coverage limitation, not an accuracy improvement. Widths depend on analysis resolution/perspective. Without a valid physical reference, none becomes millimetres or an installation alarm. All geometry observations remain `research_measurement`, with no fault event emitted.

## Before/after execution and timing

`scripts/diagnose_construction_geometry.py` freezes the executed source and protocol before processing. Each photograph is resized to the same 640/1280 maximum dimensions. Geometry timing uses one OpenCV thread, 12 repeated calls on that real still, with the first two excluded. Another construction fit was active; these are **contended engineering timings**, not sustained preview/video qualification or independent samples. The final stage also runs the real full Guided profile once and saves its actual findings and annotated image.

| Real photo / analysis size | Before | Shape-fix-only p50 / pairs | Final geometry p50 / p95 / bands | Full-profile single call |
|---|---|---|---|---|
| Cracked floor, 640×360 | Runtime error | 69.2 ms / 7 | 37.3 / 41.4 ms / 8 | 52.4 ms |
| Cracked floor, 1280×720 | Runtime error | 383.1 ms / 5 | 65.9 / 93.5 ms / 6 | 84.9 ms |
| Tiled floor, 640×360 | Runtime error | 50.6 ms / 10 | 22.8 / 27.7 ms / 4 | 34.4 ms |
| Tiled floor, 1280×720 | Runtime error | 133.0 ms / 5 | 37.3 / 44.3 ms / 1 | 43.8 ms |

Counts are observations, **not correct detections, recall or false alarms**. Final outputs are in `after-final-v1/results.json`; the intermediate compatibility/support stages are preserved instead of overwritten. No reliable 15-fps video-inspection claim follows from these still timings; the application measures live preview separately.

Example actual Guided-profile output on the damaged floor:

![Malcolm Koo (Mk2010), CC BY-SA 4.0; actual photograph resized with research edge-band measurements](../output/evaluation/construction-resumption-geometry/after-final-v1/Cracked_tile_flooring-640-actual-profile.jpg)

The bands visibly follow several joints; they do **not** identify the central broken tile or determine missing grout. The photo must not be presented as successful damage detection.

## Existing crack/spalling model: actual installed-floor transfer failure

The fixed `round4_mixed_cpu_failed` model was run once on each 640×360 public photograph. It used the original `crack`/`spalling` classes and existing 0.25/0.25 thresholds, CPU ONNX Runtime with one intra/inter-op thread, without fitting or changing a threshold. Verified model SHA256: `8bf254cc4e1079bc58bd9590f5e57630890f6d4048e2afc76a1ceb51dfaa1d2b`.

The visibly broken floor produces **one crack candidate** (score 0.7254; 5,110 predicted mask pixels). Visual inspection shows it mostly follows an ordinary right-hand tile joint, while the central missing/broken material and most fractured edges are missed. The second floor produces zero candidates. These are qualitative observations; no precision/recall is computed without annotated truth, and zero candidates do not establish a correct installation.

![Malcolm Koo (Mk2010), CC BY-SA 4.0; unchanged model prediction, including joint confusion and central damage misses](../output/evaluation/construction-resumption-geometry/defect-probe-v2/Cracked_tile_flooring-fixed-defect-probe.jpg)

`defect-probe-v2/` preserves the before-inference protocol, actual execution/model provenance, native binary PNG mask and predicted overlay. The earlier v1 probe was rejected before inference because an explicit execution option was missing; it is retained and is not counted as a model result. The new geometry measurements must not be merged with these masks to invent missing-grout or cracked-installed-tile labels.

## What public evidence is still required

| Intended visible task | Evidence needed before training/evaluation | Confusions that must be annotated |
|---|---|---|
| Plate-to-wall separation at a railing/bracket | Original real side/oblique views resolving both plate edge and substrate, gap-region masks, verified plate/connection identity, approved flush-versus-stand-off/shim reference, repeated angles grouped together; independently measured gaps for any mm claim | Intentional stand-offs/spacers, uneven render, sealant, shadows, occlusion and a correctly seated plate. Bolt/nut loosening labels cannot stand in for a plate-wall gap. Torque/hidden anchorage still needs measurement. |
| Joint spacing / inconsistent installed spacing | Installed-floor originals with both tile edges and joint region annotated; an independently measured same-plane reference and manual widths at multiple positions; scene/site IDs and approved spacing/tolerance | Perspective, lens distortion, beveled tile edges, patterned veins, reflections, spacers and intentional variable/mosaic layouts. Measuring parallel edges alone is insufficient. |
| Missing grout | Installed and completed-joint region masks distinguishing present versus absent required grout, full-view context and intentional-joint references, correct-grout negatives and original scene/site identity | Dark/dirty but present grout, black decorative grout, shadow, ungrouted work in progress, expansion/perimeter joints and sealant-filled joints. |
| Cracks/chips in installed ceramic | Original installed-tile pixel masks or boxes that separate tile bodies, cracks, chipped edges and grout; small defects retained at native scale; site/photo groups and correct-installation negatives | Ordinary joints, printed/decorative veins, grout cracks, tile bevels, dirt and reflections. A concrete-crack model or factory-tile defect dataset does not establish this task. |

The primary [2013 installed-tile alignment study](https://doi.org/10.1016/j.autcon.2013.01.009) and [UC-Tile study](https://doi.org/10.1016/j.autcon.2025.106001) support geometry-based inspection research, but this follow-up did not verify an author-provided downloadable archive with those fault annotations, licenses and physical-site groups. Searches of publisher repositories also returned rock-grouting/jet-grout data; those are different tasks and were excluded. No free public plate-to-wall separation archive with appropriate labels was verified. This is a specific verified gap, not a claim that none exists anywhere. The acquired public floor originals are usable diagnostics but cannot supply 50 independent positive/negative scenes.

## Reproduction and software verification

```powershell
.venv\Scripts\python.exe scripts/diagnose_construction_geometry.py acquire
.venv\Scripts\python.exe scripts/diagnose_construction_geometry.py after-reproduction-v1
.venv\Scripts\python.exe scripts/diagnose_construction_geometry.py defect-probe-reproduction-v1
.venv\Scripts\python.exe -m pytest -q tests/test_construction_guided_tasks.py tests/test_construction_task_detector.py
```

Choose new stage names; frozen stage directories cannot be overwritten. The focused suite passed **70 tests** in 2.93 seconds, including both Hough binding shapes, reversed endpoints, flat-surface abstention, actual measurement metadata and existing calibration/mask/runtime boundaries. Generated fixtures test software mechanics only. The two public photograph outputs supply actual execution evidence separately; neither establishes qualified fault accuracy.
