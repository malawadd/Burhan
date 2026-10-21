# Construction category catalog and unsupported checks

21 October 2026. This records the 30 guided checks implemented in the task catalog. **Zero categories are released.** Research weights can be selected explicitly; they are not reliable inspection defaults. The API also verifies installed model hashes before listing those choices.

“Unavailable” means the application must abstain from that fault decision. “Research” means an actual algorithm/checkpoint exists, with the failures and qualification gaps linked below. Neither status means a healthy installation when no candidate is produced.

## Walls and finishes

| Check / identifier | Implementation and evidence | Remaining release gap |
|---|---|---|
| Damp/stained appearance / `damp_stained_appearance` | New SSDLite on real MBDD2025 leakage/stain photos; [failed photo and owner-video checks](construction-expansion-wall-stain.md) | Severe false candidates; source grouping and healthy-wall/video negatives unqualified |
| Mold-like appearance / `mold_appearance` | PHELE publisher `Mold`, kept separate from dampness; YOLO and SSDLite [comparison](construction-expansion-phele.md) | Only three TEST positives; poor recall; no species or hidden-moisture conclusion |
| Peeling paint / `peeling_paint` | PHELE actual box annotations; both small detectors trained | Poor localization recall/precision, insufficient independent scenes |
| Paint blister / `paint_blister` | Unavailable; indoor_defects metadata lists paint_bubble | Genuine annotation archive not acquired; no training from thumbnails |
| Efflorescence / `efflorescence` | Unavailable; [real Batalha limestone photo/mask audit](construction-expansion-additional-data-leads.md) | One known site, white-joint annotation scope needs review; no local fit, indoor-wall/video validation or qualified hard negatives |
| Crack / `crack` | Existing explicit legacy research adapter retained | Existing crack evidence is not new-category qualification; thin-feature and decorative-line misses remain documented |
| Spalling / `spalling` | Existing explicit legacy research adapter retained | Existing spalling evaluation retained; no automatic qualification transfer |
| Paint run / `paint_run` | Unavailable | Need actual installed painted-surface labels, drips/texture and reflection negatives |
| Patchy coating / `patchy_coverage` | Unavailable | Need coating labels and lighting/shadow/intended-texture negatives |
| Broken sealant / `broken_sealant` | Unavailable | Need installed-joint localization and serviceable-sealant negatives |
| Missing sealant / `missing_sealant` | Unavailable | Need required-sealant references and intentional-open-joint negatives |
| Uneven door/window reveal / `uneven_door_window_reveal` | Unavailable | Need calibrated geometry, approved reference and independently measured openings |

## Brackets and railings

| Check / identifier | Implementation and evidence | Remaining release gap |
|---|---|---|
| Missing expected bolt / `missing_bolt` | Conservative adapter contract requires a positive missing-bolt class, expected location and positively visible seat | SDNET combines bolt/nut absence; no model establishes all three prerequisites |
| Missing expected nut / `missing_nut` | Same conservative reference/visibility contract | Separate identity and independent real-installation validation absent |
| Generic missing-fastener appearance / `fastener_absence_appearance` | SDNET publisher Missing; actual boxes and polygon masks; [detectors](construction-expansion-sdnet-benchmarks.md), [mask fits](construction-expansion-sdnet-masks.md) | Failed quality, unverified site identity and incomplete cross-class annotation scope; does not establish required hardware |
| Visibly loosened bolt/nut / `visibly_unseated_connection` | SDNET publisher Loosen, source-faithful research semantics | Poor accuracy; not an arbitrary plate-to-wall-gap, torque or anchorage detector |
| Corrosion appearance / `corrosion_appearance` | Unavailable | Audited SDNET archive contains no corrosion class; advertised scope is not annotation evidence |

The owner railing clip visibly separates a mounting plate from the wall. A publisher bolt/nut-loosening model does not establish that installation fault. Its replay remains a documented failure; empty detections cannot establish secure fastening.

## Tile installation and condition

| Check / identifier | Implementation and evidence | Remaining release gap |
|---|---|---|
| Edge/joint width / `joint_width` | Actual Canny/Hough parallel-edge research, optional marker-plane rectification; [implementation](construction-guided-tasks.md) | Edge identity is not grout identity; no independently measured real installed-floor validation |
| Too-wide joint / `joint_too_wide` | Unavailable; nominal/tolerance can be recorded | Physical validation and grout identification absent; no acceptance alarm |
| Too-narrow joint / `joint_too_narrow` | Unavailable | Same measurement/data gap |
| Inconsistent spacing / `inconsistent_spacing` | Unavailable | Need installed-grid joint identity and real measured variation |
| Crooked joint line / `joint_alignment` | Unavailable | Need actual reference/layout and calibrated line validation |
| Row misalignment / `row_misalignment` | Unavailable | Need intentional stagger/pattern negatives and reference layout |
| Missing grout / `missing_grout` | Unavailable | No accepted installed-grout annotation dataset |
| Cracked installed tile / `cracked_tile` | Unavailable | Manufacturing tile images/crack classes cannot establish installed-tile performance; need decorative-vein negatives |
| Chipped edge / `chipped_edge` | Unavailable | Need installed-edge labels, intended bevel and joint negatives |
| Broken corner / `broken_corner` | Unavailable | Need real installed-corner examples and deliberate cut/opening negatives |
| Missing tile / `missing_tile` | Unavailable | Need expected layout and intentional unfinished/opening negatives |
| Open perimeter joint / `open_perimeter_joint` | Unavailable | Need specified perimeter seal/detail, movement-joint and deliberate-opening negatives |
| Misaligned skirting / `misaligned_skirting` | Unavailable | Need wall/floor reference geometry and measured installed examples |

No downloaded manufacturing-tile anomaly set is counted as installed grout-spacing validation. UC-Tile research is not treated as an available, independently measured dataset just because its paper exists.

## Unsupported physical conclusions

| Requested conclusion | Additional evidence needed |
|---|---|
| Wall moisture percentage or active hidden leakage | A suitable calibrated moisture/instrument measurement and investigation; RGB staining alone cannot supply it |
| Fastening torque or preload | Specified torque/preload procedure and appropriate calibrated tooling |
| Concealed anchorage, embedment or substrate adequacy | Installation records, approved design and appropriate exposed/instrumented inspection |
| Hollow tile or tile adhesion | Appropriate physical/acoustic/bond examination and its own validation |
| Floor slope | A calibrated level/survey or independently validated measurement system |
| Tile lippage | A suitable physical height/step measurement; planar RGB edge spacing does not measure height |

These are outside the implemented RGB decision scope. The application does not declare Saudi-code compliance or workmanship acceptance.

## Qualification protocol

Each released category needs its own frozen independent source/site evaluation, at least 50 positive and 50 negative originals, localization precision≥95% and recall≥90%, appropriate rights, representative real-video event checks, sustained preview≥15fps and sampled result-age p95≤1second. Metric fault decisions also need independently measured real installations. Shadows, stains, veins, reflections, good installations, occlusion, rotation, task switching and repeated-event behavior must be evaluated at their proper level; software fixture tests do not supply these real-world counts.

The [data audit](construction-expansion-data-audit.md), [implementation index](construction-expansion-implementation.md) and per-model records distinguish what was actually acquired/trained/tested from what remains a gap. Category names and suggested checks never count as trained capabilities.
