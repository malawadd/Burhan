# Independent audit of four local construction recordings

**Integrity audit PASS: 36,701 checks, no failures.** All878 original camera exposures were checked without running a model, uploading video, training, changing thresholds or adding annotations.

Auditor: `scripts/audit_user_construction_videos_round4.py`.
Receipt: `output/evaluation/user-construction-videos-round4/independent-private-baseline-audit-v1.json`, SHA `0dff579e8c01e228426cce83d435f9f727554c2ba40aa02dbb1f32e414ab98cc`.

## What was independently verified

- All four original file SHA256 identities and byte counts match the frozen inventory and report source identities; source files remained unchanged.
- OpenCV5 FFmpeg with one requested decoder thread independently reproduced **every native auto-oriented BGR frame hash**,878/878. Dimensions and contiguous frame indices match.
- Independent PyAV source decoding reproduced every original PTS/timebase/count. Every derived MP4 retained the full exact rational relative timeline and count, with no holes; derived file SHA and1400×1240 panel dimensions match.
- All878 saved per-frame JSON records equal their corresponding report records. Every NPZ has exactly the expected mask keys, with no hidden ground-truth arrays. Every mask's packed SHA, native H×W, uint8 packing length, positive-pixel count and zero padding bits match. **2,515 actual saved masks passed.**
- Per-model class prediction counts recompute exactly to the declared summary. Flattened predictions preserve all model records. Native boxes, class IDs, class filters and finite scores are consistent.
- Every frame explicitly has `evaluated:false`, `expected:[]`; every prediction is `unscored`, with no invented GT identity/match. Accuracy summary fields remain null. Both mask models retain masks; the box-only wall model has no fabricated mask.
- The executed source archive matches the baseline plan and its source pins. The three model asset file SHAs match the frozen plan.

This verifies saved-artifact integrity and output presence. Without rerunning NN it does not independently reproduce model calculations; without external frame annotations it cannot measure recall, precision, false alarms, structural cause or compliance.

## Actual prediction presence

These are frames containing at least one **unscored model candidate**, not frames with verified damage.

| Local recording | Original frames / raw hashes matched | Author mask model: candidate frames | Wall box model: candidate frames | Broad mask model: candidate frames | Saved native masks |
|---|---:|---:|---:|---:|---:|
|VID-20261020-WA0000|269 /269|105|217|269|953|
|VID-20261020-WA0001|207 /207|13|35|81|132|
|VID-20261020-WA0002|330 /330|8|35|245|1,225|
|VID_20261020_075653953|72 /72|0|3|66|205|
|Total|878 /878|126|290|661|2,515|

The following class totals count individual emitted predictions; multiple predictions may occur in a single frame. They are **not verified fault counts**.

| Recording | Author masks | Wall appearance boxes | Broad masks |
|---|---|---|---|
|WA0000|crack122|bulge221|crack652; cavity157; alligator_crack22|
|WA0001|crack13|abscission3; bulge34|rust6; cavity93; washout7; crack5; spalling5; alligator_crack1; efflorescence2|
|WA0002|crack8|crack8; abscission1; leakage3; bulge23|efflorescence7; washout12; spalling19; weathering6; crack10; cavity1,045; rust108; wetspot10|
|075653953|none|leakage3|rust83; wetspot65; weathering55; alligator_crack2|

In particular, a label such as `wetspot`, `cavity`, `bulge` or `rust` is an appearance prediction requiring human verification. None establishes measured moisture, hidden cavities, surface depth, railing-fixing adequacy or Saudi code acceptance. A model emitting no candidates is also not an inspection pass. The original owner-reported concerns remain context, not fabricated frame ground truth.

## Scope and timing

The auditor reads originals, reports, mask archives and encoded video timelines only. It imports no project model/profile and performs no NN. OpenCV processing and both FFmpeg/PyAV decoder passes are constrained to one thread; media/hash auditing took approximately38seconds while other authorized TRAIN work could run. This is **not inference FPS** or a capacity benchmark. The initial failed baseline attempt and all original artifacts were preserved; the audit writes only its new receipt. Root owns visual QA and any UI/registry integration.
