# Separate real-positive training contingency

20 October 2026. This is an **unused acquisition/preparation artifact at creation**, prepared while the first camera-only construction candidate trained. It does not change that experiment, its YAML, model, calibration images, heldout images or thresholds. Any later use must be recorded as a separately versioned continuation **before opening heldout results**. No neural inference or GPU work was used to prepare these assets.

The [DACL10k original authors](https://github.com/phiyodr/dacl10k-toolkit) provide real inspection photographs and multilabel polygons under **CC BY-NC4.0**. The author-linked [Voxel51 mirror](https://huggingface.co/datasets/Voxel51/dacl10k), pinned at `7aa4a3e6fd8818d94bdf51b9d02ea0443808f68b`, supplied the original RGB JPEGs and complete polygons. The mirror's conflicting CC BY4 metadata does not override the authors' noncommercial restriction. No author's model weights are used by this preparation; any downstream trained model must retain its own software/initial-weight lineage and the data-use caveat.

The frozen procedure selects **40 Crack/ACrack-positive and 40 Spalling-positive distinct official-TRAIN photos** by ascending SHA256 of `dacl-round3-positive-contingency-20261020:` plus original name. The crack-family cohort is selected first; spalling excludes already selected names. All95 previously downloaded DACL source names and original RGB-byte hashes are excluded, as are exact duplicates among new photos. Public validation/test photos are not candidates. Physical bridge IDs are unavailable, so source-image/hash separation does not establish bridge independence or absence of near-duplicate structures.

The full candidate ordering, seed, exclusions and training-only crop rule were written to `protocol-before-download.json` before acquisition. The final [asset manifest](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/dacl10k/round3-positive-contingency/asset-manifest.json>) has SHA256 **`2dc962b359bba28ba8eb1871184c2c2c5d8aa841cf4540d393fccd150276c360`**. It contains paths, original/derived SHA256 values, original source URLs, category names, complete source annotation pointers, per-polygon IDs and crop geometry.

## Actual geometry and targets

For each selected photo, the lowest stable publisher-polygon ID in the requested cohort determines the target. The largest exact native4:3 crop is centered around that polygon's bounding-box center and clamped within the original image. At least one true target pixel must remain. This ground-truth-centered choice is **training-only**; it cannot be used to report evaluation accuracy. The crop becomes grayscale, resizes bilinearly to640×480 and is saved as a lossless RGB PNG with equal R/G/B channels. No damage is generated, pasted or simulated. Original RGB JPEGs remain unchanged under `original/images`.

All original polygons, including non-target conditions, remain in `original/annotations`. Every Crack/ACrack/Spalling polygon with actual nonzero pixels inside the crop is rasterized using the publisher-style fill convention. Its clipped **source-mask** bounding box is continuously projected into the resized image for YOLO targets, so a thin source annotation is not dropped solely because nearest-neighbor reduction might erase its semantic mask. Independent per-class nearest-neighbor semantic masks are also retained for auditing; they are not native instance annotations. Boxes are derived from semantic polygons and can enclose background within a curved or fragmented region.

The class mapping matches the construction candidate: **0=spalling;1=crack**, where both original `Crack` and `ACrack` map to crack. All visible target polygons are retained regardless of acquisition cohort. Other damage categories remain in the complete original annotation, but are not additional detector classes or healthy negatives.

| Artifact | Actual count |
|---|---:|
| Original official-TRAIN RGB photos / complete polygon JSONs | 80 /80 |
| Derived grayRGB640×480 crops / YOLO label files | 80 /80 |
| Retained Spalling polygon boxes | 125 |
| Retained Crack polygon boxes | 65 |
| Retained ACrack polygon boxes | 1 |
| Total retained training boxes | 191 |
| Source-visible annotations whose nearest resized mask vanished | 0 |

No stills are converted into videos. These assets train visual features; they do not validate camera motion, night timing, real-video localization or Saudi acceptance.

## Verification and visual crop QA

The lightweight audit passed in5.3seconds: source/annotation/derived hashes, all95 exclusions, unique source names/hashes, official-TRAIN provenance, equal grayRGB channels,640×480 geometry, finite positive normalized boxes, class mapping and retained-label counts. The source-image hashes in the final manifest define exact-byte disjointness. Hashes cannot prove physical-site independence or perfect publisher annotation completeness.

The first fixed example in each cohort was visually inspected. Both contain visible real damage, with source-derived GT boxes aligned to the relevant regions. These previews are **training-target QA, not model output or accuracy evidence**.

![First real crack-family training crop and publisher-derived target boxes; not model output](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/dacl10k/round3-positive-contingency/qa/crack_family-first-derived-gt.jpg>)

![First real spalling training crop and publisher-derived target boxes; not model output](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/dacl10k/round3-positive-contingency/qa/spalling-first-derived-gt.jpg>)

The [QA report](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/dacl10k/round3-positive-contingency/qa/report.json>) records counts and manifest identity. The separate asset root is `datasets/construction/dacl10k/round3-positive-contingency`; original files, derived targets and QA stay within it. Reproduction does not import targets into any active training dataset:

```powershell
.venv\Scripts\python.exe -X utf8 scripts/prepare_dacl_positive_contingency.py
.venv\Scripts\python.exe -X utf8 scripts/audit_dacl_positive_contingency.py
```

Single-process Pillow preparation was used, without Torch, OpenVINO, neural inference or artificial damage. No active inference FPS was benchmarked during preparation.

## Separate additional real-night contingency

The construction agent also requested five additional genuine Dec27 ev-CIVIL sequences for a possible later training-only continuation: `27_crack_6`, `27_crack_7`, `27_crack_8`, `27_crack_13`, `27_crack_15`. These are the complete camera H5 + frame-label NPY + event-label NPY triples available in the saved publisher ZIP inventory among the requested IDs6–20/23–31, excluding exposed10/89, calibration21/22 and heldout32/43/44. IDs9/12 lacked frame-label members and were not acquired. Metadata selection does not prove how many actual timestamp-qualified camera exposures they contain.

Original15 members were acquired by approximately2.38MB of compressed ZIP byte ranges from the [official ev-CIVIL publisher archive](https://doi.org/10.11583/DTU.25924228), under **CC BY4.0**. Every member's original ZIP size and CRC32 were verified; actual uncompressed-byte SHA256 values and source-member provenance are recorded. The full10.87GB archive checksum was not verified because the archive was not downloaded.

Files remain under `datasets/construction/evcivil-round3-contingency/raw/night_dec27/`, separate from active data. The [manifest](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/evcivil-round3-contingency/night-dec27-asset-manifest.json>) has SHA256 **`9b876fac3b7f2a103527e13fa6b355bcc5e0ce96d83c212cdd2b8f9cdb85f558`**. `night-dec27-acquisition-protocol.json` preserves exact membership and exclusions. No H5 exposure decode, timestamp association, label conversion, video rendering or NN was performed by this acquisition; these remain the training agent's responsibility if it versions a continuation. Data were unused at acquisition and did not change calibration, heldout or the current experiment.

```powershell
.venv\Scripts\python.exe -X utf8 scripts/fetch_evcivil_night_contingency.py
```

## Forty additional training closeups of the same sources

The construction agent identified a valid limitation of the largest-native4:3 crops: a high-resolution physical hairline can lose contrast when the whole view is reduced to640×480, even if its published semantic polygon still occupies pixels. Retaining a bounding box does not prove that its physical stroke remains visible. A separate deterministic40-view closeup derivation therefore uses **the same40 crack-family source photographs**, adding zero new independent source photos. The original80-view manifest and files remain unchanged.

For the lowest stable-ID crack-family polygon, find the rasterized source target pixel nearest its bbox center, with row-major tie breaking. Center a native exact4:3 crop around that true target pixel, using1.5× bbox context, bounded to native320×240 minimum and960×720 maximum, and clamp within the image. The640×480 output therefore downsamples by at most1.5× per dimension, instead of reducing a whole multi-megapixel view. All source Crack/ACrack/Spalling polygons actually visible in the crop are retained through the same clipped-source-mask box projection. The crop rule uses GT only for **training**, never evaluation or deployment input selection.

The [closeup manifest](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/dacl10k/round3-crack-closeup-contingency/asset-manifest.json>) has SHA256 **`3d376743bb6a75525c316e193c5298628fb720286271754dca774e42cca48d18`**. The records preserve original source names/hashes, original RGB and complete polygons, and add suffixed `-closeup` source-view/image identities.40views contain76 source-derived boxes:29Spalling,46Crack and1ACrack. These are overlapping training views of existing sources, not additional independent data or model accuracy evidence.

Single-process preparation completed in9.4seconds and hash/class/gray/geometry/box QA passed in1.6seconds. The original80-manifest SHA remained unchanged. I visually inspected the first closeup's clear horizontal crack and the second source1752's finer light crack strokes; no claim is made that all40 physical hairlines are easy to see. The first deterministic example is shown below, with GT boxes explicitly distinguished from model output. No model ran in this preparation.

![First actual training closeup and publisher target boxes, not model output](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/dacl10k/round3-crack-closeup-contingency/qa/first-closeup-gt.jpg>)

```powershell
.venv\Scripts\python.exe -X utf8 scripts/prepare_dacl_crack_closeups.py
.venv\Scripts\python.exe -X utf8 scripts/audit_dacl_positive_contingency.py --closeups
```
