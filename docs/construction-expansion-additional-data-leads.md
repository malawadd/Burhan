# Additional construction data: bounded source follow-up

21 October 2026. This follow-up checks real installed-surface data for tile joints, missing grout, chipped installed tiles, efflorescence, sealant and paint blisters. It adds an actual efflorescence archive audit. **No new model was fitted, no detection category was released, and no private video was uploaded.**

## Efflorescence: an actual licensed, single-site lead

The authors' [Batalha Efflorescence Dataset](https://zenodo.org/records/21453438), deposited by Inês Bourgeois, Dener Silva and Hugo Rodrigues, provides real limestone heritage-surface photographs from the Monastery of Batalha, Portugal. Its declared task is binary efflorescence/background segmentation. The [official metadata API](https://zenodo.org/api/records/21453438) explicitly identifies **CC BY 4.0**; the rendered record's empty-looking license section is insufficient by itself. Attribution to these creators and the [associated study](https://doi.org/10.1016/j.jobe.2026.116924) must accompany reuse.

The publisher archive is `Efflorescence_Limesstone.zip`, **662,950,017 bytes**, with publisher MD5 `b095764ee06a7b52bdf6e7ec31b0d7ee`. The entire archive was not downloaded, so that whole-archive checksum is **not claimed verified**. Public HTTP206 ranges were used, with a strict 80 MB budget, original ZIP CRC/size verification and per-member SHA256 receipts.

Actual central-directory inspection finds **403 non-directory entries**, including resource forks and `.DS_Store`. Excluding those leaves **100 photographs and 99 PNG masks, with 99 matching stems**. This is acquired archive evidence, rather than an inferred image count from a publication.

All 99 masks and four original photographs were acquired. The four stems were frozen by the smallest SHA256 of `audit-v1:` plus the paired filename stem before pixels were decoded. They are an audit sample, not a training split or a performance evaluation. The complete mask/pair measurements, missing-pair identity, received-byte total, and source member hashes are in the artifacts below.

The actual transfer was **37,800,860 bytes**, including inventory and repeated ZIP-directory ranges. Every acquired mask is mode `L` with exactly values **0 and 255**, no empty/full masks, and nonzero area between **1.16% and 23.56%**. No mask file is an exact duplicate of another acquired mask. All four sampled photograph/mask dimensions match; the photographs range from 4032×2268 to 6000×4000. The unpaired original is `IMG_5239`.

Visual inspection of all four paired audit panels confirms real masonry photographs with aligned foreground regions on pale deposits and some white joint lines. **These overlays show publisher annotations, not detector predictions or independently confirmed salt chemistry.** No numeric class-legend/readme file was found in the archive inventory; 255 is treated as foreground for this audit, with the semantic description supplied by the record. The inclusion of white mortar/joint areas needs a conservation-domain annotation review before a future salt-specific fit. Do not label every acquired mask a chemically verified efflorescence instance.

The source provides **one known physical site**. Camera filenames are not an independently verified surface identity table. There is no accepted site-disjoint split, healthy-wall negative set, real-video event annotation, local model or laptop speed benchmark for this source. Limestone efflorescence also does not establish detection on painted plaster, measured moisture, hidden leakage, missing grout or ceramic spacing. Efflorescence therefore stays **unavailable for inspection**; this lead improves source availability, not capability validation.

Artifacts:

- `datasets/construction/expansion-public-v1/additional-leads/batalha-record.json`: retained official rights/file metadata.
- `datasets/construction/expansion-public-v1/additional-leads/batalha/zip-inventory.json`: complete member names, compressed/original lengths and CRCs.
- `datasets/construction/expansion-public-v1/additional-leads/batalha/selection-before-pixels.json`: frozen photo audit selection.
- `datasets/construction/expansion-public-v1/additional-leads/batalha/range-ledger.json`: actual ranges, response hashes and received bytes.
- `datasets/construction/expansion-public-v1/additional-leads/batalha/image-assets.json`: 103 acquired members with SHA256s.
- `datasets/construction/expansion-public-v1/additional-leads/batalha/actual-file-audit.json`: mask values, sizes, occupancy, exact mask duplicate counts and paired original dimensions.
- `datasets/construction/expansion-public-v1/additional-leads/batalha/actual-photo-mask-audit.jpg`: original/publisher-mask overlays, explicitly **photographs**, not model detections or video.

Reproduce from the project root (the script retrieves the official metadata API when its retained record is absent):

```powershell
.venv-roads-export\Scripts\python.exe scripts/audit_additional_construction_data.py
```

The script requires the expected publisher license, freezes the deterministic selection, enforces the range budget through the shared acquisition helper, verifies every newly acquired member against its source CRC and checks sampled photo/mask dimensions. It neither starts training nor edits runtime models.

## Sealant: relevant real data, not a public downloadable archive

The primary [October 2026 building-sealant study](https://www.mdpi.com/2075-5309/16/19/3955) describes manual binary masks for 120 annotated images, drawn from real outdoor exposure specimens and buildings in Ottawa. Its **Data Availability Statement says the data are available on request because of privacy restrictions**. No author was contacted and no public annotation archive was acquired. The described task is sealant cracking; it does not supply missing-sealant references or tile-grout installation labels. Related images can share specimens, which would require grouping before any new split.

## Tiles and paint blisters: concrete unresolved gaps

| Primary source | Verified scope | Why it does not yet supply the required installation validation |
|---|---|---|
| [UC-Tile authors' university record](https://researchportal.hkust.edu.hk/en/publications/automatic-tile-position-and-orientation-detection-combining-deep-/), [paper DOI](https://doi.org/10.1016/j.autcon.2025.106001) | 1,486 tile images for segmentation, edges/corners and paving-robot positioning under different capture conditions | No free, direct author archive with verified rights, original/site identities and annotated installed-floor grout defects was established in this follow-up. Positioning results do not establish faulty joint widths, missing grout or installed chipped edges. |
| [CUHK indoor_defects](https://universe.roboflow.com/cuhk-ldjub/indoor_defects-logzf) | Publisher page lists 209 indoor-apartment images, instance segmentation and `paint_bubble`, `paint_peel`, crack and spalling, under CC BY 4.0 | Actual annotation archive and original/site grouping remain unacquired; the existing direct API probe requires an API key. A readable page or sample thumbnail is not a mask dataset. No account or cloud inference was used. |
| [Coating Defect Detection Dataset](https://zenodo.org/records/15166405) | Publisher describes 2,697 wind-tower painting-part images, YOLO boxes and acquisition metadata; 17.8 GB archive | Industrial wind-tower parts are outside this installed-building-wall task. Class semantics and building transfer are not assumed; the archive was not downloaded as wall-blister training data. |
| [DATN ceramic-tile-defects](https://universe.roboflow.com/datn-vas6s/ceramic-tile-defects) | Page describes hole, line and edge-chipping for factory/retailer tile quality | Before-installation surface defects cannot substitute for installed-floor joint/grout evidence. Marketing suggestions about construction use do not establish installed-scene annotations or measured spacing. |

The bounded search included author/publisher records and queries for installed tile/grout segmentation, missing-grout annotations, installation-defect datasets and building paint-blister data. Contractor galleries, inspection report illustrations, synthetic scenes and reposted dataset cards were not accepted as annotated licensed originals. This is a record of what was **verified in this follow-up**, not a claim that suitable data cannot exist elsewhere.

The operational gaps remain independently measured real installed joints, intentional decorative patterns/openings and correct installations; annotated missing-grout and chipped-installed-tile examples; and source-grouped paint-blister data. No new detection label is enabled from these leads.
