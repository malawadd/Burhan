# Construction expansion: public data audit

Date: 2026-10-21. This audit separates acquired annotated photographs from publisher descriptions. No private owner footage was uploaded, fitted, calibrated or used to choose thresholds.

## SDNET2025: acquired, annotation-audited, research only

- Publisher: [SDNET2025 version 1](https://data.mendeley.com/datasets/cxnj6cp4rb/1), Faezeh Jafari and Sattar Dorafshan. Record license: **CC BY 4.0**. See [the authors' data article](https://pmc.ncbi.nlm.nih.gov/articles/PMC12670936/).
- The collection mixes field photographs of steel connections with laboratory photographs, including physically prepared missing/loosened connections. These are photographs of real hardware, not rendered faults. File-level field/laboratory and physical-site identities are not supplied, so this audit does not claim every defect arose naturally in service.
- Browser file details identify `Dataset.zip`, 3,044,451,136 bytes; publisher SHA256 `5be49a5ac5fa9c7ea5b3d0de03eca8b9f3d4f337b96a89c4c7f9d7ecce385ec0`.
- Acquired the central directory, eight annotation files, and **826 unmarked publisher 640×640 photograph derivatives** through verified HTTP206 ranges. Full 3GB archive was not downloaded, so its full-archive SHA256 is **not claimed verified**. Each downloaded member passes original ZIP CRC/length verification and has its own SHA256 receipt.
- The 2,708-member inventory contains 324 loosened, 200 missing and 302 fixed photograph views, original RAW files, resized counterparts and annotated overlays. Only one unmarked resized view per photograph enters preparation. The 1,048 annotated overlays are excluded. RAW names do not establish a verified correspondence with renamed defective photographs, so they are not mixed into training or evaluation.
- Actual `Loosened.json` and `Missing.json` contain **970 polygon instances: 533 `Loosen`, 437 `Missing`**. The missing-photo annotation file includes two `Loosen` instances; they retain their actual label. These counts differ from the data article's 956 instances. No annotations were silently deleted to match the paper.
- The acquired files provide **no annotated corrosion category**, despite the broader record description. Corrosion detection is not enabled from this dataset.
- Canonical outputs are `visibly_unseated_connection` and `fastener_absence_appearance`. They preserve the visible publisher task. They do not establish torque, embedment, individual missing-bolt versus missing-nut identity, or the approved fastening schedule.
- The 302 fixed-connection photographs are target-negative views, not annotated fixed-bolt boxes. Absence of annotations on defective photographs is not treated as an assertion that every other physical component is healthy. A subsequent TRAIN visual audit found that cross-class completeness of the two defect annotation files is unverified; numerical unmatched-prediction counts must not be called proven physical false alarms. Only 30 Fixed photographs land in TEST, below the 50-negative gate.

### Frozen split and limitations

Preparation creates aligned boxes, genuine polygon masks, and YOLO labels. Before any neural model, a conservative graph joins exact duplicates, dHash distances ≤8, or small-thumbnail mean absolute differences ≤8. Entire connected components receive a deterministic 70/15/15 hash split. Result: **601 TRAIN / 127 CAL / 98 TEST photographs**, from 780 duplicate/near-duplicate components.

**Physical-site/connection independence is not verified.** The publisher has no usable per-photo physical connection/site identity table in this archive. Different graph components can still show the same structure. This split is an exploratory image-group evaluation, not the required independently verified 50-positive/50-negative deployment validation. No detector is promoted on these results.

Artifacts:

- `datasets/construction/expansion-public-v1/sdnet2025/zip-inventory.json`
- `datasets/construction/expansion-public-v1/sdnet2025/range-ledger.json`
- `datasets/construction/expansion-public-v1/sdnet2025/image-assets.json`
- `datasets/construction/expansion-public-v1/sdnet2025/prepared/manifest.json`

## Wall finishes

[PHELE version 4](https://data.mendeley.com/datasets/yzsxcvc8dp/4) is CC BY 4.0. The record discusses dampness/mold and peeling paint, but actual archive class names take precedence over this description. Its [separate acquisition/training record](construction-expansion-phele.md) documents which labels and photographs were actually obtained; labels are never broadened into measured wall humidity.

[CUHK indoor_defects](https://universe.roboflow.com/cuhk-ldjub/indoor_defects-logzf) advertises 209 images, instance segmentation, and four labels: crack, paint_bubble, paint_peel, spalling. Its page declares CC BY 4.0. The annotation archive, source groups, and original-versus-augmented identities are not yet acquired/verified. The public API returns HTTP401 requiring an API key; ordinary local page requests return HTTP403, while the browser/primary web view can read metadata. No account credentials were sought and no serverless inference was used. **No paint-blister model is trained from metadata or example thumbnails.**

## Installed tiles and remaining categories

[UC-Tile research](https://doi.org/10.1016/j.autcon.2025.106001) concerns segmentation and tile geometry; a paper link is not an acquired installation dataset. No independently measured real installed-floor joint dataset is accepted by this audit. Factory tile-surface photographs do not validate installed grout spacing, lippage, missing grout or crooked floor grids.

A targeted follow-up search checked installed-tile/grout datasets against the [primary UC-Tile article](https://www.sciencedirect.com/science/article/pii/S092658052500041X). It describes 1,486 tile-paving photographs and a camera plus point-laser positioning setup, but this audit has no verified downloadable annotation archive or real installed-joint fault/measurement holdout. The paper's reported segmentation mIoU is not installation-fault recall and is not copied into product benchmarks. Paid custom video collection and unlabeled action footage are not substitutes for the public-only fault/measurement data required here.

The UI may offer calibrated geometry as a research measurement workflow, but millimetre fault decisions remain unavailable until independently measured installations validate the method. Missing grout, chipped installed tile edges, sealant breaks, paint runs and coating coverage each require their own accepted labels and genuine negatives; they are not aliases for cracks or spalling.

## Reproduction and acquisition blockers

Run from the project root:

```powershell
.venv\Scripts\python.exe scripts/audit_construction_expansion_sources.py probe
.venv\Scripts\python.exe scripts/acquire_sdnet_expansion.py inventory
.venv\Scripts\python.exe scripts/acquire_sdnet_expansion.py batches --selection datasets/construction/expansion-public-v1/sdnet2025/image-selection-before-pixels.json
```

The safe archive inventory command in `audit_construction_expansion_sources.py` accepts a local publisher ZIP, rejects unsafe member names/encryption, checks the known full SDNET checksum, and never extracts or starts training by itself. The remote range utility rejects ignored ranges, changing archive sizes, oversized responses and incorrect CRCs.

Initially, Python `requests` received HTTP403 Cloudflare challenges. Browser metadata was readable, but browser downloads timed out twice without a download event. Normal standard-library `urllib.request` requests with an honest research user agent subsequently succeeded through the publisher's normal public-file redirect to its official S3 object. No CAPTCHA, login, paid service, private upload or security bypass was used. All successful bytes and resolved URLs are recorded in the range ledger.

## Initializer provenance

The detector comparison uses fresh official COCO initializers, not inherited road/crack weights with new class names:

| Initializer | Official source | SHA256 |
|---|---|---|
| YOLOv8n | [Ultralytics assets v8.4.0](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt) | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` |
| SSDLite320 | [TorchVision model distribution](https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth) | `a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2` |

New task classifier heads are initialized before fitting. SDNET's data license does not replace Ultralytics AGPL conditions or TorchVision/model pretraining conditions. These are local research artifacts; this audit is not a commercial-rights clearance.

## Completed local experiments

Actual downloaded photographs and publisher annotations were used in the following experiments; none passes the category release requirements:

- [SDNET box comparison and provenance](construction-expansion-sdnet-benchmarks.md): fresh YOLOv8n versus SSDLite320, before/after metrics, calibration selection, Fixed-only negative views and CPU-ONNX parity.
- [Extended SDNET fit](construction-expansion-sdnet-v2-results.md): eight logical epochs at 512px and actual deployed-adapter verification. Unseated-connection recall improves to 36.92% with 79 annotation-unmatched predictions, but remains far below the required performance. Sparse absence predictions retain only 5.63% recall.
- [PHELE wall-finish acquisition and two-model comparison](construction-expansion-phele.md): actual peeling-paint and Mold labels, source and split limitations, calibration and held-out results. Mold appearance is not relabeled as measured humidity.

The genuine SDNET polygon annotations also support the separate LR-ASPP / YOLOv8n-seg mask comparison. Pixel/region outputs are evaluated separately from detector boxes. More labels in the catalog, a completed training run or a successful ONNX export never establishes a released construction check.
