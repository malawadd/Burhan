# AV-Water Damage: public-data access and semantic audit

Date: 2026-10-21. **Not training-ready: no image/annotation archive acquired.**

## Publisher metadata and count discrepancy

The [AssuraLabs project overview](https://universe.roboflow.com/assuralabs/av-water-damage) declares object detection, CC BY 4.0, 1,899 images and five labels: crack, stain, mold, damp and dampness. Its citation dates the dataset to 2024. The project has no descriptive annotation guide and advertises zero published models. The general hosted detection example on the page is not an available trained dataset-specific model.

The [sole version, v1](https://universe.roboflow.com/assuralabs/av-water-damage/dataset/1), generated May 7, 2024, reports **1,145 images**: 747 TRAIN, 215 VALID and 183 TEST. Its preprocessing is auto-orientation and stretching to 640×640; it declares no augmentation. Consequently, **1,899 project images must not be presented as 1,899 acquired training examples**, and v1 is not an export of that full advertised count.

These are publisher-page claims read through the web tool, separately from the failed direct access below. No-augmentation does not establish unique original photographs, independent walls or independent sites. The export's actual files and duplicate groups remain unknown.

## Actual legitimate access attempts

Ordinary Edge navigation to the publisher's YOLOv8 download page returned **HTTP 403**, with a challenge page. No challenge was bypassed. Bounded standard-library GET requests also returned 403 for the overview, v1 and its publisher download route.

The official API version and YOLOv8 export routes returned **HTTP 401**, explicitly requiring an API key. Route structure follows the [official SDK reference](https://roboflow.github.io/roboflow-python/core/version/). No key, account, session credential or authenticated state was sought or supplied. No account was created, no owner media uploaded, and no cloud inference called. Example images and thumbnails were not downloaded as a substitute dataset.

| Artifact sought | Result |
|---|---|
| Actual original/export images | 0 acquired |
| Annotations and version-specific class YAML | 0 acquired |
| Accepted complete training archive | None |
| Dataset-specific downloadable model | None established |

The reproducible bounded probe is **scripts/audit_av_water_damage.py**. It records five HTTP statuses, body hashes, static publisher metadata, the original-versus-version discrepancy and explicit training blockers in **output/evaluation/construction-water-data/av-water-damage-v1-audit/audit.json**. Response bodies are metadata/error pages, not images or an archive. A repeated probe must use a new output directory; it cannot overwrite the earlier receipt.

## Semantic and validation blockers

- The relationship between **damp** and **dampness** is undefined. Do not merge their labels, infer separate physical conditions, or assume the current five-label overview matches v1's class order without its actual annotation files.
- RGB stain/damp labels could support visible appearance detection, but cannot provide moisture percentage, active hidden water, leak cause, waterproofing adequacy or a fungal diagnosis.
- Object detection boxes are not genuine segmentation masks or precise damage boundaries. The labels cannot be converted into damage masks and presented as mask supervision.
- CC BY 4.0 is the publisher's declared dataset license. Original-image origins and any third-party rights have not been independently audited.
- Actual source photographs, annotation completeness, empty/healthy examples, derivative groups and physical wall/site identity are unverified. The advertised split cannot establish leakage-free physical-site evaluation.
- Before any fit, acquire a legitimate complete export, inventory original versus derived images, audit exact/perceptual duplicates and label semantics, then freeze site/group splits and independent healthy/difficult negatives. Keep the four owner clips outside training and calibration.

This source remains an **unavailable acquisition lead**, not new training data, a trained wall-water capability or evidence that the owner's reported fault is now detected.
