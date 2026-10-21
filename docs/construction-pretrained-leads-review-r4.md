# Additional pretrained construction leads: static review only

20 October 2026. No checkpoints were downloaded, unpickled, executed, exported or measured in this review. GitHub API tree metadata and small author source files are saved under `datasets/construction/pretrained-lead-research/`. File presence and repository claims do not establish local accuracy, label correctness or speed.

## Infrastructure-defect YOLOv8

[Author repository](https://github.com/RBoabeng/infrastructure-defect-detection-yolo), pinned commit `bc43be2cd5082d0c40860c5ea53e418e6b537afb`, contains the actual binary `assets/yolov8_infrastructure_defects.pt`: **6,248,874 bytes**, Git blob SHA1 `4ab44467e7c474ac47ab2717f8c5f97d2ec6a6bf`. This is a Git object identifier, not a locally verified checkpoint SHA256.

The repository includes an MIT license. Its data download script points to Roboflow workspace `a188370`, project `infrastructure-defects-detection`, version4, using a user API key. The README explicitly identifies exclusively spalling validation and biased predictions. Its CPU timing is an author claim on different hardware. No local benchmark is inferred.

Crucially, **`config/data.yaml`, `src/train.py` and `src/augment.py` are zero-byte files** at this revision. Therefore the repository does not expose reproducible class order, training configuration or transformations through those files. The Roboflow dataset rights/source grouping and relationship between this binary and the empty training code remain unverified. A repository MIT file alone does not settle rights in upstream software and training images. This is a possible small spall-weight lead, not an established multi-condition detector or production candidate.

## Crack YOLOv12 with attention

[Author repository](https://github.com/Hhhpraise/crack-detection-benchmark), pinned commit `ee09532660bbe437891691484cc31984e68cd50f`, has these actual binary assets:

| Asset | Bytes | Git blob SHA1 |
|---|---:|---|
| `weights/yolo12s_cbam_ca_crack.pt` |18,937,545|`adc1d9155a12f2da397e8bb9ec8ff95e5ffb2603`|
| `weights/yolo12s_seg_cbam_ca_crack.pt` |20,473,074|`ad58d681993d03847dc3959d7f4939ab47238a3f`|

The author provides MIT repository licensing, custom CBAM/CoordinateAttention source, a20-image demo and training/inference scripts. The full dataset's Kaggle/Zenodo links are placeholders, so its provenance and independent source groups could not be verified here. It targets cracks, without a demonstrated spalling class. The README's benchmark descriptions remain author claims.

Static source findings matter for any later experiment:

- [Detection training](https://github.com/Hhhpraise/crack-detection-benchmark/blob/ee09532660bbe437891691484cc31984e68cd50f/yolo/detection/train.py) registers custom attention modules, stores a full Python model object, and reads pretrained weights through `torch.load` without an explicit restricted policy. These are instructions to review, not a loader to execute as-is. Default training uses192-pixel input, whereas the detection demo uses224; native video hairline resolution is unproven.
- [Segmentation organizer](https://github.com/Hhhpraise/crack-detection-benchmark/blob/ee09532660bbe437891691484cc31984e68cd50f/yolo/segmentation/organize.py) explicitly converts five-value detection boxes into **rectangular polygons**, retains existing polygons, and creates empty labels when annotations are missing. Consequently a supplied segmentation checkpoint cannot be credited with true pixel masks without auditing its actual training labels. A rectangle is not hairline ground truth.
- That organizer randomly shuffles individual images into80/20 partitions. It does not enforce physical structure/capture separation. Its results cannot be treated as source-disjoint video validation.

Neither lead is added to registered models or UI capability claims. A later bounded candidate would first establish exact classes, allowed source/weight use, restricted checkpoint inspection and deterministic export; then freeze calibration rules on our unchanged real camera inputs, retain all negative failures, and measure actual serving performance. It must not use its author's demo frames as independent test evidence.
