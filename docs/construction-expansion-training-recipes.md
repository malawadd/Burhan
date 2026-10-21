# Construction expansion training recipes

21 October 2026. All fits run on the laptop from real public photographs and their acquired annotations. No owner video frame enters TRAIN, calibration, checkpoint selection or threshold selection. Lower loss and successful ONNX export are not inspection qualification.

## Frozen experiments

| Dataset / task | Architecture and fixed recipe | Entry point / full record |
|---|---|---|
| SDNET Loosen + Missing boxes | YOLOv8n,320px,2epochs,freeze10; SSDLite320,frozen backbone,200updates | `scripts/train_sdnet_expansion.py`; [results and execution correction](construction-expansion-sdnet-benchmarks.md) |
| SDNET stronger box candidate | YOLOv8n,512px,batch4,freeze4,8totalepochs; final checkpoint, per-class CAL thresholds | `scripts/train_sdnet_expansion_v2.py`, followed by the checkpoint-bound performance continuation below |
| SDNET genuine polygon masks | LR-ASPP MobileNetV3,320px,batch8,frozen backbone,200updates; weighted cross entropy+Dice | `scripts/train_sdnet_masks_expansion.py --model lraspp`; [mask comparison](construction-expansion-sdnet-masks.md) |
| SDNET genuine polygon instances | YOLOv8n-seg,320px,batch8,freeze10,2epochs,no image augmentation | Same script with `--model yolo`; actual vertices, not filled box supervision |
| PHELE Peeling paint + Mold boxes | YOLOv8n,384px,freeze4,8epochs; SSDLite320,frozen backbone,200updates,batch8 | `scripts/train_phele_expansion.py`; [recipe, commands and failed results](construction-expansion-phele.md) |
| MBDD2025 leakage/stain appearance | SSDLite320,new background+one-target head,frozen backbone,200updates,batch8 | `scripts/train_wall_stain_expansion.py`; [data and failed results](construction-expansion-wall-stain.md) |

Use `.venv-roads-export/Scripts/python.exe` for training/export and `.venv/Scripts/python.exe` for deployed inference/tests. The runtime requires no Torch installation. Dataset preparation/acquisition commands and source/license manifests are linked from the [public-data audit](construction-expansion-data-audit.md).

Each experiment writes a protocol before its first forward, including seed, source/prepared manifest hashes, initializer checksum, script version and fitting budget. Scripts that protect a frozen output refuse to overwrite it. To conduct another experiment, declare a new version/destination rather than editing a frozen result or rerunning until a desirable TEST score appears.

## CPU performance continuation

The TRAIN-only probe compares identical two-class YOLOv8n inputs/initial parameters at 512px/batch4. It performs no optimizer updates. Four CPU threads plus explicitly converted channels-last tensors reduce median forward/backward time from 2.625s to 1.226s (2.142×); matching finite gradient shapes and small numerical differences are recorded in `output/optimization/construction-expansion-training-cpu/README.md`.

The initial stronger fit's first completed epoch checkpoint is copied and hashed before stopping. `scripts/resume_sdnet_expansion_performance.py` verifies that checkpoint and restores model/optimizer values before any resumed optimizer update. Explicit trainer hooks apply actual channels-last model/input storage; the installed Ultralytics CPU option alone does not do so. The resumed fit runs logical epochs2–8, preserving the total eight-epoch budget, 512px, batch4, freeze4, data and CAL selection policy. Its outputs and actual epoch timings live in a separate `construction-expansion-sdnet-v2-performance` directory.

A seed does not restore Python/NumPy/Torch RNG or DataLoader sampler state. Standard resume here is a new numerical/augmentation trajectory with the same declared logical budget, not an identical uninterrupted run. The initial discarded unsaved epoch2 work is recorded separately. Deployment remains one-thread CPU-ONNX; four-thread training speed is never reported as model FPS.

## Selection, localization and interpretation

Fresh COCO/VOC initializers receive new task heads; new class names are not attached to road/crack predictions. Sources retain their actual semantics: publisher Mold is separate from damp-only appearance, Missing combines bolt/nut appearance, and no SDNET corrosion labels were acquired.

Thresholds use calibration data only. Native after-training results retain fixed0.5 comparisons as well as the frozen CAL-selected point. The exposed SDNET 98-photo set is reused development for the stronger experiment; it cannot become an independent validation set again.

Deployment audits use the actual native-resolution adapter and fixed thresholds. Boxes require IoU≥0.5; mask tasks report actual pixel overlap/connected regions as well. Integer display geometry can move a borderline overlap across0.5; the stain audit preserves both native floating-point counts and served integer counts, with the exact one-photo explanation. Opposite-class SDNET annotation completeness is unverified, so annotation-unmatched predictions and publisher-Fixed negative flags remain separate.

None of these fits passes the full release protocol. [Category-specific gaps](construction-expansion-category-catalog.md) remain explicit, including absent installed-tile measurement data and unavailable blister annotations. Do not replace missing evidence with manufacturing images, synthetic defects or an increased number of class labels.
