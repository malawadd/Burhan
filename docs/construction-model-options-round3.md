# Additional crack-model research, round three

Reviewed 20 October 2026 while the measured local candidates trained. These are implementation leads, **not successful local benchmarks or installed capabilities**.

| Primary source | Verified availability | Decision |
|---|---|---|
| [EfficientCrackNet, WACV 2025 paper](https://openaccess.thecvf.com/content/WACV2025/html/Zim_EfficientCrackNet_A_Lightweight_Model_for_Crack_Segmentation_WACV_2025_paper.html) | Authors report a 0.26M-parameter CNN/transformer crack segmenter evaluated on Crack500, DeepCrack and GAPs384. An immediately usable official pretrained asset was not established in this review. | Architecture lead; not treated as a working detector merely from paper accuracy. |
| [UltraFastCrackSeg, ICRA 2025 paper](https://doi.org/10.1109/ICRA55743.2025.11128493), [official repository](https://github.com/weiqingq/UltraFastCrackSeg) | The repository has training/test/model code and Apache-2.0 code license. Its stated 80+ low-power-CPU FPS is an author measurement, not this laptop's result. No released trained checkpoint was established in the reviewed README; its older environment includes CUDA/Triton dependencies. | Not substituted for an actual laptop export/accuracy test. The README suggests using DeepCrack test as validation; this project's selection rules must keep calibration and comparison separate. |
| [LiteCrackSeg, official repository](https://github.com/1Kaleb/LiteCrackSeg) | Author README links a [TUT.pth checkpoint](https://drive.google.com/file/d/12BDNmnJYWvVhkGNI0m2zGQdJHBoGTOHs/view). The viewed repository has architecture, train/test/eval code. Download access, exact artifact hash, code/weights redistribution terms and laptop inference were not established here. | Contingency for pixel-level thin-crack work; not integrated or credited as an accuracy improvement. |

The active experiments instead have actual locally available assets and publisher labels: a two-class field-video detector trained on separate ev-CIVIL groups, and the official-author DACL multiclass semantic model. Their hashes, actual measured results and failures belong in [construction accuracy](construction-accuracy-round3.md) and [category expansion](construction-category-expansion-round3.md).

No account login, paid inference, user footage upload or remote execution was used for this review.

## Subsequent actual LiteCrackSeg experiment

The initial availability review above is historical. The official TUT checkpoint was subsequently downloaded and safely loaded with742 strict tensor matches,2,939,010 parameters. Checkpoint SHA256:`c12fb1ce991e479a32e5a26d7648ce9d20a440b9af4fd0411d321aa6a2eed497`;384px ONNX SHA256:`20936464f5fb6ef32cf5d84a87c84616bf1d5461364fbfb466dd7c4846d04055`. Exact author RGB float0..255 preprocessing was retained; unused ImageNet constants were not applied. Author Torch/ONNX maximum probability difference was7.78e-6.

All122 predeclared real calibration images were evaluated before any held-out prediction. The best-F1 threshold.15 produced6TP/119FN/111FP at component-box IoU.50, thin-box-proxy recall4/62 and flags on10/10 crack/spall-negative photos. No threshold passed the precision/false-alarm gates. This model was not registered or made a default, and no speed improvement is claimed. One fixed example visibly traces a genuine hairline but its tight predicted mask box does not match the larger publisher box at IoU.50; there is no pixel ground truth in this video dataset to turn that visual observation into a pixel accuracy score. Saved raw/GT-box/model panels, maps and full results are in `output/evaluation/construction-round3/litecrackseg-calibration/`. Whole-checkpoint licensing remains unresolved. A separately preregistered exposure-correction candidate is a new experiment, not a rewrite of this failed result.

## Further source leads, not installed detectors

The [RBoabeng infrastructure-defect repository](https://github.com/RBoabeng/infrastructure-defect-detection-yolo) states that a trained YOLO checkpoint is available in its assets directory. Its author also explicitly reports an exclusively-spalling validation set and resulting spalling bias. That warning makes independent balanced calibration essential; the stated CPU timing is not this laptop's measurement. Asset bytes, class map and license need verification before use.

The [Hhhpraise crack benchmark](https://github.com/Hhhpraise/crack-detection-benchmark) advertises YOLO12s detection/segmentation checkpoints with CBAM and coordinate attention. These are source leads only: custom architecture code, actual assets, provenance and license must be inspected, and the same complete calibration set must test any export. Paper/README accuracy and GPU timings cannot replace local measurements. Repositories with only COCO base weights or instructions to train a missing `best.pt` are not pretrained concrete-damage detectors.
