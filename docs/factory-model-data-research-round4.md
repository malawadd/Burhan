# Round4 real-data and supervised-model study

Goal: improve actual defect localization beyond the current10/11 exposed-comparison result, reduce normal alarms and retain at least15 complete-profile calls/second on the i7-8550U/UHD620 laptop. The original comparison and CAL have been repeatedly exposed during development; they are not new blind evidence. Official SDD2 TEST stays sealed.

## Immediate decision

The strongest bounded next experiment is supervised transfer on **complete, auditable real-data coverage**, followed by target-domain fine-tuning, on the already measured C200 phase/context/max-pool serving graph. This changes learned weights and source coverage materially without assuming a new backbone will run fast. Prior normal-only teacher–student trials passed or failed numerical checks but did not produce an eligible localization policy. More normal-only score formulas are not the immediate recommendation.

The candidate plan is4,096 transfer updates plus900 SDD1-focused updates, batch1/CPU1, with300/600/900 fine-tune checkpoints and all-three CAL/global freezing. It is a proposal, not an authorized fit. Before optimizer execution its ledger must prove complete source/foreground coverage and fit within the fixed budget. If it needs more than4,096 entries, report the count for root review; never silently drop sources or positive pixels. The serving graph and final measured speed remain separately pinned and must be retested with actual fitted weights.

## Dataset findings from primary sources

| Source | Genuine observations / actual labels | Rights and access | Round4 use |
|---|---|---|---|
| Original KolektorSDD | Existing399 views of electrical commutator surfaces; original pixel masks and item folders | Existing author acquisition, CC BY-NC-SA4 research terms | Keep239 TRAIN, unchanged80 CAL and exposed80 comparison. No supposedly unused original images exist outside these protected partitions. |
| KolektorSDD2 | Author-controlled industrial photographs with several visible defect types and dense masks; officialTRAIN246positive/2085normal | Direct original ZIP already local; CC BY-NC-SA4, commercial permission separate | Existing innerFIT1865 contains197positive/1668normal. Explicitly cover **all197 labeled positives and all1865 FIT sources**, rather than sparse20% rehearsal. Reserved466 and officialTEST1004 remain out. |
| Magnetic Tile Defect, author repository | 1,344 real cropped surface photographs and paired native PNG pixel GT:115blowhole,85break,57crack,32fray,952free,103uneven | Author welcomes research use with citation; no formal commercial license file found | Root assigned **all1344 to TRAIN before pixels**, no internal MTD benchmark. Acquire exact author commit and validate original pixel labels. |
| NEU-DET | Author describes1,800 grayscale200×200 steel images in six defect categories; detection locations are boxes | Original university page links Drive/Baidu and separate segmentation resources; general commercial terms not established | Box GT must not be filled in and called actual defect masks. Not immediate dense-supervision data. |
| SD-saliency-900 / NEU-Seg | Original NEU page points to these separate segmentation resources | Author repositories require their own rights/schema verification; no new acquisition made | Possible later steel-domain pretraining, but do not mix their identity with NEU-DET boxes or assume mirror licenses apply. |
| DAGM2007 | Authors explicitly say textures/defects are artificially generated and weak labels are surrounding ellipses | Public original benchmark | Excluded from requested real-defect training/evidence. A loader's phrase “real data” can mean files instead of random tensors; it does not change the benchmark's synthetic origin. |

SDD2 counts, acquisition setting and terms are from [ViCoS authors](https://www.vicos.si/resources/kolektorsdd2/). Existing canonical filtering removes two identical-copy files; frozen inner split SHA `ecee83db4becdac3133a2d4c0950ef383c511d143b5509aa8c1fdf61d445a47c` remains unchanged. No physical-item IDs are supplied for SDD2; image separation is not physical independence.

MTD pairing/research permission come from the [original author repository](https://github.com/abin24/Magnetic-tile-defect-datasets.). The [authors' paper](https://doi.org/10.1007/s00371-018-1588-5) and [author-uploaded full text](https://www.researchgate.net/publication/327701995_Saliency_defect_detection_of_magnetic_tiles) describe photographing the same tile under multiple illumination conditions. The filenames contain experiment/exposure prefixes and unique numeric strings, not verified physical-tile IDs. Root's all-TRAIN assignment avoids inventing a supposedly independent MTD split. These are real photographs; the paper's description of simulating assembly-line illumination does not mean generated damage. No stills will be turned into fake production footage.

NEU image/category/box facts and distinct segmentation links are from the [original university page](https://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm). DAGM's synthetic generation and weak ellipse annotations are explicit in the [original competition description](https://conferences.mpi-inf.mpg.de/dagm/2007/prizes.html) and [author dataset record](https://zenodo.org/records/8086136). The [SD-saliency author repository](https://github.com/SongGuorong/MCITF) is a lead, not a verified commercially licensed download in this round.

## Model evidence and practical choice

The [official mixed SegDecNet implementation](https://github.com/vicoslab/mixed-segdec-net-comind2021) is directly tied to this dataset family. It specifies dense supervision, balanced segmentation/classification training, gradient control and negative frequency sampling. Its paper metrics are not our exact defect-mask recall or this laptop's speed. No clean author checkpoint with verified exclusion of our original items was established; downloading an arbitrary “KSDD pretrained” model could leak comparison labels. Its CC BY-NC-SA code/data terms also remain separate from deployment permission.

The new supervised experiment therefore reuses the fast known graph and learns from more genuine labeled material. Full native foreground coverage is more useful than just increasing the number of random updates. Source-balanced normal exposure must remain present so auxiliary datasets do not overwhelm the239 target views or erase false-alarm control. The target-focused900-update phase uses the original native full-width640×512 crops plus declared20% auxiliary rehearsal; new-domain images are isotropically resized to width512, tiled vertically and padded only with an explicit ignored-loss region. No empty positive crop may be relabeled normal; all actual in-view annotations remain supervision.

As a training-only alternative, [clDice's primary CVPR paper](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html) and [official MIT-licensed code](https://github.com/jocpae/clDice) support topology-aware losses for thin structures. That is a plausible hypothesis for connected fine defects, not established industrial performance. Root/neural owner chose **not to add it** to this first round4 experiment, preserving the existing supervised BCE/Tversky objective while isolating data coverage/native geometry. It remains research, not a claimed accuracy improvement.

“Lightweight” paper speed is not sufficient. For example, the [MINet author paper](https://arxiv.org/abs/2405.16096) reports0.28M parameters but only6.3CPU FPS at368×368 in its own setup, despite a much higher GPU figure. Neither that CPU figure nor modern-GPU claims qualify a new1280×512 model on this laptop. Retaining the measured C200 topology is the more concrete speed-constrained starting point; final new-weight capacity/runtime are still mandatory.

## Required transfer-ledger review

The base source count is239+1865+1344=**3448**. This is not necessarily the final training-step count: positive full-foreground tiles can add entries. The protected ledger must contain source hashes, native/derived geometry, exact isotropic scale, valid observed-pixel mask, tile coordinates, original positive-pixel coverage and source-balanced sampling weights. Padding contributes no foreground or background loss. Every projected genuine foreground pixel must be covered by at least one declared training tile; negative entries require actually empty in-view GT. Source identity, rather than filename-based label guesses, determines positivity.

All masks use their original native ground truth; no box filling, GT dilation, synthetic defects, predicted labels or comparison-derived ROI. The same matched whole-patch training transforms may be used only if declared before fitting. Preserve strict warm-start C200/source hashes and frozen BN behavior, source/geometry fixtures before optimizer, all three mandatory checkpoints, fixed CAL grid/precision/normal-alarm guards and root-only globally selected original evaluation. A different/easier dataset result cannot replace the existing factory headline.

MTD acquisition/provenance/QA paths and exact completed counts are recorded additively below after original download and validation. At this stage there is no new NN, trained weight, accuracy or FPS result.

## Completed author acquisition and label curation

Original author commit `cc4eb530b57b9cb7225bf76023873043ffd50655` was acquired without inference. The 53,186,096-byte original ZIP SHA is `592b050953d1de4ffa538b93b6b91aee7268d8e59b73d4d9c92cfce187d7ce66`. All 2,688 original image/mask files matched the author's Git blob identities and archive CRC/length checks. The immutable acquisition manifest SHA is `5b24bfebd36396f9192bbefc12c59e50f69d5f759436167ed002246a2fcc2fa3`.

The first conservative binary-only QA **failed and is preserved**, rather than being silently overwritten. All 388 nonempty defect masks contain intermediate grayscale boundary values; their interiors/background are generally 255/0. Four `MT_Uneven` category images have completely empty paired masks. They are quarantined and never treated as healthy negatives: `exp3_num_45042`, `exp4_num_124690`, `exp4_num_352527`, and `exp4_num_45057`.

Root approved an additive, TRAIN-only convention: **native mask >0 defines annotation support**. This is our deterministic binary conversion, not a claim that the author specified that threshold. It includes gray boundary pixels, preserves every original PNG, and performs no dilation, box filling or invented defect labeling. Native codes and full per-image histograms remain recorded. The author saliency toolbox's example program does not establish a separate GT binarization convention. The inherited `mask_foreground_definition` text in initial QA rows referred to the failed binary-code expectation; the new manifest's explicit top-level `mask_convention` and per-row `binary_conversion` supersede that description.

The additive [curated TRAIN manifest](../datasets/factory/round4/magnetic-tile-author/curated-TRAIN-manifest-v1.json), SHA `53cb4a0eac6ed52e26eb713933fcf9739c2111b576c666c6ad806a34a3280158`, accepts **1,340 original pairs: 388 positive and 952 normal**, with four separately retained quarantine records. Its curation protocol SHA is `4ef68ae4a5f8646bbc5cce16c15bc89a85e3e40c79d03348383e0195c412e3a4`. Thus the actual prospective source count is **239+1,865+1,340=3,444**, superseding the earlier pre-QA 3,448 planning count. The complete mandatory tile count is still an executable ledger gate, not an assumed fit within 4,096.

All three fixed source/GT sheets were reviewed. They show real annotated blowholes, boundary breaks, cracks, fray and broad unevenness regions, with matching native image bounds. Several views visibly repeat a tile under illumination changes, confirming why no physical-independent MTD test split is claimed. These sheets are RAW / publisher annotation / annotation overlay only; **there is no model prediction** and no assertion that the new model works yet:

- [Fixed GT sheet 0](../datasets/factory/round4/magnetic-tile-author/fixed-publisher-GT-sheet-0.jpg)
- [Fixed GT sheet 1](../datasets/factory/round4/magnetic-tile-author/fixed-publisher-GT-sheet-1.jpg)
- [Fixed GT sheet 2](../datasets/factory/round4/magnetic-tile-author/fixed-publisher-GT-sheet-2.jpg)

MTD supplies extra industrial texture/defect supervision, not proof that ferrite tile defects equal commutator defects or construction damage. Its source rights remain research-use/citation only; no formal commercial grant was established.

## Completed cross-source screening

The separately pinned screen protocol `c522838fb0a1afca19ab8c6f45ab85a6fc4658483ae5ea3c614d5d65b1ab36cc` fixed the source list and pHash threshold before additional TRAIN decodes. The [screen report](../datasets/factory/round4/magnetic-tile-author/cross-source-screen-v1.json), SHA `39c258323c0ad2a60852c171d7ba9a9ce1c58aa9ebd1ac10cbe9a5a35e1ecb20`, found:

- **0 exact file-SHA matches** between accepted MTD and 239 SDD1 TRAIN +1,865 SDD2 FIT +80 CAL metadata +80 exposed-comparison metadata records.
- **0 exact native-pixel matches** and **0 pHash Hamming ≤6 flags** between MTD and the 2,104 TRAIN images, using a fixed 63-AC-coefficient DCT hash.
- No CAL/comparison image or mask was decoded. SDD2 reserved466 and official TEST were not accessed.

This establishes no known cross-source duplicate from the checks performed. It does not establish physical/site independence, nor does it establish absence of perceptual near-duplicates against protected data whose pixels were not reopened. No universal duplicate-proof claim or new accuracy result follows. The screen made no automatic acceptance/exclusion changes and ran no NN.

## Actual complete training-coverage plan

The finalized data-only proposal SHA is `89adfa6e24de1c4dd3b2fbe13c2644b3948dc521cec520e840ceadf9ab7d1a2e`. It contains 3,444 unique sources (616 positive/2,828 normal) and 3,678 mandatory tiles. All 14,792,296 **projected** annotation pixels are covered exactly once by mandatory entries. Another418 positive source-balanced entries fill the fixed4,096 transfer budget. The900 fine entries contain360 target-positive,180 hard-location,180 unbiased and180 auxiliary entries (90 SDD2/90 MTD). This includes every197 SDD2 FIT positive and every accepted source; it is no longer a sparse rehearsal plan.

An independent metadata audit verified those sums, source uniqueness, per-source foreground coverage, nonempty positive crops, every-fifth auxiliary cadence and proposal artifact hashes. It did not open additional pixels or run a model. See [the detailed independent review](factory-round4-supervised-independent-review.md). A first preparation failed on two native85-valued annotation pixels in old SDD1 TRAIN; the source guard was corrected while preserving the already established `mask>0` interpretation. No GT threshold was changed. Root approval and actual fitting/evaluation remain separate from this completed data plan.

## Prospectively declared calibration tradeoff

Before the new fit, root approved two separately frozen development arms: **primary normal FP ≤1/70 remains unchanged**; secondary FP ≤3/70 uses the same threshold/area/growth grids, pixel precision ≥0.65 and weakest-positive rank. All three checkpoint outcomes and a global winner per arm must be frozen before either original-development comparison is scored. Both arms, including failures, must be reported. No threshold/model choice may be revised from those comparison outcomes. Each eligible winner still needs actual fitted ≥15 complete-profile calls/second and the fixed 16FPS runtime validation. With only 70 CAL normal views, neither empirical alarm budget establishes a production false-alarm rate.
