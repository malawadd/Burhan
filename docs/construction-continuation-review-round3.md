# Independent construction continuation review

20 October 2026. Review scope: static training/data code and existing **train/calibration** raw/GT/model overlays. No heldout data, new NN run or model changes were used in this review. The first candidate and its failures must remain preserved. This note proposes a separately versioned continuation; it is not evidence that the continuation has succeeded.

## First candidate: actual learned behavior, poor generalization

The training agent reported five completed epochs and calibration mAP50 approximately0.0342. The existing fixed-cutoff diagnostics report:

| Split / class | Class cutoff | TP | FN | FP |
|---|---:|---:|---:|---:|
| Train crack | .65 | 179 | 211 | 7 |
| Calibration crack | .65 | 2 | 123 | 0 |
| Train spalling | .20 | 79 | 28 | 64 |
| Calibration spalling | .20 | 1 | 37 | 12 |

At crack cutoff.10, calibration still has10TP/115FN/31FP. Lowering a cutoff does not establish useful localization. These are publisher-box diagnostics at matching IoU0.50; they are not pixel-mask accuracy or independent field reliability. Correlated selected frames also do not equal independent physical defects.

I independently inspected the following actual triptychs. The publisher boxes align with real visible damage, and the model panels contain actual predictions rather than copied annotations:

- [Calibration spall75](<C:/Users/IAM/Documents/hackathon/opencv/project1/output/evaluation/construction-round3/evcivil-yolov8n-r3/fit-diagnostics/calibration-spalling-first.jpg>): a clear small chipped/surface-loss patch is missed.
- [Calibration crack129](<C:/Users/IAM/Documents/hackathon/opencv/project1/output/evaluation/construction-round3/evcivil-yolov8n-r3/fit-diagnostics/calibration-crack-first.jpg>): a visible horizontal crack is missed.
- [Training mixed crack122](<C:/Users/IAM/Documents/hackathon/opencv/project1/output/evaluation/construction-round3/evcivil-yolov8n-r3/fit-diagnostics/train-spalling-first.jpg>): the broad rough/spalled band is localized as spalling; the narrower crack is missed.

No swapped class IDs, bbox units, panel semantics or obvious geometry error was found in these examples. The large training/calibration gap supports a representation, object-scale or domain-generalization problem. It does not prove a single causal explanation. The first training example's spall is a large near-full-width band; the calibration example's spall is a small isolated patch. Indoor/day/night/texture and size diversity matter.

## Static training review

`scripts/train_construction_damage_round3.py` preserves the source model's class1 crack mapping, transfers matching-shaped learned tensors and explicitly resets the newly supervised class0 spalling classifier. It does not relabel the old joint score as spalling. The custom trainer returns the initialized model instead of silently reseeding it from YAML. Locally generated continuation checkpoints retain learned class heads without another class reset. Parameter-change provenance verifies that the spall classifier actually learned.

The first setup freezes ten early blocks and uses physicalbatch4/nominalbatch16. Few completed epochs therefore provide relatively few optimizer updates, and much of the backbone cannot adapt its weights to gray camera/night imagery. These are plausible constraints, not independently measured causes. Existing augmentation does not paste or simulate damage; source masks/boxes come from real publisher data. Static extraction and YOLO normalization match the inspected original camera coordinates.

## Recommended separate train/calibration-only continuation

The training agent should freeze an exact protocol before fitting, including starting checkpoint SHA, data manifests, actual added timestamp-qualified exposures, optimizer, epochs, patience and selection rule. A concrete first continuation is: retain the learned heads, freeze four early blocks, AdamW learning rate.0005, physical and nominalbatch4, warmup0 and416 input. Choose one exact maximum within the proposed8–12epochs and a recorded calibration-only early-stop/checkpoint-selection rule. Keeping416 first isolates data/adaptation changes from a separate resolution experiment. It does not guarantee improvement or15FPS.

Add only the separately acquired real-data training contingencies:80 official-TRAIN DACL photos with complete polygons and five additional safe Dec27 raw sequences. The DACL data supply125Spalling and66Crack/ACrack source-polygon boxes, with real gray4:3 crops. Every visible source target is retained; boxes derived from semantic polygons are not native instance labels. Before fitting, audit the box-size distribution at416 and any augmentation filter that could discard very thin targets. Preserve original geometry and record any discarded training targets instead of concealing them.

Extra night sequences require the same publisher timestamp-association convention. Canonical physical defect IDs must be excluded against all exposed, calibration and heldout IDs across day/night naming, not merely different filenames. The new train YAML may add fitting data, while calibration and heldout images/labels stay byte-identical. Physical bridge IDs are unavailable for DACL; source-image/hash separation cannot establish bridge independence. Mixed training inherits DACL noncommercial data restrictions.

Evaluate train and unchanged calibration again before opening heldout. Keep raw/GT/pred panels and class-specific TP/FP/FN, thin-box subgroup outcomes, negative flags and runtime geometry explicit. A higher input geometry is a separate candidate requiring its own predeclared calibration comparison and actual laptop speed measurement. Repeated calibration-driven development is allowed, but the result must not be presented as independent evaluation; the final frozen candidate gets the untouched heldout assessment.

Acquisition/provenance: [real-positive and night contingencies](<C:/Users/IAM/Documents/hackathon/opencv/project1/docs/construction-positive-contingency-round3.md>). At this note's creation, these assets are preparation artifacts and do not replace the failed first candidate's reported results.

## Frozen continuation subsequently started

The training owner subsequently froze [the separate continuation protocol](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/evcivil-round3-continuation/protocol.json>), SHA256`99afdeb6e315ecf133c8c1cbb53d4ceadd99131fee71b49dfea69daa0470a39d`, and started fitting. The exact configuration is416input, freeze4, physical/nominalbatch4, AdamWlr.0005, warmup0, three CPU threads,12maximum epochs, patience8 and no wall-time cap. Learned class heads are retained. Training duration under other fixed-work CPU activity is not an inference speed measurement.

Actual fitting data are596 positive training samples:476 genuine camera exposures plus120 photo training views from80 unique source photographs;20 verified class-negative background samples are additional. The extra Dec27 timestamp-qualified training exposures are2/3/4/0/1 for IDs6/7/8/13/15, respectively—only10 extra actual camera exposures. Sequence13 contributed zero qualified exposures; no event-only labels were fabricated into camera frames or treated as healthy observations. The40 additional photo closeups are overlapping views of existing training sources, not40 new independent scenes.

I statically reviewed the versioned builder and YAML. Calibration and heldout refer directly to the original directories; original evaluation records are retained, canonical-ID exclusions are enforced, and source/image/label hashes are verified. Class0 remains spalling and class1 crack. The owner reports unchanged112calibration exposures+10backgrounds and80sampled heldout exposures+10backgrounds. No heldout predictions were opened for this review.

One provenance limitation was reported to the owner: the frozen plan identifies the prior candidate's exported ONNX SHA, while the actual starting Torch`.pt` checksum is recorded by the trainer rather than compared with an explicit predeclared`.pt` checksum in that plan. Preserve the actual checkpoint path/hash as observational training provenance; do not rewrite an already-started protocol as though it had contained a stronger assertion.

## Bounded tiny-segmentation fallback search

No replacement was installed, downloaded or run. [UltraFastCrackSeg's official repository](https://github.com/weiqingq/UltraFastCrackSeg) has Apache2 code and configurable checkpoint paths, but this search did not verify a supplied trained checkpoint. Its advertised CPU/GPU FPS is not this laptop's measurement. [DeepCrack's author implementation](https://github.com/yhlleo/DeepSegmentor) publishes trained-weight links, but its architecture is not established as a tiny real-time replacement here; separate dataset noncommercial terms and weight-file safety remain relevant. [CrackRefineNet's official repository](https://github.com/AmmarOkran/CrackRefineNet) publishes trained CFD/Crack500 checkpoint links and Apache2 project terms, but uses a ConvNeXt/MMSegmentation configuration rather than an established small local model. Linked files' downloadability, tensor-only format and laptop performance were not verified. FPCNet's paper alone did not establish a ready-to-use trained asset in this bounded check.

These remain research leads. There is no verified small-checkpoint improvement to substitute into the active continuation or cite as measured thin-crack support.
