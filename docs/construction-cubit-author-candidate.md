# Author crack/spall instance-segmentation candidate

This separately evaluated public checkpoint **failed the requested accuracy goals**. Technical export/backend guards and all122 actual serving-parity checks passed, but crack localization remained weak and the model produced no spalling detections at its fixed threshold. It adds no qualified capability. The prior local wall, thin-crack and spalling failures remain documented unchanged.

The [author CUBIT project](https://benyunzhao.github.io/CUBIT-Dataset/) links shared infrastructure segmentation benchmark materials. Their [YOLOv8 nano folder](https://drive.google.com/drive/folders/1CSNmnecW-CrG_KnUP6PhUhV7x0-PyBzM) contains the original `weights/best.pt` and training arguments. The checkpoint is 6,827,683 bytes, SHA256 `e463069e06daf66fd99c21f5d9a3d88fe7a48c71d43263ed4d338a56e6a86d30`. Original arguments describe YOLOv8n segmentation, 640 input and 300 epochs. Exact training sources and their overlap with our development recordings are unknown. The standalone checkpoint license is not declared; the CUBIT photograph dataset's CC BY4.0 declaration does not independently clear model redistribution or separate demonstration-video rights. This is research evaluation only.

## Safe source restoration

The original ZIP CRC and nonexecuting pickle opcode/global inventories are retained under `models/construction/cubit-author-release-v1`. Reviewed extraction used `torch.load(weights_only=True)`, an explicit official Torch/Ultralytics/numeric allowlist and restricted replacements for two legacy globals. The dill replacement accepts only the string `set`; the getattr replacement accepts only the official `Detect` class and `forward`. There is no general dill loader or arbitrary attribute lookup.

The extraction produced 417 finite state tensors, 3,275,176 total elements. Plain state SHA256 is `afe2ae068dd6b502ac69baa940c5da8eb7d907a037a907739451589644af2a8c`. The actual author order is **class0 crack, class1 spalling**, opposite the prior locally trained two-class camera detector. The head has two classes and 32 mask coefficients. No network constructor, dummy stride forward, export or inspection inference ran during restoration.

A subsequent independent static read found no generic dill/getattr execution path. It recommended future hardening: require exact unsafe-global-set equality, reject pre-existing ambient safe-global allowances and pin installed class-module versions/hashes before restoration. The executed source used an exact checkpoint byte hash and a subset check, and did not record those additional preconditions. Its source and original receipts are preserved; these recommendations have not been retroactively represented as executed checks or used to rerun extraction.

Source YAML is data only. The proposed exporter first requires an exact structural match with the installed trusted YOLOv8-seg schema: identical scales, backbone and head, with reviewed scalar metadata `nc=2`, `scale=n`, `ch=3`. Only a copy of the installed schema enters `SegmentationModel`; author YAML is never sent to `parse_model`. Model construction does execute a dummy stride forward and therefore requires a separately approved NN window. All 417 source state keys/shapes must match and every tensor must transfer exactly before export. No head reset, class remapping, fitting or selected-checkpoint search is allowed.

## Frozen proposed operating policy

The original unexecuted proposal is `output/evaluation/construction-cubit-author-v1/candidate-proposal-before-network.json`, SHA256 `464061df5921532e4d7f8b4958de60f49bcf2e38fc453447171ec209c21641aa`. Its unexecuted revision2, SHA256 `fe54790f1b53684817c555191f766595e45c8e7aafd44c7e25ff200a345aeb11`, additionally binds the original bestepoch1 complete cached122 comparison. Final revision3 adds independent numeric export gates, persistent partial mask artifacts and explicit empty-mask accounting. It passed independent static review and is `output/evaluation/construction-cubit-author-v1/candidate-proposal-revision3-before-network.json`, SHA256 `9cdbf9212bfb8fbb1cbfd436129f739a899f84a7cf61b7060850662dc2467bda`. After explicit approval, the executed protocol was frozen at SHA256 `801c2da95ae112991f0754c60a9cab735e8f622128036d057f1e1567511fae24`. All prior proposals remain preserved.

| Item | Fixed choice |
|---|---|
| Network | Trusted installed YOLOv8n-seg, strict author state |
| Source class order | crack0, spalling1 |
| Native input | Original full-frame BGR; shared bounded global low-light gain once |
| Network input | 640 letterbox, RGB float32 /255 |
| Detection cut | .25 for both classes; no grid search or threshold rescue |
| NMS | Class-aware IoU .45, existing serving decoder |
| Instance mask | Actual 32 prototype/coefficient reconstruction; logits >0, native resize and detector-box crop |
| Backends | Local OpenVINO f32 or ONNX Runtime CPU, one stream |
| Accuracy matching | One-to-one publisher-box IoU ≥.50; no pixel accuracy from box annotations |

The adapter reports requested settings separately from actually queried execution devices, precision, streams and thread settings. Model paths must resolve inside the local construction-model directory and model/wrapper hashes must match. Operator class selection is explicit; source material is not automatically identified.

## Complete fixed evaluation

The first stage uses the same **122 development CAL inputs**, including 125 crack boxes, 38 spalling boxes and 10 independently annotated two-class-negative images. It performs no fitting, grid search or protected-heldout loading. All wrong-class/unmatched candidates remain counted. Masks are saved with native shape, bit-packed references and hashes, clearly separate from publisher boxes.

For every CAL input, cached direct-decoder predictions must match an actual `ConstructionProfile.process` call in labels, boxes, scores and native mask hashes. The profile's observations must also retain those candidates. Initialization warmups are discarded and do not become source observations. This tests serving integration, not just direct-model inference.

Before any CAL source forward, the proposed pipeline separately compares one fixed nonconstant Torch export tensor against ONNX Runtime CPU and the actual compiled OpenVINO GPU. Coordinates have a one-NN-input-pixel guard, class scores .01, mask coefficients/prototype logits .1, with finite outputs and queried GPU/float32/one-stream configuration. All first outputs and receipts are retained; failure prohibits CAL and rerunning or relaxing the guard. This is numerical initialization, never defect footage or accuracy evidence. Cached BEFORE source/image/index/GT identities must match the fixed122 sources before candidate NN as well.

Minimum prototype review guards remain precision≥80% and box recall≥50% for **each** class, geometric-thin crack recall≥50%, combined negative-image alarm rate≤10%, and full serving parity. The geometric-thin subset is defined from native publisher-box dimensions, not physical crack width. Passing these modest minima would not establish a solved or production-safe system. Unknown checkpoint training overlap prevents independent accuracy credit.

The fixed genuine moving-camera development sources are unchanged: exposed crack89, CAL crack129, CAL surface-loss75 and CAL night ros_crack21. Surface-loss76 is a separate **post-hoc known-hit illustration**, selected previously from old-model outcomes and carrying no independent credit. Their source hashes are pinned before any new candidate NN. A separate reviewed exporter must preserve every original exposure/PTS, RAW versus actual gained input, external GT boxes, actual model masks/boxes, all misses and false alerts. No source replacement, still-image video fabrication or annotation-derived model pixels are allowed.

## Implementation and current state

- `scripts/safe_extract_cubit_author_checkpoint.py`: reviewed extraction completed; no NN.
- `scripts/export_cubit_author_candidate.py`: first approved attempt failed during tracing; original source and failure preserved.
- `edge_runtime/profiles/construction_cubit.py`: unregistered candidate adapter; dispatcher branch is available, but no model variant or app default is installed for this checkpoint.
- `scripts/evaluate_cubit_author_candidate.py`: prepared proposal/freeze/fixed122 evaluation; no CAL forwards yet.

The original exporter subsequently ran under the approved protocol. Installed trusted architecture comparison, strict 417-key source transfer/equality and the first Torch output shape/finite guards passed. ONNX tracing then failed because the initial `torch.inference_mode()` forward cached inference tensors in the detection head; a subsequent traced autograd path could not save them for backward. The failure receipt and console are preserved under `output/evaluation/construction-cubit-author-v1`. No ONNX/ORT/GPU numeric result, CAL forward or accuracy result was produced, and no numeric guard was relaxed or retried. That first Torch output was checked but not serialized before the failed trace; this limitation is preserved rather than claimed as saved evidence.

Separate v2 recovery sources passed focused review and were approved under proposal SHA256 `54579568eb71297241215d32afc1c95526ac0a07fad4171e5b771da66bdb3329`. They use `torch.no_grad()` for the same initial fixture so cached anchors are ordinary tensors, and save that first fixture and progress receipt before tracing. New output/model names bind the failed original lineage. Architecture, source weights, author class order, raw data, fixture values, thresholds, numerical guards and evaluation method remain unchanged. The executed original sources/receipt are not edited. Before execution the approved v2 protocol was frozen at SHA256 `0131a3191a95d69829214b11690adcd77563f096f81935228b45217392b578a7`.

## Actual completed v2 calibration result

The recovered model is `models/construction/cubit-author-release-v1/author-yolov8n-seg-640-v2.onnx`, SHA256 `df13a679081023d9093f731631ad199dbaaa9c467bb1e3c0df7054240c51bb6f`. Source tensor transfer remained exact. The first Torch→ORTCPU fixture passed: maximum coordinate error .004272 NN-input pixels, class-score error 1.33e-7, mask-coefficient error 1.67e-5 and prototype error 2.50e-5. The first actual GPU fixture also passed: .005188 pixels /2.23e-9 /3.02e-5 /2.74e-5 respectively. All outputs were finite. Queried actual execution was GPU.0, float32 and one stream. These are numerical checks, not defect accuracy or FPS measurements.

All122 fixed-policy inputs then completed, with actual `ConstructionProfile.process` labels, boxes, scores, native masks, observations and native-gain provenance matching the direct decoder. No operating threshold was selected from results. The original bestepoch1 cached122 predictions were reused only after exact source/frame/image/GT identities matched; their counts were recomputed.

| Same122 development inputs | Original bestepoch1 frozen point | Author candidate fixed .25 |
|---|---:|---:|
| Crack TP / missed / unmatched | 7 /118 /73 | 12 /113 /81 |
| Crack precision / recall | 8.75% /5.60% | 12.90% /9.60% |
| Spalling TP / missed / unmatched | 5 /33 /1 | 0 /38 /0 |
| Spalling precision / recall | 83.33% /13.16% | Undefined /0% |
| Geometric-thin crack matches /62 | 0 /62 | 7 /62 (11.29%) |
| True-negative images flagged /10 | 3 /10 | 5 /10 |
| All-class TP / missed / unmatched | 12 /151 /74 | 12 /151 /81 |

The frozen BEFORE thresholds are crack.10/spalling.25; the author candidate uses .25 for both, as predeclared. Both are reused development evaluations and the author's training overlap is unknown. The candidate trades some crack matches for lost spalling coverage and more overall false alerts. It is not an overall improvement or a thin-crack/spalling fix. Every accuracy minimum guard failed. Spalling abstention is not successful negative detection.

All93 actual native masks are retained, including per-frame first-pass mask files and hashes. None is empty, and none of the12 bbox matches has an empty mask. This only proves predicted foreground exists; it does not establish correct pixel tracing because the camera dataset supplies boxes, not pixel masks. Report SHA256 is `c3baf88d973c11cc70b083364fae48d5cf35ea4c3a34f0f8223c5949a2177f69`, at `output/evaluation/construction-cubit-author-v2/fixed-policy-calibration-report.json`. Full first raw Torch/ORTCPU/GPU results and all runtime records remain preserved. No heldout inference, threshold rescue, extra fit or automatic model promotion followed.

## All five complete genuine moving-camera recordings

The exact previously declared five sources completed under `scripts/export_cubit_author_development_videos.py`; frozen video-protocol SHA256 is `6a60564e0c8944ed050d5eb1c9374c7e60fe5f6b37f7d7e673946a731871c1e4`. They execute actual full-profile inference once per original exposure. RAW, actual gained input, external GT and actual mask-on-gained-input panes remain separate. All models retain FAILED labels, all misses/unmatched predictions and unscored unlabelled frames. No model-generated faults, repeated-frame simulations or photograph slideshows are used. Annotated encoding time is not inference capacity.

| Complete real source | Cached BEFORE bestepoch1 TP / missed / unmatched | AFTER author masks TP / missed / unmatched |
|---|---:|---:|
| Thin crack89 | 0 /8 /8 | 0 /8 /7 |
| Daytime horizontal crack129 | 1 /70 /1 | 38 /33 /3 |
| Surface material loss75 | 0 /77 /0 | 0 /77 /29 |
| Night ros_crack21 | 10 /432 /412 | 0 /442 /177 |
| Surface loss76, **post-hoc known-hit source** | 48 /159 /4 | 0 /207 /1 |

These are correlated per-frame annotation instances, not independent defects. BEFORE clips use the unchanged .10 crack/.25 spalling operating point; AFTER is fixed .25 for both. All original BEFORE predictions, source/GT/raw/gain/frame/PTS identities and counts were checked and reused without additional BEFORE inference. Some predictions on unlabelled frames are preserved as unverified rather than counted as false alerts. The two material-loss sources have **zero AFTER spalling matches**, despite visible damage; unmatched crack predictions on those scenes do not become spalling detections.

Crack129 is a concrete improvement on one predeclared daytime recording: precision92.68%, box recall53.52%, compared with the prior1/71 box matches. All33 misses and3 unmatched predictions remain visible. This does not repair crack89, night footage or spalling, and it does not change the failed full122 result. The fixed first publisher-annotated crack129 frame1 remains a miss and its preview was not replaced by a favorable example.

For explicit post-result output inspection, actual encoded crack129 **sourceframe3 at0.150000s** was decoded sequentially and its microsecond PTS verified. PNG: `output/evaluation/construction-cubit-author-v2/videos/qa/crack129-source-frame3-actual-encoded-mask.png`. That frame has one matched box, IoU.82947, score.4050 and1,816 predicted foreground pixels. Visual inspection shows the foreground follows the visible zig-zag horizontal crack path rather than filling the rectangular wall background. The predicted band is noticeably thicker than the dark crack core; there is no pixel-mask GT or calibrated width. This post-result image illustrates an existing result and supplies no independent accuracy credit.

The crack89 fixed first-annotated preview instead shows broad intersecting dark-line masks while the narrow publisher-labelled feature remains missed. Empty-mask counts are zero across all five clips; nonempty masks do not excuse those localization failures.

All five files are `output/construction-cubit-author-v2-{crack_89,crack_129,spalling_75,ros_crack_21,spalling_76}.{mp4,json}`. Every native actual mask has a per-frame bit-packed NPZ/hash; raw source arrays and original timestamps remain unchanged. Export verified complete frame counts and exact microsecond PTS. The executed-source snapshot manifest is `output/evaluation/construction-cubit-author-v2/executed-source-archive/manifest.json`, SHA256 `f43221285ccde1d8c6e09c48e3603894ffa119e0a68b846498271e0b98df5746`. The diagnostic-only integration proposal, SHA256 `deb71b1fc56a6e0f268459afc6e98df069bec767d01a3a41f54575d6a4948230`, uses IDs prefixed `cubit-author-v2-` and explicitly typed lossless-video source presets. It requests no category/default promotion. Root independently audits the media, source pixels, model masks, PTS and counts before registration.

The separate five-class wall continuation completed with all accuracy guards failed; its full result is in `docs/construction-wall-five-class-plan.md`. This new author model supplies crack/spalling only. It does not add humidity, paint identity, bulge depth, flatness, tile installation, hidden-leak or Saudi-code compliance detection.
