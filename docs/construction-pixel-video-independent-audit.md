# Independent audit of the fixed crack pixel videos

All six fixed originalTUT/FINAL300 recordings pass independent saved-count, source-input, native-mask and actual encoded timeline checks. The audit runs no model and grants no accuracy or real-time qualification. The first app-environment invocation stops at missing PyAV before data processing; the existing export environment completes the same auditor without installing packages. Both consoles remain.

Machine-readable receipt: `output/evaluation/construction-pixel-videos/independent-pixel-video-audit.json`, SHA **`47d35050de6a520c204acfae6c7431f8c9dbb57070ea93049ba2782ad49b267c`**. All350 original exposures per model are retained:24 crack_89,104 crack_129 and222 ros_crack_21. Actual MP4 decoding reproduces their exact original microsecond PTS shifted to zero, with no missing or added frames.

| Source | Frames per model | Original TUT TP / FN / unmatched boxes | FINAL300 TP / FN / unmatched boxes |
|---|---:|---|---|
|crack_89|24|0 /8 /20|0 /8 /49|
|crack_129|104|8 /63 /60|7 /64 /84|
|ros_crack_21|222|68 /374 /1667|136 /306 /3563|
|Total|350|76 /445 /1747|143 /378 /3696|

The fixed>.50 comparison increases publisher-box matches on the night source while greatly increasing unmatched boxes; it fails to improve the exposed crack_89 sequence and slightly regresses crack_129. All three are existing exposed/calibration development sources, not independent heldout videos. Unmatched boxes follow the declared publisher whole-instance IoU≥.50 protocol, including fragmented coverage and limited annotation scope; they are not independently confirmed material false alarms. No video pixel GT or completely healthy population is available.

Every one of700 raw camera BGR and once-gained native input hashes agrees with untouched original arrays. Bit-packed raw and serving masks reproduce recorded native hashes and positive-pixel counts. Serving pixels are a subset of the raw strict-threshold mask, retain complete raw8-connected components of area≥4, and every emitted candidate box equals an actual serving-mask component, with at most100 components. Separate original source bytes and protocol/model identities also match.

The video mask assets do not persist the actual384 probability arrays: their recorded probability hashes cannot recreate thresholding, component mean scores or top100 ranking without inference. A separate additive check verifies the format of all700 recorded probability hashes and binds the executed auditor source SHA; it does not claim probability reproduction. Numerical and complete serving parity are established separately on the protected122-view calibration for every fitted checkpoint. Trace colors display model masks; they do not label true positives against nonexistent video pixel annotations.

The trained model remains rejected by its all-three calibration guard and also regresses actual reserved-photo pixel recall/IoU. These complete before/after videos expose its behavior rather than justify promotion, a15FPS claim or construction-code acceptance.

## Separate post-hoc known-hit illustrations: static review only

The separate `compare_construction_posthoc_known_hit_videos.py` protocol deliberately chooses CAL crack129 and spalling76 after known matches are observed. This outcome-based selection is disclosed as **post-hoc illustration**, with no independent accuracy credit, model promotion or change to the existing cases/guards. Independent static review finds no material blocker: only those two CAL sources are whitelisted and heldout membership rejected; source/model/point/metadata/cache/code identities are pinned; reused before129 predictions must match every actual raw/input hash, GT/index/PTS and complete frame count; all other predictions use the unchanged full-frame model. Publisher GT is separate from model input, all exposures are retained, and unlabelled exposures are unscored. This review runs no NN, tests or image decode and does not imply an export has occurred. Explicit coordinator media release and a subsequent complete output/count/PTS audit remain required.
