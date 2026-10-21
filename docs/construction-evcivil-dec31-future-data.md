# Separate Dec31 real-night data proposal: acquired, not accepted for fitting

The future-only acquisition contains **20 real camera sequences,2,164 original camera exposures,841 publisher-timestamp-qualified crack exposures and410 uniformly sampled proposed views**. All841 qualified exposures and all410 proposed views genuinely trigger the unchanged low-light gain; this is actual dark camera input, not synthetic darkening. No NN ran, no model/default changed, and nothing was copied into the current832 or approved880 training datasets. Visual/duplicate acceptance is still pending coordinator review.

## Before-pixel selection and integrity

The [immutable acquisition protocol](C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/evcivil-dec31-future/protocol-before-acquisition.json) was saved before any candidate pixel decoding, SHA256 `9c6c68d1a7c86491bc92f5fb2bb93c056413f5ea4d0e3b830a4ca0cf91f19919`. It pins the script, publisher record, central-directory inventory, every existing exposures/heldout protocol, latest880 data protocol, original annotation association and shared preprocessing source hashes.

The original [ev-CIVIL publisher release](https://doi.org/10.11583/DTU.25924228) declares **CC BY4.0**. The selected source is the public `evCIVIL_Field_dataset.zip` camera-only byte ranges. Every downloaded member has exact original ZIP name, size, compressed size, offset, compression method, CRC32 and computed SHA256. Downloaded raw H5 and NPY bytes remain untouched. No event streams, complete10.87GB archive, accounts or paid services were used. The original full-archive hash was not verified; local SHA identities and publisher ZIP member CRCs/lengths were verified, not falsely presented as publisher-supplied SHA checksums.

All28 Dec31 camera folders were inventoried before selection. The union of **existing TRAIN, CAL, heldout, old exposed and conservative aliases** excluded eight candidates: `31_crack_4,6,7,13,15,21,32,43`. The full exclusion union also explicitly preserves CAL21/22, heldout32/43/44, conservative121, original exposed10/89/spalling35 and all protected class-specific global IDs. For example, `crack_27` is not excluded solely because a different class `spalling_27` exists. The remaining20 complete camera/label pairs were selected before observing darkness or defects, with no replacement or relaxed exclusions.

The actual selected payload is55,347,698 compressed bytes across40 original members. Dec31 publishes `_frames.h5` and `_label.npy`, with **no separate `_frame_label.npy`**. This absence is recorded; no substitute frame annotations are invented. All original asset integrity checks finish before pixel/annotation analysis.

## Exposure, labels and actual darkness

Each source camera timeline has strictly increasing original timestamps. PTS is the original camera timestamp minus its first original timestamp, with no frame manufacture or re-timing. The established shared association selects the first publisher annotation timestamp at or after the exposure, with a maximum gap of median camera period/2+1000us. Actual qualified association deltas range1–242us. Publisher class0 maps to crack and1 to spalling; source boxes are clipped to native image bounds and all valid crack/spall boxes in selected exposures are retained. Labels provide **boxes, not pixel masks**. Unannotated camera exposures are never turned into sound negatives.

At most24 unique uniformly spaced qualified exposure indices per sequence were saved as unchanged grayscale PNGs. Selection never uses darkness, image appearance, model confidence or benchmark results. The [asset/view manifest](C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/evcivil-dec31-future/asset-manifest.json), SHA256 `b549ed181a57ce847298614e5e83463743f9931a1881d70dd34a02fd0ad31bdc`, retains every raw asset identity, all2,164 camera p99/gain/pixel hashes, all841 qualified indices/timestamps/GT and410 chosen input identities.

| Scope | Exposures | Actual low-light gain applied |
|---|---:|---:|
| All original camera frames |2,164|2,162|
| Timestamp-qualified crack exposures |841|841|
| Fixed uniform proposed views |410|410|

All-camera native gray p99 min/median/max is2/8/77. Selected-view p99 is3/7/27, with actual shared gain min/median/max4.74/8/8. This is materially different from the existing20 night-named TRAIN views that did not trigger gain. Very dark, noisy inputs can still lack recoverable visual detail: native darkness and publisher labels **do not themselves prove a model can detect the crack**.

## Duplicate screening and acceptance blockers

The [read-only duplicate audit](C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/evcivil-dec31-future/qa/duplicate-audit.json), SHA256 `10f4f3b5987f877002c95827b66e50d52dc179eee8dec0ceb33217d6bf8795b7`, finds:

- **Zero exact matches** for all2,164 original candidate gray camera pixel hashes against stored protected-split source identities.
- **Zero exact matches** for410 proposed PNG, native grayscale and native BGR identities against4,047 stored hash values from existing fit/CAL/heldout raw/gained manifests.
- A non-neural near-duplicate screen against1,716 unique already-used TRAIN file representations flags **one proposed view**: `31_crack_40` frame11 against raw/gained versions of existing TRAIN `crack_143` frame101. dHash4 with correlation−.179 suggests a coarse dark-texture collision, but its disposition remains for explicit visual review.
- **107 cross-candidate pair flags**, retained in full for review; similarity is not automatically physical identity or independence. Correlated frames from one sequence are excluded from cross-sequence flagging, not counted as independent captures.

The screen uses64-bit dHash plus mean-centered32x24 gray correlation. It is not exhaustive proof against geometric transformations or unseen-frame duplication. CAL/heldout pixels were not opened; their stored exact hash metadata was used, and no reusable universal heldout perceptual fingerprint cache was found. The absence of universal physical-site/perceptual proof is a limitation, not automatic rejection of every safe unused TRAIN candidate. **A specific unresolved alias/duplicate flag is an acceptance blocker.**

All109 rendered pair entries (two known-TRAIN representations plus107 cross-candidate flags) have raw review pages and19 supplementary [fixed shared-gain review pages/index](C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/evcivil-dec31-future/qa/gain-pair-review/index.json). The gain pages apply only the actual immutable shared input policy for visibility; they are not model outputs, synthetic damage or altered source assets. Initial independent visual inspection of page0 shows distinct visible geometry for the known-TRAIN candidate pair; full coordinator disposition is still required. The20 fixed sequence sheets show first/middle/last sampled exposures as **RAW | actual shared-gain INPUT | publisher GT, NO MODEL**.

No acceptance JSON clears these sources, no training YAML references them and no fit starts automatically. Any later acceptance/data version must link these immutable hashes, explicitly resolve flagged source relationships, preserve complete existing CAL/heldout groups and global aliases, and count alternate views conservatively. Physical project/site independence remains unknown. These sources contain genuine recorded exposures; the still PNG/contact-sheet preparation is for training/QA, not a photo slideshow presented as video.

## Fixed sequence review

Counts below are original camera frames / timestamp-qualified crack exposures / proposed uniform views. All proposed views in every row receive actual gain; the linked boxes are publisher annotations, never local model detections.

The later [complete pair-review recommendation](C:/Users/IAM/Documents/hackathon/opencv/project1/docs/construction-evcivil-dec31-pair-review.md) records all109 inspected entries:77 different visible layouts,32 uncertain, zero confirmed same damage layouts. Its additive [v2 JSON](C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/evcivil-dec31-future/qa/pair-review-v2.json), SHA256 `50d9a1d8b7f857882572b54d9fd2f4fa4b853b72a86d159565a05af34e2fe021`, recommends conservatively quarantining all six unresolved endpoints30/34/40/45/49/50 (136 views). The other14 sequences/274 views remain only eligible for coordinator source-quality review, not accepted fitting data. Source52's largely black flagged frame also requires a separate visibility/quality decision.

| Sequence | Camera / qualified / selected | Fixed RAW/input/GT sheet |
|---|---:|---|
| 31_crack_16 | 162 / 19 / 19 | [Review](C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/evcivil-dec31-future/qa/31_crack_16-fixed-raw-input-gt.jpg) |
| 31_crack_17 | 111 / 37 / 24 | [Review](C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/evcivil-dec31-future/qa/31_crack_17-fixed-raw-input-gt.jpg) |
| 31_crack_18 | 52 / 15 / 15 | [Review](C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/evcivil-dec31-future/qa/31_crack_18-fixed-raw-input-gt.jpg) |
| 31_crack_20 | 75 / 19 / 19 | [Review](C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/evcivil-dec31-future/qa/31_crack_20-fixed-raw-input-gt.jpg) |
| 31_crack_24 | 96 / 30 / 24 | [Review](C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/evcivil-dec31-future/qa/31_crack_24-fixed-raw-input-gt.jpg) |
| 31_crack_27 | 66 / 17 / 17 | [Review](C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/evcivil-dec31-future/qa/31_crack_27-fixed-raw-input-gt.jpg) |
| 31_crack_30 | 92 / 92 / 24 | [Review](C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/evcivil-dec31-future/qa/31_crack_30-fixed-raw-input-gt.jpg) |
| 31_crack_34 | 180 / 60 / 24 | [Review](C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/evcivil-dec31-future/qa/31_crack_34-fixed-raw-input-gt.jpg) |
| 31_crack_35 | 146 / 41 / 24 | [Review](C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/evcivil-dec31-future/qa/31_crack_35-fixed-raw-input-gt.jpg) |
| 31_crack_37 | 88 / 38 / 24 | [Review](C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/evcivil-dec31-future/qa/31_crack_37-fixed-raw-input-gt.jpg) |
| 31_crack_40 | 76 / 60 / 24 | [Review](C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/evcivil-dec31-future/qa/31_crack_40-fixed-raw-input-gt.jpg) |
| 31_crack_41 | 82 / 3 / 3 | [Review](C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/evcivil-dec31-future/qa/31_crack_41-fixed-raw-input-gt.jpg) |
| 31_crack_42 | 135 / 73 / 24 | [Review](C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/evcivil-dec31-future/qa/31_crack_42-fixed-raw-input-gt.jpg) |
| 31_crack_45 | 89 / 16 / 16 | [Review](C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/evcivil-dec31-future/qa/31_crack_45-fixed-raw-input-gt.jpg) |
| 31_crack_46 | 127 / 64 / 24 | [Review](C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/evcivil-dec31-future/qa/31_crack_46-fixed-raw-input-gt.jpg) |
| 31_crack_48 | 85 / 9 / 9 | [Review](C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/evcivil-dec31-future/qa/31_crack_48-fixed-raw-input-gt.jpg) |
| 31_crack_49 | 119 / 51 / 24 | [Review](C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/evcivil-dec31-future/qa/31_crack_49-fixed-raw-input-gt.jpg) |
| 31_crack_50 | 170 / 115 / 24 | [Review](C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/evcivil-dec31-future/qa/31_crack_50-fixed-raw-input-gt.jpg) |
| 31_crack_51 | 102 / 33 / 24 | [Review](C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/evcivil-dec31-future/qa/31_crack_51-fixed-raw-input-gt.jpg) |
| 31_crack_52 | 111 / 49 / 24 | [Review](C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/evcivil-dec31-future/qa/31_crack_52-fixed-raw-input-gt.jpg) |
