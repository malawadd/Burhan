# More real ev-CIVIL training captures: read-only inventory audit

20 October 2026. This proposal reads the saved publisher ZIP directory and existing protocols only. It does **not** download media, decode H5, run a model, open heldout predictions or change any active dataset/run. Source: [official ev-CIVIL release](https://doi.org/10.11583/DTU.25924228), CC BY4.0. Counts below are original source sequence IDs, not verified independent physical defects or guaranteed annotation-qualified frame counts.

**There are37 additional crack-sequence candidates inside the already allowed fitting folders, but no additional spalling sequences there.** The current fit has only four day spall IDs54/55/56/72. More frames from those same objects do not add spall scene diversity. Broader spall options exist in folders excluded by the current exposed-capture protocol; they require a distinct future experiment rather than silently changing that protocol.

## Exact inventory result

The [machine-readable proposal](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/evcivil/unused-train-day-inventory-proposal-round3.json>) records every original camera-H5/label-NPY member pair, its published compressed/uncompressed sizes and exclusion reason. It is reproduced by `scripts/audit_evcivil_unused_training_inventory.py`; no inference is performed.

| Pool | Additional sequence IDs | Classes | Compressed bytes | Uncompressed bytes | Status |
|---|---:|---|---:|---:|---|
| Existing fit folders dset3/4 | 37 | 37crack,0spall | 244,933,892 | 446,923,432 | Preferred metadata candidates; duplicate/timestamp screening still required |
| Other IDs in dset1/2 | 108 | 72crack,36spall | 2,173,585,193 | 3,222,103,408 | Not permitted by the current whole-folder exclusion; conditional future experiment only |
| Entire dset5/6 | 28total | 23crack,5spall | 180,949,931 | 354,399,960 | Excluded in full, including unused IDs; reserved evaluation capture groups |

Existing/reserved/known-duplicate IDs are counted separately in the JSON. Exclusions include all global IDs from the original fit/calibration/heldout mapping, the continuation's extra night IDs, exposed crack10/crack89/spall35, Jan2 heldout32/43/44, and known duplicate crack121. Nov18 calibration and Jan2 heldout captures are never candidate fitting sources. Other originally excluded capture folders, such as day-time challenges and at-ETH material, are not introduced by this proposal.

The preferred37 are:

- **dset3:** crack127,128,132,134. Four member pairs total83,847,644 compressed bytes.
- **dset4:** crack148–152,156–179,181–184.33member pairs total161,086,248 compressed bytes.

The spall70/71 names occurring under dset4 are excluded because their canonical identities occur in heldout dset5. Likewise, dset3 crack129/130 are excluded because those identities are reserved under calibration dset6. This is conservative cross-folder identity protection rather than reliance on different filenames.

## What can and cannot be inferred without pixels

The37 candidates have actual camera H5 and publisher label NPY members. Day labels use the publisher's `*_label.npy` naming rather than a separate `*_frame_label.npy`; actual camera association must use the same first-at-or-after timestamp convention already adopted. A file's existence does not guarantee a positive camera exposure: the additional night sequence13 previously yielded zero qualified camera frames despite containing labels. At the existing24-per-training-sequence cap,37 candidates permit at most888 sampled exposures; actual yield remains unknown.

Directory metadata cannot establish absence of near duplicates. Adjacent IDs may show the same wall/defect, and different folders may share a structure. Known crack120/121 duplication is already excluded, but equal/different file sizes alone cannot clear other cases. Therefore this proposal does **not** claim37 new independent defects, independent projects, or a leakage-free acquired training set.

Before any future fitting, acquisition must retain original member CRC/source SHA and verify annotation timing, source image dimensions and class mapping. Exact camera-pixel hashes must be compared with existing fitting, calibration, heldout and exposed-source pixels. Perceptual-near-duplicate candidates require conservative scene review, especially against all reserved folders; metadata alone cannot provide that clearance. Any uncertain reserved-scene match stays out. Keep actual publisher boxes, and inspect fixed raw/GT examples without choosing only favorable model outputs.

## Evidence-driven options for a future version

1. **Add the37 allowed-folder crack candidates after screening.** About245MB compressed is bounded and substantially expands sequence IDs without relaxing the existing capture-folder exclusions. This may help crack appearance diversity; it supplies no new spall IDs and cannot guarantee improved calibration performance. Existing calibration/heldout remain byte-identical, and the active fit is untouched.
2. **If spall generalization still fails, design a distinct spall-data experiment.** The36 unused-ID spall sequences in dset1/2 are an actual public-data option, but both source groups were excluded because the old displayed challenges came from those capture domains. A future version must declare that changed training scope before fitting, exclude all exposed/reserved canonical IDs and actual/near-duplicate views, keep entire dset5/6 and night evaluation groups out, and retire any claim that the old exposed-domain examples provide independent validation. A metadata-hash-selected bounded subset can avoid downloading all2.17GB. This is a proposal, not authorization to mutate the active protocol or evidence that those candidates are safe already.

The current model's failure should remain reported while new data are considered. More source names, frames, or training views alone do not establish working thin-crack/spall localization; the unchanged calibration and eventual frozen heldout results must demonstrate it.
