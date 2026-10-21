# LCSD: acquired real low-light stills for a future training proposal

20 October 2026. Acquisition initially changed no active data or model. After bounded visual review, all41 pairs were accepted solely as TRAIN transfer into the separately versioned1171-view continuation. No LCSD accuracy benchmark, synthetic image, still-loop video or empty negative label was created; ev-CIVIL calibration/heldout identities and ground truth remain fixed. The acquisition-only statements below describe the preserved earlier stage.

The [official CrackNex repository](https://github.com/zyaocoder/CrackNex) links the [LCSD distribution](https://huggingface.co/datasets/Chrisathy/LCSD) directly. Its card declares Apache-2.0. This is the author-linked distribution's rights declaration; it does not independently clear unlisted source material. The [authors' dataset description](https://arxiv.org/html/2403.03063v1#S4.SS1.SSS2) identifies genuine iPad Pro campus captures. These are released 400×400 still photographs, not camera videos or proof that every image was captured outdoors at night.

| Pinned source | Identity |
|---|---|
|HF dataset revision|`254e9a987d1ee9d925472a42750c6f5f58112d9f`|
|Official author code revision|`49310ef4a6e6648d1a3c50284034883aaddce5e2`|
|Protocol saved before pixel downloads|`2e8009cc9935b8a676e47137d8a9e24e04506bb1511f522c5861ff522ea9edff`|
|Acquisition/audit manifest|`6a138d4a26a84ec5ed2da7eef93b3b9519d3d745df1d7fc7428e54aef7478ce1`|

The original `val.txt` lists exactly 41 image/mask pairs. The author's paper calls these the low-light test set. Any later accepted use here would explicitly repurpose them **solely as our TRAIN transfer data**, relinquishing LCSD test/validation benchmarking. The complete ev-CIVIL calibration/heldout sources and existing 832/880-image versions stay fixed. Physical scene IDs are unavailable, so source-image distinctness is not a claim of independent physical sites.

## Original masks and exact proposed box geometry

The actual [author loader](https://github.com/zyaocoder/CrackNex/blob/49310ef4a6e6648d1a3c50284034883aaddce5e2/dataset/fewshot.py) identifies class 2 as low-light crack, 0 as background and 255 as ignore. It reads the unconverted PIL array. The acquisition preserves original PNG bytes and checks those exact codes; it never guesses a grayscale inversion.

All 41 downloaded masks are two-dimensional `L` images with only values **0 and 2**, matching that convention. No image has an ignore pixel. Geometry agrees exactly with the paired released image at 400×400. Proposed boxes are every eight-connected class-2 component, including one-pixel components: no closing, dilation, merging, minimum-size rejection or top-K. Bounds are tight inclusive-pixel geometry `[xmin, ymin, xmax−xmin+1, ymax−ymin+1]`. This produces **92 crack component boxes**, while retaining every original pixel mask. These are mask-derived regions rather than publisher object-instance boxes.

No YOLO labels or training copies were produced. The manifest's `ordinary_box_training_eligible` flag records mask-code/geometry eligibility only; it is **not visual acceptance or permission to import data**. Crack masks do not certify the rest of a photograph as physically sound or free of other damage. An unresolved/empty mask would remain quarantined, never become an invented negative image.

## Measured inputs and leakage screen

All 41 actual raw photographs trigger the existing shared low-light policy. Native grayscale p99 is **9 / 24 / 52** minimum/median/maximum; bounded global gain ranges from **2.46× to 8×**. Raw RGB bytes, decoded raw BGR identity, original mask identity and actual native gained-input identity are separately hashed. Gain changes intensity only; the raw photographs remain unchanged.

The exact-file/native-gray screen finds **zero exact matches** against its predeclared materialized input references. There is one conservative dHash flag: LCSD_0231 versus the existing TRAIN spalling13 frame 320, Hamming distance 4 and normalized gray correlation .847. It needs visual pair review; it is not proof of duplication. No within-LCSD pair crosses the declared similarity flag. The screen covers the reference manifest's materialized sources, not every unselected exposure or every original full-resolution photograph, and cannot establish physical-site independence.

Native first/middle/last sheets are available under `datasets/construction/lcsd-low-light-future-r4/qa/`:

- `fixed-00-LCSD_0227-raw-mask-input.jpg`
- `fixed-20-LCSD_0254-raw-mask-input.jpg`
- `fixed-40-LCSD_0285-raw-mask-input.jpg`

Each shows the untouched released image, explicit source mask codes, source-mask overlay and actual shared gained input separately. Red means source class 2 and magenta would mean source ignore 255; none is a model prediction. Root visual review and the flagged pair review precede any accepted combined version.

Acquisition is reproducible with `scripts/acquire_lcsd_future_low_light.py`; it refuses to repeat a completed manifest. The first invocation stopped before source metadata/pixels because an unrelated reused helper imported an unavailable HDF5 library. Its console is preserved. The successful invocation uses the same pure non-neural signature in `scripts/construction_image_signatures.py`, with no installation. Both processes exited; acquisition contributes no inference throughput claim.

## Additive acceptance and separately versioned training

The root reviewer inspected the three fixed raw/source-mask/gained sheets and directly compared the flagged LCSD_0231 photograph with TRAIN spalling13 frame320. These show distinct crack branching versus mottled surface loss; no duplicate was confirmed. This is a bounded three-sheet plus flagged-pair review, not manual inspection of every photograph or proof of physical-site independence. Additive acceptance receipt `acceptance-v1.json`, SHA-256 `28b13decfb62bd695dcc5a68af3cc7cbf5874191108e9184342d943d218e355f`, pins the original acquisition, exact41 source identities/masks and92 component boxes.

Only the new `evcivil-round4-real-lowlight-gain` dataset imports these TRAIN stills. Its1171 fitting views also include approved broad880 data and250 accepted genuine Dec31 camera exposures. Original832/880 roots, the same122 calibration identities/ground truth and all heldout data remain unchanged. The four-epoch adaptation is protected by protocol `f77168288653f986ddf23a8c601141767d84a172ee8e0d31f9d5059c95da625d`; training these released test photographs forfeits any subsequent LCSD benchmark claim. No photographs are converted into purported camera footage.
