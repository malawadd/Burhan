# Fitting-source aspect-ratio admission proposal

FIT/TRAIN-only source image header+byte hash audit; no image decode, mask read, calibration pixels, official ZIP/test contents or NN.

Preserved ratio gate: **[2.3, 2.7]**. Source identity checks: **PASS**. No serving policy changed.

| Fitting domain | Images | Width range | Height range | Observed H/W | Outside preserved gate |
|---|---:|---|---|---|---|
| SDD1_TRAIN | 239 | 500–500 | 1235–1282 | 2.47000000–2.56400000 | 0 (0.00%) |
| SDD2_inner_FIT | 1865 | 184–240 | 602–660 | 2.61206897–3.40760870 | 1793 (96.14%) |
| combined_fitting_sources | 2104 | 184–500 | 602–1282 | 2.47000000–3.40760870 | 1793 (85.22%) |

Proposed separate TRAIN-derived geometric envelope: **[2.4206000000, 3.4757608696]**, using observed min×0.98/max×1.02. This rule uses fitting data only; the2% margin is a bounded engineering allowance, not a measured accuracy result.

All included appearances were already eligible fitting sources before this header audit. A2% multiplicative margin accommodates small acquisition/crop dimension changes; it is an engineering proposal, not an empirically validated accuracy tolerance.

- Training-dimension coverage is not physical domain, material, imaging-quality or defect-localization validation.
- The continuous interval includes ratios between observed sources; data do not establish performance throughout that interval or under arbitrary stretching.
- Keep original frozen policy/reports unchanged; any admission variant requires distinct pinned metadata and a pre-opening protocol, with abstention accounting reported separately.
- Header/source identity audit cannot qualify15FPS, blind-test generalization, product rejection or acceptance.

No official test archive, calibration images, annotations or image pixels were read. PNG IHDR/JPEG SOF headers supplied dimensions; complete source-file bytes were hashed. SDD2 hashes agree with the protected fit manifest when the identity check passes. The raw proposal retains every source path/hash/dimension and its current-gate result.

[Machine-readable identities and proposal](C:/Users/IAM/Documents/hackathon/opencv/project1/output/optimization/factory-round3/fit-admission-geometry-proposal.json)

Any later admission variant must retain the same frozen NN/thresholds while naming its changed gate separately. The preserved policy remains the baseline; a broader geometric gate cannot be credited as improved learned defect localization without actual spatial evaluation.
