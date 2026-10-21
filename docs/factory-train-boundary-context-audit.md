# TRAIN-only factory boundary/context audit

Protected239 original SDD1 TRAIN views only; actual masks and JPG headers/file hashes. No calibration/comparison pixels, no SDD2 archive/test, no NN.

The protected30 fitting items contain239 views, 31 positive views and33 training-grid GT components. Original masks contain33 components. The exact512×1280 occupancy projection matches the active trainer. These components are not counts of independent physical damage sites.

Distances are empty pixels between each component bounding box and a FULL SOURCE IMAGE edge, not a verified mechanical part edge.0 means touching. A component contributes once to “any edge” and can contribute to multiple directional columns.

| Distance at most(px) | Any source-image edge | Left | Right | Top | Bottom |
|---:|---:|---:|---:|---:|---:|
| 0 | 30 | 0 | 30 | 0 | 0 |
| 4 | 30 | 0 | 30 | 0 | 0 |
| 8 | 30 | 0 | 30 | 0 | 0 |
| 16 | 30 | 0 | 30 | 0 | 0 |
| 32 | 31 | 0 | 31 | 0 | 0 |

Minimum-edge distance min/median/max: [0, 0.0, 162]. Components with at least32 real source pixels of context on every side: 2/33.

**20/33 component boxes are wider than the previous256-pixel crop.** This agrees with the preserved TRAIN orientation audit ({'horizontal': 32, 'diagonal': 1, 'vertical': 0}). Flips alone do not exchange horizontal/vertical orientation.

| Patch | Cannot wholly contain GT bbox | Cannot fit bbox plus32px each side by size |
|---|---:|---:|
| Previous640×256 | 20 | Not measured |
| D 256×256 →320×320 | 20 | 28 |
| D 384×384 →320×320 | 5 | 11 |
| D 512×512 →320×320 | 0 | 1 |

- Containability is a necessary geometric possibility, not observed sample frequency or NN accuracy. Positive-centred jitter can still crop part of a containable component.
- D chooses256/384/512 real source squares uniformly and resizes isotropically320. A512square retains both horizontal source-image bounds; no square spans all1280vertical pixels.
- Quarter turns/flip rotate patch boundaries and labels together. Most patch borders are internal crop boundaries, not full source-image bounds; zero-padded convolutions also see these internal cut edges.
- The previous640x256rectangle cannot contain any component widerthan256 regardless of positive centring. Larger D squares remove that impossibility for some/all components, but may downsample thin detail.
- Right-source-image-edge GT is abundant:30/33 touch it, whereas no GT component is within32px of the other source edges before augmentation. This is directional boundary concentration, not generic lack of boundary GT.
- This only supports a future TRAIN-derived boundary/context hypothesis if a frozen model underperforms. It selects no padding, threshold or new model and makes no claim about original comparison misses.

No causal failure diagnosis or architecture recommendation follows from these counts. The measurable limitations are directional source-boundary concentration and patch-context truncation: right-edge examples dominate, while left/top/bottom source-image-boundary examples are absent within32px before augmentation. Quarter turns supply transformed patch views; they do not establish independently captured conditions at those other image edges. Any future hypothesis needs a separately frozen experiment after the current model is evaluated. The current model, padding, operating point and serving admission policy remain unchanged.

[Source identities and all component distances](C:/Users/IAM/Documents/hackathon/opencv/project1/output/optimization/factory-round3/attempt4-train-boundary-context-audit.json)
