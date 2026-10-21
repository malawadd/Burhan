# Construction round4 mixed-supervision review

Reviewed statically: `construction_round4_mixed_loss.py`, `prepare_construction_round4_mixed_data.py`, `train_construction_round4_mixed.py`, installed Ultralytics loss/assignment implementation, and shared serving preparation. No model construction, checkpoint loading, NN, data decode or gradient test was performed by this reviewer.

**No material math, gradient-routing, class-map or geometry blocker found in the current source draft.** This is source review, not approval of an as-yet unprepared dataset/protocol or proof of accuracy.

The actual-mask branch retains fractional coverage from native source polygons at640, with matching letterbox geometry. Parent loss upsamples32 prototypes to that plane before the custom per-instance ROI computation. Half foreground/background-mass-balanced BCE and half soft Dice operate on the actual fractional ROI target. The loss is averaged over matched foreground anchors by the parent criterion; an extra inverse box-area factor is unnecessary for this declared mass-normalized loss. No rectangle is filled as a pixel target.

Box-only camera samples call the detection criterion and add explicit graph-connected zero prototype/coefficient terms. The five loss fields retain installed order: box, mask, class, DFL, semantic. The per-image known-class mask broadcasts across all classification anchors, giving an unannotated other class zero BCE gradient. It does not imply all other real objects of the annotated class were exhaustively annotated. Same-class partial-label uncertainty remains relevant to training and unmatched-box interpretation.

The source state is restored strictly with417 finite tensors, author crack0/spalling1 mapping preserved. Existing camera0spalling/1crack maps explicitly to the author order. Native camera gain occurs once before RGB/255 letterbox. Stored uint8 roundtrip matches the source uint8 preparation; horizontal flips update input, normalized box center and masks together. First four model layers are frozen and put in evaluation mode, preserving their normalization state; later trainable normalization is a declared part of adaptation.

The installed task-aligned assigner expands very small *candidate-search* sides while retaining actual target boxes/masks. Therefore a claim that network stride categorically prevents continuous thin boxes would be incorrect. Actual assignment/mask-gradient checks on the selected real TRAIN cases remain necessary.

Zero box-only mask **loss gradient** is the correct check. It is stronger to claim mask weights cannot move on such an optimizer step: AdamW decay and stored momentum from earlier actual-mask batches can still update parameters with zero current data gradient. The reviewed source does not require that stronger claim.

The proposed real TRAIN probe checks finite actual forward/backward, nonzero detector/class gradients, zero unsupported-class score gradient, zero box-only mask gradient, nonzero actual-mask gradient and first-four state identity. Source/data manifest pins, actual prepared-target QA and an exact root-approved plan remain prerequisites to any probe/900-step fit. All three snapshots and unchanged calibration guards are required before selection; private video results must not feed fitting or selection.

## Focused pre-freeze update

Root subsequently approved camera **positive-assignment-only classification BCE**, addressing same-class partial annotation too. The reviewed branch multiplies the known-class mask by the actual assigned soft target being positive; unlabeled background anchors and the other class receive zero classification data gradient. Actual polygon/verified-negative sources keep their declared supervision. This changes no camera box into a mask and does not establish that unlabelled surfaces are healthy.

The memory optimization overrides parent mask-loss iteration to take each matched GT mask as a view rather than advanced-index-copying a full640 plane for every foreground anchor. Under the enforced batch1 contract, source-instance order and `target_gt_idx` match the parent. Native ROI coordinates, prototype/coefficients and division by total foreground anchors preserve the declared objective. The parent calls this override only with nonzero foreground assignments. **Focused static review PASS** for both changes; actual tiny value/gradient equivalence fixtures and the bounded real TRAIN probe remain separately recorded by the owner. No NN or test execution by this reviewer.

## V2 numeric correction and V3 grouping review

V1's first real TRAIN case passed; the second encountered an actual area-resampling target of1.0000001192092896 and failed the strict range guard. The failed receipt/source archive remain intact. V2 adds only fixed±1e-6 tolerance for that bounded arithmetic noise, a non-inplace local clamp to[0,1], and a finite-logit guard. Larger/nonfinite errors still reject; source NPZ/masks and stroke geometry remain unchanged. Focused static PASS for proposal `6117468619b73ac9b92d1e266161e6269d55860e52655b02765dde61e4f46c03` was sent before root's separately authorized probe.

After V2's actual five-case probe passed but exposed expensive scalar per-anchor backpropagation, V3 proposed a math-equivalent grouping implementation. Independent source review of `construction_round4_mixed_loss_v3.py` and `train_construction_round4_mixed_v3.py` found no material blocker:

- Grouping is per image and actual matched GT index. Repeated coefficients for that GT require exactly equal true ROI coordinates; masks from different GTs are not pooled.
- BCE and Dice spatial reductions remain independent per coefficient row. Those losses are **summed**, preserving each anchor's contribution, then divided by total foreground anchors exactly as V2. Dividing by group count would change the objective; the code does not do that.
- A grouped coefficient/prototype contraction retains gradients to both tensors. The shared full-prototype zero graph path is needed once rather than for every anchor. Parent's zero-foreground branch is unchanged.
- The bounded probe compares the same actual outputs with unchanged scalar V2 on detached tensor copies. The reference constructs a criterion, not a second model; it performs no second real-input network forward. Full score/box/prototype/coefficient output gradients and losses are compared within declared float32 tolerances, plus actual assignment metadata.
- Root's source state, true-mask data, five fixed real TRAIN cases,900-step recipe, classes, thresholds and calibration guards remain unchanged in new artifact namespaces. No CAL/private pixels or second-reference input images enter this review.

**Focused V3 source review PASS**, before its proposal/probe. This is mathematical equivalence subject to measured float32 tolerances, not a bit-identical promise. “One real TRAIN forward per case” is precise: unchanged official model construction may perform initialization work outside the timed real-input forward/backward. The owner must retain actual numeric/gradient/RSS results and any failure. No additional NN/tests were run by this reviewer.
