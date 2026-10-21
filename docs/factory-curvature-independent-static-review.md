# Independent static review of native-width curvature training

This review reads source only; no model inference, training, image decoding, source-pixel inspection or performance test is performed. The protected active fit and selection rule are unchanged.

No material active geometry or data-separation defect was found.

- The inverse remap correctly uses `source_y=baseY+destination_y+f(x)`. Retained selected-component bounds use `source_y−f(x)` with a one-pixel nearest-sampling margin, not the opposite sign. The per-column unit-step vertical map preserves occupancy when the component fits, and a separate actual selected-mask remap asserts exact retained pixel count.
- Curvature is bounded to amplitudes8/16. BaseY16..624 makes every lookup in the actual1280-row source; the code checks map bounds before bilinear image and nearest-label remapping. The declared constant border is unreachable. If selected positive geometry cannot fit, identity fallback recomputes its supported placement and retains the component; it does not fabricate GT.
- Hard/unbiased crops remap the full real occupancy target, including all defect pixels inside the sampled coordinates. The same curvature Bernoulli applies to positives, hard/unbiased and true normals. Matching180-degree/horizontal transforms follow image/mask together. SDD2 rehearsal explicitly retains a sampled positive pixel rather than claiming a whole component.
- The immutable C200 source PT/ONNX/metadata/freeze identities are checked before execution. Strict complete state loading prevents uninitialized same-architecture residual keys. All239 SDD1 source-pair hashes and consumed SDD2 inner-FIT pair hashes are verified. No comparison/test image is accessed by the sampler.
- Required phase-max/attempt3 flags activate the fixed weakest-positive rank and precision guard. Separate prefix, source-pinned calibration cache and200/400/600 defaults keep this branch distinct. The freezer requires the completed fixed600-step fit, all three exact checkpoints and identical ordered80-view/10-positive CAL identities, metadata/policy/source hashes and unchanged grid/rank.

Optional future hardening: freezer source validation currently requires the same80/10 identity across checkpoints; binding its digest to the already protected known CAL identity would additionally reject a self-consistent replaced source set. The actual calibrator reads the immutable source model's calibration partition, so this is not an active data-selection flaw or a reason to change the running protected protocol.

Reviewed source SHA256 values:

- `scripts/factory_curvature_sampling.py`: `2fa0d7b3945456678e8aaa8880fc64a70fe6fe25ae0b3937e992cb51c47559d3`
- `scripts/train_factory_round3j.py`: `af94bcea89268b6fd77d858301fe3f805a76db6464224e9217fd54e40b8324d8`
- `scripts/freeze_factory_round3j.py`: `30e56cf80cbb85c7bf7768ae93c50476b12dfb9e0e8d740eaa3f2643fc32193a`
- `scripts/calibrate_factory_round3.py`: `9832792d6c187b75caf126e0a5822f4e04085ad0a976d3d952c2a016f808a705`
