# L4: one fixed real-TRAIN BN initialization

This is a new, separately named experiment. L1/L2/L3 numeric failures remain unchanged. Updating running BN statistics is TRAIN-based model fitting, even without gradient updates. At preparation time, **no L4 model forward, BN update, export, fit or CAL inference has run**.

The motivation is numerical sensitivity in the original random student, not evidence that this method fixes defects. The single earlier TRAIN case had87.78% of deepest nonzero student activations below the FP16 normal boundary; that neither establishes a GPU cause nor measures a population distribution.

## Fixed source selection and actual factory configuration

`selected64-before-any-forward.json` selects two normal views per each of30 original TRAIN items, then the third normal view from the first four items. Its ordered source identity is `bdd7458ceede545360d640664b0009ab41e7558c21c9311f6611177584842d12`. Selection and source SHA verification used metadata/bytes, without image decodes or forwards. The original208-normal identity stays `f7ec8222431065047de889ff2b03f24172bc59257d6962c51103377566f733b4`. No positive TRAIN, CAL, exposed comparison or official SDD2 test is used for BN fitting.

Actual installed Torch2.14.1/torchvision0.29.1 factory inspection found18 student BN modules, all with epsilon **.001**, momentum **.01** and counter0. This was module construction/attribute inspection only, without weight loading or forwards. The installed `mobilenetv3.py` SHA is `8c817b1ccc10edfa823ed598d63c3b3d3d7b28c208558117440ab5ec89eb4122`. The earlier L1 protocol's .1 momentum text did not match the actual factory; no L1 BN fitting/600-step training ran. This version explicitly sets .01 for future gradient training.

## Implementation and fail-closed gates

`prepare_factory_stfpm_bn.py` freezes all parameters, resets only student BN running state and counters, then uses cumulative momentumNone for exactly64 full real TRAIN views. It preserves each actual epsilon. The teacher never forwards during this pass. Image preprocessing is the unchanged full512×1280 INTER_AREA resize and RGB/ImageNet normalization; zero/white numerical initialization tensors never enter BN fitting.

Only the exact student running-mean/running-variance/counter keys may change. Complete before/after state hashes guard all convolution weights, BN affine parameters, full teacher state and input mean/std. Every counter must be64, variances finite/nonnegative, and restored momentum.01. A distinct full fitted state and receipt are saved once; interruption or any existing state/start marker prevents silently repeating the pass.

The new exporter regenerates Torch scores on the same fixed normal/zero/255 **inputs** from the saved BN-fitted state. It exports the original architecture to new ONNX/sidecar names. Fitted BN folding changes deployed model bytes; old speed or numeric receipts do not transfer. The GPU recipe remains L3's f16/no-WinogradTrue, with actual compiled configuration/runtime precision queried and saved.

The first actual GPU pass must have max absolute score error≤.01 on all three fixed references, finite bounded output and nonconstant real TRAIN output. Only a successful new numeric receipt permits the new16-source/64-complete-profile capacity run. All measured calls must infer, and mean capacity must be≥15. First receipts, including tails/failures, are retained.

Any later600-step normal-only training needs another explicit execution release. It loads the exact saved BN-fitted full state before the optimizer and checks its teacher/student digests against the capacity graph. Momentum is not in `state_dict`, so the trainer explicitly restores .01. The original batch1/CPU1/AdamW/crop/source schedule stays fixed;200/400/600 checkpoints require all18 counters equal64+step, finite complete student state and unchanged teacher. Each must pass actual GPU numeric checks and the same ordered80-view CAL identity before its fixed grid/search. The global freezer checks all three, exact raw/growth search completeness/rank, FP≤1/70 and pixel precision≥.65 guards, and one shared BN initialization lineage. Root alone can audit the single globally selected winner on the exposed comparison.

## Preparation checks and execution status

Eight new pure fixtures cover exact64 selection/source coverage, unavailable-source failure, forbidden state changes, counters/epsilon/momentum/variance failures, actual factory metadata, no-optimizer BN fitting, exact fitted-state loading before optimizer, and explicit unchanged GPU recipe. Nine reused pure fixtures cover FP16-safe bounded math, normal-source split identity, sampler coverage, full search/rank validation and score semantics. These fixtures execute no Torch model, image decode or inference.

The immutable protocol, preparation receipt and independent static review are recorded before any L4 forward. Concrete execution commands and actual receipts will be appended after root review/release; this document itself does not authorize a pass, fit or deployment. A failure stops this fixed procedure without changing pass count, epsilon, score grid or numeric tolerance.

The conditional single CPU-FP32/4-thread fallback remains proposal only, in `factory-stfpm-bn-preconditioning-proposal.md`; it has no runtime hook or forward here.

## Frozen preparation receipts, before any forward

Independent static review found no material blocker. The final pure fixture receipt records **17 passing checks in0.398 seconds**, zero image decodes, zero weight loads and no NN execution. The selected source SHA checks/module metadata inspection are separately disclosed above.

| Artifact under `output/optimization/factory-round3l4-bn64` | SHA-256 |
|---|---|
| `protocol-before-BN-fitting.json` | `6ff093a5c51dc8e799bf316bf80e0a6799e6cf78210b1a7316ea7087af63ac6c` |
| `preparation-fixtures.json` | `5be256d84089551e81cfef3211ef62f1363d05b9526cdf25b6f1b149635d437b` |
| `actual-BN-factory-metadata.json` | `90ebdc248526bcc8e478ae1467ae83fe1b487899ac9e47dee3c153678cdfaf0d` |
| `selected64-before-any-forward.json` | `d31dfcbf737e3d486a973d41f7e93cb9a396336cdca8c12773dd3840d79b3fa0` |
| `pre-forward-source-snapshot/snapshot-manifest.json` | `17243b75cca34ebf15da4dae56f80d39883da06eb34a82d75e98710f9e7a9c6b` |

The fixed new source prefix is `spatial_round3l4_stfpm_bn64`. No old executed L1/L2/L3 source, runtime hook, model or receipt changed. Actual forward execution remains held until root reviews the above protocol and releases the stage. The staged commands are:

```powershell
# CPU1 real TRAIN fitting, exactly64 student-only forwards; separate root release required.
.venv-roads-export\Scripts\python.exe -X utf8 scripts/prepare_factory_stfpm_bn.py --stage fit-bn --root-released
# Exact saved BN state -> distinct ONNX and corresponding Torch references.
.venv-roads-export\Scripts\python.exe -X utf8 scripts/prepare_factory_stfpm_bn.py --stage export-preflight --root-released
# GPU after construction hands off; stop on the first numeric failure.
.venv\Scripts\python.exe -X utf8 scripts/probe_factory_stfpm_bn.py --parity-step 0 --root-released
# Only after numeric success AND root's quiet performance release.
.venv\Scripts\python.exe -X utf8 scripts/probe_factory_stfpm_bn.py --preflight --root-released
```

The `--root-released` flags document coordination; their presence is not permission to execute a held stage. A600-step fit remains separately gated and unexecuted.

## Authorized real-TRAIN BN fit and export: passed

After root reviewed the frozen protocol, the **single fixed64-view BN fitting pass completed with exit0**. All18 BN counters equal64, all running statistics are finite with nonnegative variances, every epsilon remains.001 and momentum is restored to.01. Exactly54 permitted state entries changed:18 running means,18 running variances and18 counters. Every student parameter, complete teacher state and input-normalization buffer remains identical. There was no optimizer and zero gradient updates. The complete teacher state digest remains `64ceb1279f4bf4bd460bb5fb3738fdb52795dff8d7b3fcda623a55e8be712feb`.

The saved `real-TRAIN-BN64-fitted-state.pt` SHA is `3a8030a7e50d0e2d1c7795bc6cad1139d5d9427aa6020ca09347cbaf096a4207`; `BN-fitting-complete.json` SHA is `dba5c934745c030417006d58927f531112e0250ccd0c31fe228bc553e02bcf00`. This is genuine normal-TRAIN-based fitting, not an unfitted model or mere warmup. The construction agent's bounded TRAIN parity diagnosis could overlap; no capacity measurement is inferred from these job durations.

The separately authorized export reconstructed and strictly loaded this exact full state, passed the complete saved-state hash comparison and regenerated all three same-input Torch references. Export completed with exit0. The new ONNX `models/factory/spatial_round3l4_stfpm_bn64-step0.onnx` SHA is `3eb5f1cd0936342098658e4a808176e4b6bdb8bc1e2103dc3873e7542ed07ace`, sidecar SHA `6fd16a030863726ef7f0728c0df3927d1e18f5a85aec36767fd8b5eedf5bfb6b`, and reference NPZ SHA `5e870bccd9243f1835f2bc2d68e666110bfe4d4903ddcbc605bd577b19e08409`. The fixed real TRAIN Torch map spans.2843051–.6744165, so this exported case is nonconstant. It is not an accuracy measurement.

At this stage, actual GPU numeric parity waits for construction's explicit GPU handoff. Capacity, the600-update optimizer fit, all-three CAL, exposed comparison and deployment remain unexecuted/unproven for L4. No earlier receipt, existing model or default was replaced.

## Actual GPU numeric check: failed; procedure stopped

After the construction agent's bounded TRAIN diagnosis exited, it explicitly handed off the GPU. The one approved L4 three-case check ran on the new BN-fitted graph and **failed with exit1**. All scores were finite, bounded and nonconstant, but that is insufficient: all three maximum absolute errors exceed the unchanged.01 acceptance limit.

| Fixed numeric input | Maximum absolute GPU–Torch score error | Actual GPU map range |
|---|---:|---:|
| First fixed real normal TRAIN source | .0895002484 | .3178711–.7226563 |
| Zero initialization tensor | .0430544913 | .3496094–.5332031 |
| All255 initialization tensor | .0496118367 | .3215332–.5161133 |

The actual queried compiled properties report deviceGPU.0, inference precisionfloat16 and `GPU_DISABLE_WINOGRAD_CONVOLUTION=True`; the runtime node inventory contains114 FP16,2 FP32 and1 uint8 nodes, with no Winograd-named execution node. This records queried properties rather than relying only on requested metadata. It does not prove the underlying cause of the mismatch.

The complete retained numeric report is `output/optimization/factory-round3l4-bn64/spatial_round3l4_stfpm_bn64-step0-gpu-parity.json`, SHA `2349463e4c38a29e7d47a71fd29ce6af7fc5b7d6517484655b697c08273cbc9b`. It binds model, initial sidecar, export receipt, all three references, exact64 fitted state/receipt and frozen protocol. The regenerated Torch references correspond to the intentionally changed BN state; they are not the old unconditioned model's targets.

**No64-call capacity run,600-step optimizer fit, CAL selection or exposed-comparison inference followed.** No pass count, precision flag, epsilon or tolerance was changed, and no failed receipt was overwritten or retried. BN preconditioning passed its state-integrity checks but did not solve the GPU numerical-agreement gate. The saved real-TRAIN-fitted model and all earlier failures remain available as evidence, with no deployment/default change. All own compute exited and GPU ownership returned to root.

The previously described single CPU-FP32/4-thread full-profile path remains an unexecuted proposal requiring a distinct root decision/protocol. CPU agreement on the earlier unconditioned model cannot substitute for new BN-fitted numeric agreement or15FPS capacity.
