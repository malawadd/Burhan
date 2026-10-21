"""F: record protocol/export a shared-feature normal-density graph locally."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from train_factory_spatial_unet import SpatialUNet, export

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/optimization/factory-round3'
NAME = 'spatial_round3h_normal_density'
SOURCE = ROOT / 'models/factory/spatial_unet_v10_sdd2-step300.pt'
SOURCE_ONNX = ROOT / 'models/factory/spatial_unet_v10h_sdd2-step300.onnx'
CANDIDATES = [(1., 0.)] + [(alpha, beta) for alpha in (.5, 1.) for beta in (1., 2., 4., 8.)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol(precision_corrected=False):
    original = json.loads(SOURCE_ONNX.with_suffix('.onnx.json').read_text(encoding='utf-8'))
    value = {
        'name': NAME, 'seed': 20261026, 'architecture':
        'Frozen v10h SpatialUNet plus same-forward stride2/16ch column-conditioned normal-density Mahalanobis fusion; one raw-logit output',
        'warmstart': str(SOURCE), 'warmstart_sha256': sha(SOURCE),
        'source_onnx_sha256': sha(SOURCE_ONNX), 'source_metadata_sha256': sha(SOURCE_ONNX.with_suffix('.onnx.json')),
        'split_items': original['split_items'], 'input_shape': [1, 3, 1280, 512],
        'output_shape': [1, 1, 640, 256], 'fine_steps': 0,
        'save_at': list(range(len(CANDIDATES))), 'candidate_id_is_not_sgd_step': True,
        'candidates': [{'id': i, 'alpha': a, 'beta': b} for i, (a, b) in enumerate(CANDIDATES)],
        'normal_fit': 'All208 actual-normal views among239 original TRAIN views,30physicalitems; nativeGT empty; no SDD2 fit/cal/test mixture',
        'feature_fit': 'Frozen/eval original firstencoderblock16x640x256, actual OpenVINO GPU f16 features, no optimizer or BN updates',
        'moments': 'Float64 all-pixel column mean16x256 and global secondmoment16x16; residualcov=E[ffT]-mean_x(mu_x mu_xT)',
        'covariance': 'Symmetrize; .9cov+.1diag(max(diag(cov),0)); eigenvalues floored max(1e-6,.001*median nonnegative diagonal)',
        'numeric_guards': 'Globally rescale whitening max row-L1<=8; centered features and whitened activations clipped+-32; mean squares; log(max(D,1e-6)); normalize by trainq95/median span>=.001; final signed evidence clipped+-8',
        'normal_quantiles': 'Second fixed normal-only pass;2048 spatial samples per view without replacement, NumPy seed20261026;425984points; q50/q95 on scaled guarded logdistance',
        'fusion': 'Raw pre-sigmoid logit=alpha*original_rawlogit+beta*signed_normal_evidence; noOR, no full-frame fallback; one shared encoder forward and one score map',
        'selection_gate': 'NormalcalFP<=1/70 and aggregateforegroundprecision>=.65',
        'selection_rank': ['meaningfulIoU>=.1viewrecall', 'worst_positive_IoU', 'coverage50pct_viewrecall', 'worst_positive_GTcoverage', 'microIoU', 'fewerNormalFP', 'foregroundrecall'],
        'raw_high_grid': [.05, .1, .25, .5, .7, .9, .95, .98, .99, .995, .999, .9999, .99999, .999999],
        'area_grid': [1, 2, 4, 8, 16, 32], 'growth_grid': [None, .05, .1, .25, .5, .7, .8, .9, .95, .98, .99],
        'seed_search': 'Best normal-FP-eligible area per high by weakest-positive rank; top6distinct highs; final guard includes precision; heuristic not exhaustive',
        'calibration': 'All9 actual exported single-output GPU graphs calibrated; exact ordered80 source image/mask hashes and native geometry; no CPU-fusion scoring substitute',
        'control': 'Candidate0 alpha1 beta0 preserves original raw logits; frozen/eval train-mode-BN parity check before export; unchanged source model weights',
        'parity_gate': 'Before calibration, actual candidate0 GPU raw-logit source parity on fixed TRAIN input (atol/rtol1e-3) and source-policy mask identity; fitted diagnostic head versus NumPy/effective feature extractor error recorded, max evidence error<=.1; all outputs finite including zero input',
        'clipping_diagnostics': 'Fixed guards are not tuned; count centered/whitened/A clipping fractions on TRAIN normals and calibration, with actual feature graph frozen before cal; diagnostics do not refit bank or change thresholds',
        'comparison_gate': 'All9cal then globalfreeze; if control wins or no strict rank improvement over eligible control, reject without new originalaudit',
        'capacity_gate': 'Nonconstant dense-whitening fullshape completeprofile TRAIN preflight>=15 before normal-stat fitting; final fitted graph project-quiescent paired capacity/runtime>=15 mandatory before promotion',
        'hypothesis_scope': 'Iterative exposed-development strategy, not blind-test evidence; no originalcomparison pixels, IDs or ROIs used for fit/mining/calibration/inference rules',
        'official_sdd2_test_still_sealed': True, 'comparison_images_used_for_fit_selection': 0}
    if precision_corrected:
        prior='spatial_round3h_normal_density'
        capacity=OUT/(prior+'-fullprofile-preflight.json')
        parity=OUT/(prior+'-pre-fit-gpu-parity.json')
        value.update(numeric_variant='F2 exact representable constants and full compiler-context normal features',
            feature_fit='Frozen source firstblock16x640x256 emitted by pinned FULL semantic+densehead pre-fit diagnostic with all3outputs retained; no pruned fine-only extraction',
            moments='Float64 full-context moments mu64/EffT; round deployed muq to f16; covariance=EffT-mean_x(mu64 muqT+muq mu64T-muq muqT)',
            parameter_representation='mu nearestf16; W magnitudes rounded towardzero f16 after same shrink/floor/globalrowL1<=8; normalquantiles useexactstored muq/Wq; q95 and explicitqspan=max(.001,rawq95-rawq50) nearestf16; graph/reference read storedqspan directly, not roundedq50 difference',
            normal_quantiles='Same second fixed208normal-onlypass/2048pixelsperview/seed20261026, with exactstoredmuq/Wq; rawq50/q95 kept, then quantizeq95 and independentqspan; no calibration feedback',
            reused_capacity_receipt=str(capacity),reused_capacity_receipt_sha256=sha(capacity),
            reused_source_parity_receipt=str(parity),reused_source_parity_receipt_sha256=sha(parity),
            preserved_failed_F_fit_sha256=sha(OUT/(prior+'-normal-fit.json')),
            preserved_failed_F_parity_sha256=sha(OUT/(prior+'-fitted-gpu-parity.json')),
            full_context_feature_source=str(ROOT/'models/factory'/(prior+'-pre-fit-diagnostic.onnx')),
            full_context_feature_source_sha256=sha(ROOT/'models/factory'/(prior+'-pre-fit-diagnostic.onnx')),
            capacity_gate='Reuse immutable F nonconstant23.48/s receipt as identical-topology architecture gate only; no newF2speedclaim; final actualfittedcandidate pairedprojectquiescent capacity/runtime>=15 mandatory')
    return value


class NormalDensityGraph(nn.Module):
    def __init__(self, bank=None, alpha=1., beta=4.):
        super().__init__()
        self.base = SpatialUNet()
        self.base.load_state_dict(torch.load(SOURCE, map_location='cpu', weights_only=True), strict=True)
        if bank is None:
            rng = np.random.default_rng(20261026)
            matrix = rng.normal(0, .15, (16, 16)).astype(np.float32) + np.eye(16, dtype=np.float32)
            matrix /= max(1., float(np.abs(matrix).sum(axis=1).max()) / 8.)
            mean = np.linspace(-.1, .1, 256, dtype=np.float32)[None].repeat(16, axis=0)
            bank = {'mean': mean, 'whitening': matrix, 'q50': -.5, 'q95': .5}
        self.register_buffer('normal_mean', torch.tensor(np.asarray(bank['mean']), dtype=torch.float32)[None, :, None, :])
        self.register_buffer('whitening', torch.tensor(np.asarray(bank['whitening']), dtype=torch.float32)[:, :, None, None])
        self.register_buffer('q95', torch.tensor(float(bank['q95']), dtype=torch.float32))
        qspan=float(bank['qspan']) if 'qspan' in bank else max(.001, float(bank['q95']) - float(bank['q50']))
        self.register_buffer('qspan', torch.tensor(qspan, dtype=torch.float32))
        self.alpha, self.beta = float(alpha), float(beta)
        if self.normal_mean.shape != (1, 16, 1, 256) or self.whitening.shape != (16, 16, 1, 1):
            raise ValueError('Normal-bank geometry changed')
        if not all(torch.isfinite(value).all() for value in (self.normal_mean, self.whitening, self.q95, self.qspan)):
            raise ValueError('Normal bank must be finite')
        if float(bank['q95']) < float(bank['q50']) or float(torch.abs(self.whitening).sum(dim=1).max()) > 8.001:
            raise ValueError('Invalid ordered quantiles or bounded whitening')
        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()

    def parts(self, bgr):
        base = self.base
        x = (bgr.to(torch.float32)[:, [2, 1, 0]] / 255. - base.mean) / base.std
        fine = middle = None
        for index, block in enumerate(base.encoder):
            x = block(x)
            if index == 0:
                fine = x
            elif index == 3:
                middle = x
        x = F.interpolate(base.coarse(x), size=middle.shape[-2:], mode='bilinear', align_corners=False)
        x = base.middle_spatial(F.relu(x + base.middle(middle)))
        x = F.interpolate(x, size=fine.shape[-2:], mode='bilinear', align_corners=False)
        logits = base.classifier(base.fine_spatial(F.relu(x + base.fine(fine))))
        return logits, fine

    def evidence(self, fine):
        centered = torch.clamp(fine - self.normal_mean, -32., 32.)
        white = torch.clamp(F.conv2d(centered, self.whitening), -32., 32.)
        distance = torch.mean(white * white, dim=1, keepdim=True)
        return torch.clamp((torch.log(torch.clamp(distance, min=1e-6)) - self.q95) / self.qspan, -8., 8.)

    def forward(self, bgr):
        logits, fine = self.parts(bgr)
        if self.beta == 0.:
            return logits * self.alpha
        return self.alpha * logits + self.beta * self.evidence(fine)


class NormalFeatureGraph(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.base = source.base

    def forward(self, bgr):
        x = (bgr.to(torch.float32)[:, [2, 1, 0]] / 255. - self.base.mean) / self.base.std
        return self.base.encoder[0](x)


class DiagnosticDensityGraph(NormalDensityGraph):
    def forward(self, bgr):
        logits, fine = self.parts(bgr)
        evidence = self.evidence(fine)
        return self.alpha * logits + self.beta * evidence, evidence, fine


def main():
    global NAME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record-protocol', action='store_true')
    parser.add_argument('--export-probe', action='store_true')
    parser.add_argument('--export-fitted', action='store_true')
    parser.add_argument('--precision-corrected', action='store_true')
    parser.add_argument('--prepare-precision-variant', action='store_true')
    args = parser.parse_args()
    if args.precision_corrected:
        NAME='spatial_round3i_normal_density_f16'
    value = protocol(args.precision_corrected)
    path = OUT / (NAME + '-protocol-before-fit.json')
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != value:
        raise ValueError('Protected density protocol differs')
    if not path.exists():
        path.write_text(json.dumps(value, indent=2), encoding='utf-8')
    if args.record_protocol:
        print(json.dumps({'protocol': str(path), 'sha256': sha(path)}, indent=2))
        return
    if args.prepare_precision_variant:
        if not args.precision_corrected:raise ValueError('Explicit F2 variant required')
        prior='spatial_round3h_normal_density'
        receipt=json.loads(Path(value['reused_capacity_receipt']).read_text(encoding='utf-8'))
        if receipt['profile_ms']['serial_calls_per_second']<15:raise ValueError('Reused architecture gate failed')
        pairs=[('capacity-probe','capacity-probe'),('pre-fit-control','pre-fit-control'),
               ('pre-fit-diagnostic','pre-fit-diagnostic'),('pre-fit-diagnostic','normal-feature-extractor')]
        for old_suffix,new_suffix in pairs:
            original=ROOT/'models/factory'/(prior+'-'+old_suffix+'.onnx')
            target=ROOT/'models/factory'/(NAME+'-'+new_suffix+'.onnx')
            if target.exists() and sha(target)!=sha(original):raise ValueError('Prepared F2 copy differs')
            if not target.exists():shutil.copyfile(original,target)
        manifest={'protocol_sha256':sha(path),'probe_sha256':sha(ROOT/'models/factory'/(NAME+'-capacity-probe.onnx')),
                  'feature_extractor_sha256':sha(ROOT/'models/factory'/(NAME+'-normal-feature-extractor.onnx')),
                  'pre_fit_control_sha256':sha(ROOT/'models/factory'/(NAME+'-pre-fit-control.onnx')),
                  'pre_fit_diagnostic_sha256':sha(ROOT/'models/factory'/(NAME+'-pre-fit-diagnostic.onnx')),
                  'source_pt_sha256':sha(SOURCE),'source_onnx_sha256':sha(SOURCE_ONNX),
                  'source_metadata_sha256':sha(SOURCE_ONNX.with_suffix('.onnx.json')),
                  'scope':'Byte-identical pinned full-context/reused architecture graphs; no newNN export or performance measurement'}
        manifest_path=OUT/(NAME+'-export-manifest.json')
        if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8'))!=manifest:raise ValueError('Protected F2 manifest differs')
        if not manifest_path.exists():manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        print(json.dumps(manifest,indent=2));return
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.export_probe:
        if args.precision_corrected:raise ValueError('F2 explicitly reuses immutable F architecture receipt')
        model = NormalDensityGraph()
        torch.manual_seed(20261026)
        sample = torch.randint(0, 256, (1, 3, 1280, 512), dtype=torch.uint8)
        with torch.inference_mode():
            original = model.base(sample)
            copied, _ = model.parts(sample)
            assert torch.equal(original, copied), 'Copied original raw-logit path changed'
            # All BN remains frozen/eval for any allowed no-gradient fit mode.
            model.train()
            for layer in model.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eval()
            fixed_train, _ = model.parts(sample)
            assert torch.equal(copied, fixed_train), 'Frozen BN train/eval path differs'
            model.eval()
            fused = model(sample)
            assert fused.shape == (1, 1, 640, 256) and torch.isfinite(fused).all()
            parity = {'raw_logit_torch_identity': True, 'frozen_bn_train_eval_identity': True,
                      'probe_fused_raw_logit_min_max': [float(fused.min()), float(fused.max())],
                      'seed': 20261026, 'scope': 'Nonconstant initialization tensor; no accuracy credit'}
        (OUT / (NAME + '-export-parity.json')).write_text(json.dumps(parity, indent=2), encoding='utf-8')
        probe_path = ROOT / 'models/factory' / (NAME + '-capacity-probe.onnx')
        features_path = ROOT / 'models/factory' / (NAME + '-normal-feature-extractor.onnx')
        export(model, probe_path)
        export(NormalFeatureGraph(model).eval(), features_path)
        control_path = ROOT / 'models/factory' / (NAME + '-pre-fit-control.onnx')
        diagnostic_path = ROOT / 'models/factory' / (NAME + '-pre-fit-diagnostic.onnx')
        export(NormalDensityGraph(alpha=1., beta=0.), control_path)
        diagnostic = DiagnosticDensityGraph().eval()
        with torch.inference_mode():
            torch.onnx.export(diagnostic, torch.zeros((1, 3, 1280, 512), dtype=torch.uint8), str(diagnostic_path),
                              input_names=['bgr'], output_names=['fused_logits', 'evidence', 'fine_features'],
                              opset_version=17, dynamo=False, do_constant_folding=True)
        manifest = {'protocol_sha256': sha(path), 'probe_sha256': sha(probe_path),
                    'feature_extractor_sha256': sha(features_path),
                    'pre_fit_control_sha256': sha(control_path), 'pre_fit_diagnostic_sha256': sha(diagnostic_path),
                    'source_pt_sha256': sha(SOURCE), 'source_onnx_sha256': sha(SOURCE_ONNX),
                    'source_metadata_sha256': sha(SOURCE_ONNX.with_suffix('.onnx.json')),
                    'scope': 'Actual exported files bound to verified immutable source before normal-stat fitting'}
        (OUT / (NAME + '-export-manifest.json')).write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(json.dumps(parity, indent=2))
    elif args.export_fitted:
        preflight_path=Path(value['reused_capacity_receipt']) if args.precision_corrected else OUT/(NAME+'-fullprofile-preflight.json')
        if args.precision_corrected and sha(preflight_path)!=value['reused_capacity_receipt_sha256']:raise ValueError('Reused capacity receipt drift')
        preflight = json.loads(preflight_path.read_text(encoding='utf-8'))
        if preflight['profile_ms']['serial_calls_per_second'] < 15.:
            raise ValueError('Density preflight failed capacity gate')
        bank_path = OUT / (NAME + '-normal-bank.npz')
        fit = json.loads((OUT / (NAME + '-normal-fit.json')).read_text(encoding='utf-8'))
        if fit['bank_sha256'] != sha(bank_path) or fit['protocol_sha256'] != sha(path):
            raise ValueError('Fitted bank/protocol drift')
        manifest = json.loads((OUT / (NAME + '-export-manifest.json')).read_text(encoding='utf-8'))
        if fit['export_manifest_sha256'] != sha(OUT / (NAME + '-export-manifest.json')) or manifest['source_pt_sha256'] != sha(SOURCE):
            raise ValueError('Fitted export/source provenance drift')
        with np.load(bank_path) as bank:
            fitted_exports = []
            for row in value['candidates']:
                model = NormalDensityGraph(bank, row['alpha'], row['beta'])
                target = ROOT / 'models/factory' / f'{NAME}-step{row["id"]}.onnx'
                export(model, target)
                fitted_exports.append({**row, 'model_path': str(target), 'model_sha256': sha(target)})
                print(json.dumps({'candidate': row, 'model': str(target), 'sha256': sha(target)}), flush=True)
            diagnostic = DiagnosticDensityGraph(bank)
            diagnostic.eval()
            diagnostic_path = ROOT / 'models/factory' / (NAME + '-offline-diagnostic.onnx')
            with torch.inference_mode():
                torch.onnx.export(diagnostic, torch.zeros((1, 3, 1280, 512), dtype=torch.uint8),
                                  str(diagnostic_path),
                                  input_names=['bgr'], output_names=['fused_logits', 'evidence', 'fine_features'],
                                  opset_version=17, dynamo=False, do_constant_folding=True)
        fitted_manifest = {'protocol_sha256': sha(path), 'normal_bank_sha256': sha(bank_path),
                           'normal_fit_report_sha256': sha(OUT / (NAME + '-normal-fit.json')),
                           'source_pt_sha256': sha(SOURCE), 'source_onnx_sha256': sha(SOURCE_ONNX),
                           'source_metadata_sha256': sha(SOURCE_ONNX.with_suffix('.onnx.json')),
                           'initial_export_manifest_sha256': sha(OUT / (NAME + '-export-manifest.json')),
                           'diagnostic_sha256': sha(diagnostic_path), 'candidates': fitted_exports}
        (OUT / (NAME + '-fitted-export-manifest.json')).write_text(json.dumps(fitted_manifest, indent=2), encoding='utf-8')
    else:
        raise ValueError('Choose an explicit protocol or export operation')


if __name__ == '__main__':
    main()
