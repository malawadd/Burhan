"""Fixed TRAIN-only actual GPU control/feature/density parity; no GT read."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import openvino as ov
from edge_runtime.profiles.factory_encoder import seeded_growth_mask
from factory_c_density_math import normalized_log_distance
from factory_c_density_identity import validate_density_artifacts
from factory_c_density_common import verify_protocol


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_graph(path):
    core = ov.Core()
    return core.compile_model(core.read_model(str(path)), 'GPU', {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1', 'INFERENCE_PRECISION_HINT': 'f16'})


def run(compiled, value):
    result = compiled.create_infer_request().infer({compiled.input(): value})
    return [np.array(result[out], dtype=np.float32, copy=True) for out in compiled.outputs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fitted', action='store_true')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    out = ROOT / 'output/optimization/factory-round3'
    name = 'spatial_round3k_c200_density_f16'
    modeldir = ROOT / 'models/factory'
    protocol_path,protocol=verify_protocol()
    target = out / (name + ('-fitted-gpu-parity.json' if args.fitted else '-pre-fit-gpu-parity.json'))
    if target.exists():raise ValueError('Preserve failed/passed C parity artifact; retry requires a new authorized version')
    manifest = json.loads((out / (name + '-export-manifest.json')).read_text(encoding='utf-8'))
    if manifest['protocol_sha256'] != sha(protocol_path):
        raise ValueError('Source export protocol changed')
    source = modeldir / 'spatial_round3e_phase_max_refine-step200.onnx'
    control = modeldir / (name + ('-step0.onnx' if args.fitted else '-pre-fit-control.onnx'))
    features_path = modeldir / (name + '-normal-feature-extractor.onnx')
    diagnostic_path = modeldir / (name + ('-offline-diagnostic.onnx' if args.fitted else '-pre-fit-diagnostic.onnx'))
    if sha(source) != protocol['source_onnx_sha256'] or sha(features_path) != manifest['feature_extractor_sha256']:
        raise ValueError('Source/extractor changed')
    if not args.fitted and (sha(control) != manifest['pre_fit_control_sha256'] or sha(diagnostic_path) != manifest['pre_fit_diagnostic_sha256']):
        raise ValueError('Pre-fit GPU parity graphs changed')
    fitted_manifest_path = out / (name + '-fitted-export-manifest.json')
    if args.fitted:
        validate_density_artifacts(name=name)
        fitted_manifest = json.loads(fitted_manifest_path.read_text(encoding='utf-8'))
        if (fitted_manifest['protocol_sha256'] != sha(protocol_path)
                or fitted_manifest['normal_bank_sha256'] != sha(out / (name + '-normal-bank.npz'))
                or fitted_manifest['diagnostic_sha256'] != sha(diagnostic_path)):
            raise ValueError('Fitted bank/diagnostic/export provenance changed')
        for row in fitted_manifest['candidates']:
            if sha(Path(row['model_path'])) != row['model_sha256']:
                raise ValueError('A fitted candidate differs from its export identity')
    first = next(iter(sorted((ROOT / 'datasets/factory/KolektorSDD' / protocol['split_items']['train'][0]).glob('*.jpg'))))
    frame = cv2.imread(str(first))
    value = np.ascontiguousarray(cv2.resize(frame, (512, 1280), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)[None])
    source_raw = run(compile_graph(source), value)[0]
    control_raw = run(compile_graph(control), value)[0]
    raw_close = bool(np.allclose(source_raw, control_raw, atol=1e-3, rtol=1e-3))
    scores = [1. / (1. + np.exp(-np.clip(raw[0, 0], -40., 40.))) for raw in (source_raw, control_raw)]
    policy=protocol['source_policy']
    masks = [seeded_growth_mask(score, policy['high'], policy['area'], policy['growth']) for score in scores]
    mask_same = bool(np.array_equal(*masks))
    feature_outputs=run(compile_graph(features_path), value)
    fine=next(array[0] for array in feature_outputs if array.shape==(1,16,640,256))
    diagnostic = compile_graph(diagnostic_path)
    raw_diag, evidence_diag, fine_diag = run(diagnostic, value)
    zero_results = run(diagnostic, np.zeros_like(value))
    single_diag_close = True
    single_diag_error = None
    if args.fitted:
        # Diagnostic uses alpha1,beta4; candidate7 is its actual one-output graph.
        served_path = modeldir / (name + '-step7.onnx')
        served_raw = run(compile_graph(served_path), value)[0]
        single_diag_error = float(np.max(np.abs(served_raw - raw_diag)))
        single_diag_close = bool(np.allclose(served_raw, raw_diag, atol=1e-3, rtol=1e-3))
    if args.fitted:
        with np.load(out / (name + '-normal-bank.npz')) as values:
            bank = {key: values[key] for key in values.files}
    else:
        from factory_c_density_math import dummy_bank
        bank=dummy_bank()
    log_distance = normalized_log_distance(fine, bank['mean'], bank['whitening'])
    qspan=float(bank['qspan']) if 'qspan' in bank else max(.001, float(bank['q95']) - float(bank['q50']))
    expected = np.clip((log_distance - float(bank['q95'])) / qspan, -8., 8.)
    evidence_error = float(np.max(np.abs(evidence_diag[0, 0] - expected)))
    finite = all(np.isfinite(array).all() for array in [source_raw, control_raw, fine, raw_diag, evidence_diag, fine_diag, *zero_results])
    report = {'all_checks_passed': bool(raw_close and mask_same and finite and evidence_error <= .1 and single_diag_close),
              'protocol_sha256': sha(protocol_path), 'source_onnx_sha256': sha(source),
              'control_onnx_sha256': sha(control), 'diagnostic_onnx_sha256': sha(diagnostic_path),
              'fixed_train_input': str(first.relative_to(ROOT / 'datasets/factory/KolektorSDD')),
              'input_image_sha256': sha(first), 'raw_logits_allclose_atol_rtol_1e_3': raw_close,
              'source_policy_mask_identical': mask_same, 'raw_logits_max_abs_error': float(np.max(np.abs(source_raw - control_raw))),
              'source_raw_logit_min_max': [float(source_raw.min()), float(source_raw.max())],
              'source_score_min_max': [float(scores[0].min()), float(scores[0].max())],
              'standalone_vs_diagnostic_feature_max_error': float(np.max(np.abs(fine - fine_diag[0]))),
              'actual_gpu_vs_numpy_evidence_max_error': evidence_error, 'evidence_error_guard': .1,
              'single_output_vs_diagnostic_fused_logits_allclose': single_diag_close,
              'single_output_vs_diagnostic_max_error': single_diag_error,
              'diagnostic_actual_evidence_min_max': [float(evidence_diag.min()), float(evidence_diag.max())],
              'zero_input_evidence_min_max': [float(zero_results[1].min()), float(zero_results[1].max())],
              'all_outputs_finite': bool(finite), 'scope': 'One fixed TRAIN image plus zero initialization input; no mask labels, CAL/comparison or accuracy credit'}
    if args.fitted:
        report.update(normal_bank_sha256=sha(out / (name + '-normal-bank.npz')),
                      fitted_export_manifest_sha256=sha(fitted_manifest_path))
    target = out / (name + ('-fitted-gpu-parity.json' if args.fitted else '-pre-fit-gpu-parity.json'))
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if not report['all_checks_passed']:
        raise ValueError('GPU source/density parity guard failed; no fitting/calibration allowed')


if __name__ == '__main__':
    main()
