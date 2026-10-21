"""Bounded F arithmetic diagnosis: fixed TRAIN input and zero, immutable models."""
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
from check_factory_density_gpu_parity import compile_graph, run
from factory_density_math import normalized_log_distance
from factory_density_identity import validate_density_artifacts

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/optimization/factory-round3'
NAME = 'spatial_round3h_normal_density'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def errors(a, b):
    delta = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))
    finite = delta[np.isfinite(delta)]
    return {'max': float(finite.max()), 'p50': float(np.percentile(finite, 50)),
            'p95': float(np.percentile(finite, 95)), 'p99': float(np.percentile(finite, 99)),
            'p99_9': float(np.percentile(finite, 99.9)), 'nonfinite': int(delta.size - finite.size)}


def numpy_evidence(fine, bank, rounded_parameters=False):
    mean, white = bank['mean'], bank['whitening']
    q95 = float(bank['q95'])
    span = max(.001, q95 - float(bank['q50']))
    if rounded_parameters:
        mean, white = mean.astype(np.float16).astype(np.float32), white.astype(np.float16).astype(np.float32)
        q95, span = float(np.float16(q95)), float(np.float16(span))
    logd = normalized_log_distance(fine, mean, white)
    return np.clip((logd - q95) / span, -8., 8.)


def main():
    cv2.setNumThreads(1)
    protocol, fit, exports = validate_density_artifacts()
    with np.load(OUT / (NAME + '-normal-bank.npz')) as data:
        bank = {key: data[key] for key in data.files}
    modeldir = ROOT / 'models/factory'
    graphs = {'standalone_features': modeldir / (NAME + '-normal-feature-extractor.onnx'),
              'pre_fit_full_diagnostic': modeldir / (NAME + '-pre-fit-diagnostic.onnx'),
              'fitted_full_diagnostic': modeldir / (NAME + '-offline-diagnostic.onnx')}
    compiled = {key: compile_graph(path) for key, path in graphs.items()}
    first = next(iter(sorted((ROOT / 'datasets/factory/KolektorSDD' / protocol['split_items']['train'][0]).glob('*.jpg'))))
    frame = cv2.imread(str(first))
    value = np.ascontiguousarray(cv2.resize(frame, (512, 1280), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)[None])
    reports = []
    for label, input_value in [('fixed_train', value), ('zero_initialization', np.zeros_like(value))]:
        standalone = run(compiled['standalone_features'], input_value)[0][0]
        pre = run(compiled['pre_fit_full_diagnostic'], input_value)
        fitted = run(compiled['fitted_full_diagnostic'], input_value)
        actual = fitted[1][0, 0]
        fine_pre, fine_fit = pre[2][0], fitted[2][0]
        reports.append({'input': label,
                        'head_error_using_its_own_actual_fine': errors(actual, numpy_evidence(fine_fit, bank)),
                        'head_error_own_fine_rounded_constant_parameters': errors(actual, numpy_evidence(fine_fit, bank, True)),
                        'head_error_using_standalone_fine': errors(actual, numpy_evidence(standalone, bank)),
                        'head_error_using_pre_fit_full_fine': errors(actual, numpy_evidence(fine_pre, bank)),
                        'pre_fit_full_vs_fitted_full_fine': errors(fine_pre, fine_fit),
                        'standalone_vs_pre_fit_full_fine': errors(standalone, fine_pre),
                        'standalone_vs_fitted_full_fine': errors(standalone, fine_fit),
                        'finite_counts': {key: {'finite': int(np.isfinite(a).sum()), 'total': int(a.size)}
                                          for key, a in [('standalone', standalone), ('pre_fine', fine_pre), ('fitted_fine', fine_fit), ('actual_evidence', actual)]}})
    identity_after = validate_density_artifacts()
    report = {'scope': 'Authorized fixed TRAIN/zero numeric diagnosis only, no labels/CAL/original; no refit or parameter changes',
              'protocol_sha256': sha(OUT / (NAME + '-protocol-before-fit.json')),
              'bank_sha256': fit['bank_sha256'], 'fitted_export_manifest_sha256': sha(OUT / (NAME + '-fitted-export-manifest.json')),
              'graph_sha256': {key: sha(path) for key, path in graphs.items()},
              'compiler_properties': {key: {'execution_devices': list(graph.get_property('EXECUTION_DEVICES')),
                                             'inference_precision_hint': str(graph.get_property('INFERENCE_PRECISION_HINT')),
                                             'num_streams': str(graph.get_property('NUM_STREAMS'))}
                                     for key, graph in compiled.items()},
              'fixed_train_image': str(first.relative_to(ROOT / 'datasets/factory/KolektorSDD')), 'fixed_train_image_sha256': sha(first),
              'observed_workload': 'Concurrent construction CPU4 fitting and ordinary host services; no speed qualification',
              'results': reports, 'artifacts_unchanged': identity_after[1]['bank_sha256'] == fit['bank_sha256']}
    target = OUT / (NAME + '-numeric-diagnosis.json')
    if target.exists():
        raise ValueError('Preserve the first fixed diagnostic report')
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
