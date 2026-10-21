"""Fail-closed C200/source/new-bank/export validation; no NN or pixel decode."""
from pathlib import Path
import hashlib
import json
import numpy as np
from factory_c_density_common import verify_protocol

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/optimization/factory-round3'
NAME = 'spatial_round3k_c200_density_f16'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_density_artifacts(calibration_row=None, name=NAME):
    NAME=name
    protocol_path,protocol=verify_protocol()
    if NAME!=protocol['name']:raise ValueError('Separate C-density prefix required')
    initial_path = OUT / (NAME + '-export-manifest.json')
    initial = json.loads(initial_path.read_text(encoding='utf-8'))
    source = Path(protocol['warmstart'])
    onnx = ROOT / 'models/factory/spatial_round3e_phase_max_refine-step200.onnx'
    source_hashes = {'source_pt_sha256': sha(source), 'source_onnx_sha256': sha(onnx),
                     'source_metadata_sha256': sha(onnx.with_suffix('.onnx.json'))}
    if source_hashes['source_pt_sha256'] != protocol['warmstart_sha256']:
        raise ValueError('Protected source PT changed')
    for key in ('source_onnx_sha256', 'source_metadata_sha256'):
        if source_hashes[key] != protocol[key]:
            raise ValueError('Protected source ONNX/metadata changed')
    if initial['protocol_sha256'] != sha(protocol_path) or any(initial[k] != v for k, v in source_hashes.items()):
        raise ValueError('Initial export identity differs from source/protocol')
    for suffix, key in [('capacity-probe', 'probe_sha256'), ('normal-feature-extractor', 'feature_extractor_sha256'),
                        ('pre-fit-control', 'pre_fit_control_sha256'), ('pre-fit-diagnostic', 'pre_fit_diagnostic_sha256')]:
        if sha(ROOT / 'models/factory' / (NAME + '-' + suffix + '.onnx')) != initial[key]:
            raise ValueError('Initial graph/extractor changed')
    fit_path = OUT / (NAME + '-normal-fit.json')
    fit = json.loads(fit_path.read_text(encoding='utf-8'))
    bank_path = OUT / (NAME + '-normal-bank.npz')
    if (fit['protocol_sha256'] != sha(protocol_path) or fit['bank_sha256'] != sha(bank_path)
            or fit['export_manifest_sha256'] != sha(initial_path)
            or fit['source_pt_sha256'] != source_hashes['source_pt_sha256']
            or fit['feature_extractor_sha256'] != initial['feature_extractor_sha256']):
        raise ValueError('Normal fit/bank/extractor lineage changed')
    rows = fit['normal_sources']
    if rows!=protocol['normal_sources'] or fit['normal_source_identity_sha256']!=protocol['normal_source_identity_sha256']:
        raise ValueError('C normal-fit sources differ from exact protected208 identities')
    if initial['feature_extractor_sha256']!=initial['pre_fit_diagnostic_sha256']:
        raise ValueError('Pruned or foreign normal-feature context is forbidden')
    preflight=json.loads((OUT/(NAME+'-fullprofile-preflight.json')).read_text())
    source_parity=json.loads((OUT/(NAME+'-pre-fit-gpu-parity.json')).read_text())
    if preflight['protocol_sha256']!=sha(protocol_path) or preflight['model_sha256']!=initial['probe_sha256'] or preflight['profile_ms']['serial_calls_per_second']<15:
        raise ValueError('Fresh C-specific preflight has not passed or changed')
    if not source_parity['all_checks_passed'] or source_parity['protocol_sha256']!=sha(protocol_path):
        raise ValueError('Fresh C source/control parity has not passed')
    if (len(rows) != 208 or len({row['image'] for row in rows}) != 208
            or any(row['defective'] or Path(row['image']).parts[0] not in protocol['split_items']['train'] for row in rows)
            or hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest() != fit['normal_source_identity_sha256']):
        raise ValueError('Protected normal-source manifest changed')
    with np.load(bank_path) as bank:
        if (bank['mean'].shape != (16, 256) or bank['whitening'].shape != (16, 16)
                or not all(np.isfinite(bank[key]).all() for key in ('mean', 'whitening', 'q50', 'q95'))
                or float(bank['q95']) < float(bank['q50'])
                or np.abs(bank['whitening']).sum(axis=1).max() > 8.001):
            raise ValueError('Normal-bank numeric geometry changed')
        if protocol.get('numeric_variant'):
            if 'qspan' not in bank or not np.isfinite(bank['qspan']).all() or float(bank['qspan'])<=0:
                raise ValueError('F2 must read explicit positive quantized qspan')
            if any(not np.array_equal(bank[key],bank[key].astype(np.float16).astype(bank[key].dtype)) for key in ('mean','whitening','q95','qspan')):
                raise ValueError('F2 stored constants are not exactly half-representable')
    exports_path = OUT / (NAME + '-fitted-export-manifest.json')
    exports = json.loads(exports_path.read_text(encoding='utf-8'))
    if (exports['protocol_sha256'] != sha(protocol_path) or exports['normal_bank_sha256'] != sha(bank_path)
            or exports['normal_fit_report_sha256'] != sha(fit_path)
            or exports['initial_export_manifest_sha256'] != sha(initial_path)
            or any(exports[k] != v for k, v in source_hashes.items())):
        raise ValueError('Fitted export source/bank/fit identity changed')
    if sha(ROOT / 'models/factory' / (NAME + '-offline-diagnostic.onnx')) != exports['diagnostic_sha256']:
        raise ValueError('Fitted diagnostic graph changed')
    expected = protocol['candidates']
    actual = exports['candidates']
    if len(expected) != 9 or len(actual) != 9 or protocol['save_at'] != list(range(9)):
        raise ValueError('Exactly nine fixed fusion candidates are required')
    for declared, artifact in zip(expected, actual):
        if {key: artifact[key] for key in ('id', 'alpha', 'beta')} != declared:
            raise ValueError('Candidate order/ID/alpha/beta declaration changed')
        standard = ROOT / 'models/factory' / f'{NAME}-step{declared["id"]}.onnx'
        if Path(artifact['model_path']).resolve() != standard.resolve() or sha(standard) != artifact['model_sha256']:
            raise ValueError('Candidate standard path or actual model identity changed')
    if calibration_row is not None:
        artifact = next((a for a in actual if a['id'] == calibration_row['step']), None)
        if (artifact is None or Path(calibration_row['checkpoint']).resolve() != Path(artifact['model_path']).resolve()
                or calibration_row['model_sha256'] != artifact['model_sha256']):
            raise ValueError('Calibration report is not tied to its declared exported candidate')
    return protocol, fit, exports
