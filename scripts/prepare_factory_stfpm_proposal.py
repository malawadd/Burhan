"""Record a metadata-only architecture/preflight proposal; never imports Torch/NN."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = 'spatial_round3l_stfpm_mobile_preflight'
OUT = ROOT / 'output/optimization/factory-round3'
SOURCE = OUT / 'spatial_round3k_c200_density_f16-protocol-before-fit.json'
SOURCE_SHA = '69433138b7a8a8c8d18ac68ae901487647d5e103db8330c2a7f8ece16bf0662a'
NORMAL_SHA = 'f7ec8222431065047de889ff2b03f24172bc59257d6962c51103377566f733b4'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def proposal():
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError('Established TRAIN metadata source changed')
    old = json.loads(SOURCE.read_text(encoding='utf-8'))
    normals = old['normal_sources']
    actual = hashlib.sha256(json.dumps(normals, sort_keys=True).encode()).hexdigest()
    if len(normals) != 208 or actual != NORMAL_SHA or any(r['defective'] for r in normals):
        raise ValueError('Only established208 GT-empty TRAIN identity metadata permitted')
    fit_items = set(old['split_items']['train'])
    if len(fit_items) != 30 or any(Path(r['image']).parts[0] not in fit_items for r in normals):
        raise ValueError('Original TRAIN item boundary changed')
    weights = ROOT / 'models/factory/encoder_sources/mobilenet_v3_small-047dcff4.pth'
    weight_sha = sha(weights)
    if not weight_sha.startswith('047dcff4'):
        raise ValueError('Existing official torchvision ImageNet teacher checkpoint changed')
    return {
        'name': NAME, 'stage': 'ARCHITECTURE/PREFLIGHT PROPOSAL ONLY',
        'implementation_ready': False,
        'authorization': 'No export/NN/preflight/fit/CAL authorized by this metadata record; root schedules exact next work separately',
        'source_method': 'STFPM-inspired MobileNet adaptation, NOT claimed reproduction of ResNet/STFPM paper',
        'sources': ['https://arxiv.org/abs/2103.04257', 'https://github.com/gdwang08/STFPM',
                    'https://raw.githubusercontent.com/open-edge-platform/anomalib/main/src/anomalib/models/image/stfpm/torch_model.py'],
        'teacher': {'file': str(weights.relative_to(ROOT)), 'sha256': weight_sha,
                    'state_load': 'strict weights_only=True; ImageNet teacher frozen/eval; no C/SDD fitted weights'},
        'student': {'initialization': 'separate random parameters, never equal teacher', 'seed': 20261030},
        'architecture': {'input_shape': [1, 3, 1280, 512], 'input_dtype': 'uint8 BGR',
                         'normalization': 'RGB float/255 then torchvision ImageNet mean/std',
                         'backbone': 'two independent MobileNetV3Small.features[:7]',
                         'feature_indices': [0, 3, 6], 'feature_channels': [16, 24, 40],
                         'output_shape': [1, 1, 640, 256],
                         'score': 'per-scale channel-normalized squared disagreement/4, bounded[0,1]; lower-scale maps bilinear to fine; exp(mean(log(clamp(map,min1e-6))))',
                         'norm_squared_floor': 1e-6, 'map_floor': 1e-6,
                         'fusion': 'fixed equal geometric mean only; no semantic C/density bank/OR/ROI/GT input'},
        'normal_identity_metadata_source_sha256': SOURCE_SHA,
        'normal_source_identity_sha256': actual, 'future_normal_source_count': len(normals),
        'normal_sources': normals,
        'preflight_sources': normals[:16],
        'preflight': {'warmup_calls': 4, 'measured_fullprofile_calls': 64,
                      'source_order': 'first16 pinned normal source rows; cycle4times, predecode before measurement',
                      'minimum_serial_calls_per_second': 15,
                      'source_policy': 'no graph skips/abstentions; retain every first-run tail/failure',
                      'device': 'actual installed OpenVINO GPU, f16 hint,1stream',
                      'parity_inputs': 'fixed first normal TRAIN input andzero only',
                      'maximum_absolute_map_error': .01,
                      'guards': 'all finite; nonconstant actual two-encoder graph; actual normalized-score/map geometry/source hash checks',
                      'timing_scope': 'new root-authorized project-quiescent slot; no transfer of C/F2/paper receipts; final fitted-model timing still mandatory'},
        'future_fit': 'NOT specified or authorized here; requires new protected exact optimizer/studentBN/sampling/checkpoint protocol BEFOREfirstupdate; only exact208 normals eligible',
        'future_evaluation': 'ALL declaredcheckpoints actualGPU maps/sourceGTpins beforeglobalfreeze; precision/FP guards and weakestpositive rank declaredbeforeCAL; rootONLYwinneroriginalaudit; no originalselection by this task',
        'forbidden': ['CAL/comparison fitting', 'original IDs/ROIs/masks/mining', 'officialSDD2testopening',
                      'reusing olddensity learnedbank/moments', 'synthetic damage inspection evidence',
                      'global256resize or alternative student selected after speed failure'],
        'calibration_images_opened_by_preparation': 0, 'images_decoded_by_preparation': 0,
        'model_weights_loaded_by_preparation': 0,
        'implementation_source_sha256': sha(Path(__file__)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record-proposal', action='store_true', required=True)
    parser.parse_args()
    value = proposal()
    path = OUT / (NAME + '-metadata-proposal-only.json')
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != value:
        raise ValueError('Preserve previous proposal; additive version required')
    if not path.exists():
        path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'proposal': str(path), 'sha256': sha(path), 'implementation_ready': False,
                      'NN_executed': False, 'images_decoded': 0}, indent=2))


if __name__ == '__main__':
    main()
