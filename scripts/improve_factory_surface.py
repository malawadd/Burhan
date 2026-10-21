"""Train-only hard-negative mining and calibration-only high-recall selection.

The reused 80-image comparison is deliberately a separate invocation. This is
an experiment, not a new independent test or factory acceptance certificate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from edge_runtime.profiles.factory_defect import SurfaceDefectModel, extract_features_fast
from train_factory_surface import samples, matrix

THRESHOLDS = [.5, .6, .7, .75, .8, .85, .9, .925, .95, .975, .98, .99, .995]
AREAS = [2, 4, 8, 16, 32, 64, 128]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose(records):
    choices = []
    for threshold in THRESHOLDS:
        for area in AREAS:
            result = matrix([r['defective'] for r in records],
                            [r['maximum_areas'][str(threshold)] >= area for r in records])
            choices.append({'threshold': threshold, 'area': area, **result})
    eligible = [c for c in choices if c['precision'] >= .9]
    if not eligible:
        raise ValueError('No calibration operating point meets >=90% precision')
    # Recall first; then precision avoids unnecessary nuisance rejects. Among
    # equal decisions prefer lower score then smaller supported defect area.
    selected = max(eligible, key=lambda c: (c['recall'], c['precision'], -c['threshold'], -c['area']))
    return selected, choices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT/'datasets/factory/KolektorSDD')
    parser.add_argument('--baseline', type=Path, default=ROOT/'models/factory/kolektor_surface_rf.joblib')
    parser.add_argument('--model', type=Path, default=ROOT/'models/factory/kolektor_surface_rf_v3_hardneg.joblib')
    parser.add_argument('--output', type=Path, default=ROOT/'output/optimization/factory-accuracy/training-calibration.json')
    args = parser.parse_args()
    cv2.setNumThreads(2)
    rng = np.random.default_rng(20261021)
    started = time.perf_counter()
    baseline = SurfaceDefectModel(args.baseline, backend='native_cascade')
    split = baseline.metadata['split_items']
    # Protocol materialized BEFORE fitting or accessing any comparison labels.
    protocol = {'seed': 20261021, 'training_items': split['train'], 'calibration_items': split['calibration'],
                'excluded_reused_comparison_items': split['test'], 'trees': 16, 'max_depth': 16,
                'min_samples_leaf': 2, 'class_weight': 'balanced_subsample',
                'random_normal_per_image': 1000, 'hard_normal_per_image': 1000,
                'hard_mining_normal_pool': 20000, 'positive_pixel_cap': 2000, 'positive_repeats': 2,
                'mask_resize': 'INTER_AREA > 0; preserve any annotated source coverage in a grid cell',
                'threshold_grid': THRESHOLDS, 'component_area_grid': AREAS,
                'selection': 'Calibration precision >=90%; maximize recall, precision, then lower vote threshold and smaller component area',
                'selection_caveat': 'Calibration was already used by v1; development evidence, not blind validation',
                'baseline_sha256': baseline.sha256}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_name('protocol-before-training.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    xrows, yrows = [], []
    train_rows = samples(args.dataset, split['train'])
    for i, (path, mask, _) in enumerate(train_rows):
        features, shape = extract_features_fast(cv2.imread(str(path)), baseline.width)
        target = cv2.resize(mask.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_AREA).ravel() > 0
        pos, neg = np.flatnonzero(target), np.flatnonzero(~target)
        normal = rng.choice(neg, min(1000, len(neg)), replace=False)
        pool = rng.choice(neg, min(20000, len(neg)), replace=False)
        # Score only original TRAIN pixels; no calibration/test labels involved.
        score, _ = baseline.native.score(np.ascontiguousarray(features[pool]), 0.)
        hard = pool[np.argpartition(score, max(0, len(score)-1000))[-1000:]]
        positive = rng.choice(pos, min(2000, len(pos)), replace=False) if len(pos) else pos
        chosen = np.concatenate([normal, hard, np.tile(positive, 2)])
        xrows.append(features[chosen]); yrows.append(target[chosen].astype(np.int32))
        if (i+1) % 40 == 0:
            print(f'Mined {i+1}/{len(train_rows)} training images', flush=True)
    x, y = np.concatenate(xrows), np.concatenate(yrows)
    forest = RandomForestClassifier(n_estimators=16, max_depth=16, min_samples_leaf=2,
                                   class_weight='balanced_subsample', n_jobs=2, random_state=20261021)
    forest.fit(x, y)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(forest, args.model, compress=3)
    metadata = {**baseline.metadata, 'model_name': 'Kolektor surface RF v3: hard negatives, coverage-preserving masks',
                'trained_utc': datetime.now(timezone.utc).isoformat(), 'model_sha256': digest(args.model),
                'trees': 16, 'maximum_tree_depth': 16, 'pixel_threshold': .5, 'minimum_component_pixels': 4,
                'training_protocol': protocol, 'threshold_selection': protocol['selection'],
                'profile_version': 'kolektor_rf_v3_hardneg', 'field_validated': False}
    meta_path = args.model.with_suffix(args.model.suffix+'.json')
    meta_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    candidate = SurfaceDefectModel(args.model, backend='native_cascade')
    records = []
    for path, _, label in samples(args.dataset, split['calibration']):
        candidate.pixel_threshold = .5
        scores, _, _ = candidate.predict(cv2.imread(str(path)))
        areas = {}
        for threshold in THRESHOLDS:
            _, _, stats, _ = cv2.connectedComponentsWithStats((scores >= threshold).astype(np.uint8), connectivity=8)
            areas[str(threshold)] = int(stats[1:, cv2.CC_STAT_AREA].max()) if len(stats)>1 else 0
        records.append({'path': str(path.relative_to(args.dataset)), 'defective': label, 'maximum_areas': areas})
    selected, choices = choose(records)
    jackknife = []
    for item in split['calibration']:
        fit = [r for r in records if Path(r['path']).parts[0] != item]
        held = [r for r in records if Path(r['path']).parts[0] == item]
        point, _ = choose(fit)
        predictions = [r['maximum_areas'][str(point['threshold'])] >= point['area'] for r in held]
        jackknife.extend({'path': r['path'], 'defective': r['defective'], 'predicted': p,
                          'selected_without_item': point} for r, p in zip(held, predictions))
    metadata['pixel_threshold'] = selected['threshold']
    metadata['minimum_component_pixels'] = selected['area']
    metadata['calibration_selected_point'] = selected
    meta_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    report = {'protocol': protocol, 'selected': selected, 'choices': choices, 'calibration_records': records,
              'calibration_item_jackknife': matrix([r['defective'] for r in jackknife], [r['predicted'] for r in jackknife]),
              'jackknife_records': jackknife, 'training_rows': len(x), 'positive_rows': int(y.sum()),
              'model_sha256': digest(args.model), 'metadata_sha256': digest(meta_path),
              'scikit_learn': sklearn.__version__, 'elapsed_seconds': time.perf_counter()-started}
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in {'protocol','choices','calibration_records','jackknife_records'}}, indent=2), flush=True)


if __name__ == '__main__':
    main()
