"""Fixed TUT versus FINAL300 reserved-photo diagnostic; all acquisition is gated."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
INPUT_OUT = ROOT / 'output/optimization/construction-pixel-supervision-v1'
OUT = ROOT / 'output/optimization/construction-pixel-supervision-v2'
RESERVED = INPUT_OUT / 'lcsd-102-reservation-metadata'
V1_SHA = '9dbd73d736270fb5732ad4025986fbf1565812b1196ead41b2a926c9fd5289ee'
V2_SHA = '051f9f7cdbda2198776f019f267127061177b27d192238007ef6105ce2977034'
BASELINE_SHA = '20936464f5fb6ef32cf5d84a87c84616bf1d5461364fbfb466dd7c4846d04055'
RECOVERY_SHA = '3e70d9bf5c796fc0a38b5f9bb00852f5333c3c8cbf8958366a9a24fee3460e54'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding_value():
    """No images/NN: fail closed unless final300 and ALL3 CAL/freezer are complete."""
    from train_construction_pixel_supervision_v2 import validate_execution
    from evaluate_construction_pixel_supervision_v2 import checkpoint
    v1 = RESERVED / 'proposed-pixel-test-protocol-before-opening.json'
    v2 = RESERVED / 'proposed-pixel-test-protocol-v2-fixed-pair-before-opening.json'
    recovery = RESERVED / 'fixed-pair-path-recovery-v3-before-opening.json'
    if sha(v1) != V1_SHA or sha(v2) != V2_SHA or sha(recovery) != RECOVERY_SHA:
        raise ValueError('Pre-opening fixed-pair diagnostic proposal changed')
    receipt, execution, _, _ = validate_execution()
    if sha(receipt) != 'bed5a4c06f24fe5d9799845726203607396d02850527a5208adabb7048cf85df':
        raise ValueError('Exact metadata-only restart execution lineage required')
    completed_path = OUT / 'training-complete.json'
    completed = json.loads(completed_path.read_text())
    if completed['fixed_steps_completed'] != 300 or [r['step'] for r in completed['history']] != list(range(1, 301)) or completed['execution_receipt_sha256'] != sha(receipt):
        raise ValueError('Exact complete fixed300-step lineage required before any reserved pixel opening')
    frozen_path = OUT / 'winner-frozen-before-future-evaluation.json'
    frozen = json.loads(frozen_path.read_text())
    if frozen['training_complete_sha256'] != sha(completed_path) or frozen['execution_receipt_sha256'] != sha(receipt):
        raise ValueError('All-three global CAL freeze is missing/stale')
    if sorted(frozen['calibration_report_sha256']) != ['100', '200', '300']:
        raise ValueError('Every fixed checkpoint CAL must finish even if globally rejected')
    for step in [100, 200, 300]:
        report_path = OUT / f'checkpoint{step}-calibration.json'
        if sha(report_path) != frozen['calibration_report_sha256'][str(step)]:
            raise ValueError('Actual all-three CAL report changed')
        report = json.loads(report_path.read_text())
        parity_path = OUT / f'checkpoint{step}-fitted-gpu-parity.json'
        parity = json.loads(parity_path.read_text())
        if not report['serving_parity_all122_masks_boxes_scores'] or sha(parity_path) != report['parity_report_sha256'] or not parity['all_checks_passed']:
            raise ValueError('Fitted numeric/serving parity required; CAL metric failure alone is permitted')
    final, meta, _, _, _ = checkpoint(300)
    baseline = ROOT / 'models/construction/thin-contingency/litecrackseg-tut-384.onnx'
    baseline_pt = ROOT / 'models/construction/thin-contingency/litecrackseg-TUT.pth'
    if sha(baseline) != BASELINE_SHA or sha(baseline_pt) != 'c12fb1ce991e479a32e5a26d7648ce9d20a440b9af4fd0411d321aa6a2eed497':
        raise ValueError('Fixed original TUT baseline changed')
    return {'v1_protocol_sha256': V1_SHA, 'v2_protocol_sha256': V2_SHA, 'path_recovery_v3_sha256': RECOVERY_SHA,
            'execution_receipt_sha256': sha(receipt), 'training_complete_sha256': sha(completed_path),
            'all_three_freeze_sha256': sha(frozen_path), 'calibration_global_accepted': frozen['accepted'],
            'fixed_pair': {'original_tut': {'path': str(baseline), 'sha256': sha(baseline),
                                          'metadata_sha256': sha(baseline.with_suffix('.provenance.json'))},
                           'final300': {'path': str(final), 'sha256': sha(final), 'checkpoint_sha256': meta['checkpoint_sha256'],
                                        'metadata_sha256': sha(final.with_suffix('.provenance.json'))}},
            'source_code_sha256': sha(Path(__file__)),
            'shared_preprocessing_sha256': sha(ROOT / 'edge_runtime/profiles/construction_preprocessing.py'),
            'known_reference_inventory_sha256': sha(ROOT / 'datasets/construction/lcsd-low-light-future-r4/known-reference-files.json'),
            'fixed_threshold': .5, 'operator': 'strictly_greater_than',
            'diagnostic_only': True, 'model_selection_or_promotion_allowed': False,
            'interpretation': 'OUR reserved well-lit photo pixel diagnostic using authorTRAIN. Unknown TUT pretraining/site overlap; not video GT, author benchmark or CAL-gate replacement.'}


def bind():
    value = binding_value()
    target = RESERVED / 'actual-final300-fixed-pair-binding-before-pixels.json'
    originals = OUT / 'reserved102-original'
    if (OUT / 'reserved102-source-manifest.json').exists() or originals.exists() and any(originals.rglob('*')):
        raise ValueError('Binding must precede any102 pixel acquisition')
    if target.exists():
        old = json.loads(target.read_text())
        if {k: v for k, v in old.items() if k != 'bound_utc'} != value:
            raise ValueError('Refusing to replace an existing different fixed-pair binding')
    else:
        target.write_text(json.dumps({'bound_utc': datetime.now(timezone.utc).isoformat(), **value}, indent=2), encoding='utf-8')
    print(json.dumps({'binding': str(target), 'sha256': sha(target), 'pixels_downloaded_by_this_command': 0}, indent=2))


def checked_binding():
    path = RESERVED / 'actual-final300-fixed-pair-binding-before-pixels.json'
    old = json.loads(path.read_text())
    if {k: v for k, v in old.items() if k != 'bound_utc'} != binding_value():
        raise ValueError('Fixed-pair binding/source/prerequisites changed; no pixel opening or inference allowed')
    return path, old


def acquire():
    """Only called explicitly AFTER all postfit/preopening guards; no NN."""
    binding_path, binding = checked_binding()
    target = OUT / 'reserved102-source-manifest.json'
    if target.exists():
        raise ValueError('Preserve completed102 source acquisition; no overwrite')
    from pip._vendor import truststore
    truststore.inject_into_ssl()
    protocol = json.loads((RESERVED / 'proposed-pixel-test-protocol-before-opening.json').read_text())
    revision = protocol['hf_revision']
    if revision != '254e9a987d1ee9d925472a42750c6f5f58112d9f' or len(protocol['pairs']) != 102:
        raise ValueError('Pinned original102 source inventory changed')
    records = []
    for pair in protocol['pairs']:
        record = {'id': pair['id']}
        for field in ['image', 'mask']:
            relative = Path(pair[field])
            if relative.is_absolute() or '..' in relative.parts or relative.parts[:2] != ('train', field):
                raise ValueError('Unexpected pinned source path')
            destination = OUT / 'reserved102-original' / relative
            receipt = destination.with_suffix(destination.suffix + '.download.json')
            url = f'https://huggingface.co/datasets/Chrisathy/LCSD/resolve/{revision}/{pair[field]}'
            if destination.exists():
                saved = json.loads(receipt.read_text())
                if saved['url'] != url or saved['sha256'] != sha(destination):
                    raise ValueError('Partial acquisition source bytes/receipt differ')
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                partial = destination.with_suffix(destination.suffix + '.writing')
                if partial.exists():
                    raise ValueError('Preserve failed partial download; separately review before retry')
                with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'LocalFixedPixelDiagnostic/1.0'}), timeout=40) as response, partial.open('wb') as output:
                    while block := response.read(1024 * 1024):
                        output.write(block)
                partial.rename(destination)
                receipt.write_text(json.dumps({'url': url, 'sha256': sha(destination), 'bytes': destination.stat().st_size}, indent=2), encoding='utf-8')
            record[field] = str(destination.relative_to(ROOT))
            record[field + '_sha256'] = sha(destination)
        records.append(record)
    target.write_text(json.dumps({'binding_sha256': sha(binding_path), 'v2_protocol_sha256': V2_SHA,
                                 'revision': revision, 'count': 102, 'records': records,
                                 'pixels_acquired_after_all_postfit_guards': True, 'neural_inference_started': False}, indent=2), encoding='utf-8')
    print(json.dumps({'source_manifest_sha256': sha(target), 'pairs': len(records), 'nn_started': False}, indent=2))


def image_signature(frame):
    import cv2
    import numpy as np
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 24), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    centered = small - small.mean()
    norm = float(np.linalg.norm(centered))
    normalized = centered / norm if norm > 0 else np.zeros_like(centered)
    d = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (d[:, 1:] > d[:, :-1]).ravel()
    dhash = sum(int(value) << i for i, value in enumerate(bits))
    native = hashlib.sha256(str(frame.shape).encode() + frame.tobytes()).hexdigest()
    return {'native_sha256': native, 'dhash': dhash, 'normalized': normalized}


def checked_source_inventory(binding_path):
    source_path = OUT / 'reserved102-source-manifest.json'
    source = json.loads(source_path.read_text())
    protocol = json.loads((RESERVED / 'proposed-pixel-test-protocol-before-opening.json').read_text())
    if source['binding_sha256'] != sha(binding_path) or source['count'] != 102 or [r['id'] for r in source['records']] != [r['id'] for r in protocol['pairs']]:
        raise ValueError('Every102 source must retain its exact original declared order')
    for record, declared in zip(source['records'], protocol['pairs']):
        for field in ['image', 'mask']:
            path = OUT / 'reserved102-original' / declared[field]
            if Path(record[field]) != path.relative_to(ROOT) or sha(path) != record[field + '_sha256']:
                raise ValueError('Reserved source path/byte identity differs from original inventory')
            download = json.loads(path.with_suffix(path.suffix + '.download.json').read_text())
            expected_url = f'https://huggingface.co/datasets/Chrisathy/LCSD/resolve/{protocol["hf_revision"]}/{declared[field]}'
            if download['url'] != expected_url or download['sha256'] != record[field + '_sha256']:
                raise ValueError('Reserved bytes are not tied to the pinned source download receipt')
    return source_path, source


def near_signature(first, second):
    distance = (first['dhash'] ^ second['dhash']).bit_count()
    correlation = float(first['normalized'] @ second['normalized'])
    return distance <= 4 or distance <= 8 and correlation >= .95, distance, correlation


def quarantine_prior_ambiguous_view(prior, current_id, distance, correlation):
    """Quarantine both endpoints of a within-reserved ambiguity before NN."""
    prior['leakage_flags'].append({'reference': current_id,
                                   'reason': 'within_reserved_duplicate_or_ambiguous_near_view',
                                   'dhash_distance': distance, 'correlation': correlation})
    reason = 'leakage_or_ambiguous_view_quarantine_before_NN'
    if reason not in prior['quarantine_reasons']:
        prior['quarantine_reasons'].append(reason)
    prior['clean'] = False


def screen():
    """Pre-NN encoding/leakage review; every ambiguous near flag is quarantined."""
    import cv2
    import numpy as np
    from PIL import Image
    cv2.setNumThreads(1)
    binding_path, binding = checked_binding()
    source_path, source = checked_source_inventory(binding_path)
    target = OUT / 'reserved102-leakage-encoding-freeze-before-inference.json'
    if target.exists():
        raise ValueError('Preserve pre-NN leakage decision; do not revise it after outcomes')
    prepared = json.loads((INPUT_OUT / 'prepared-train-manifest.json').read_text())
    references = {r['image']: r['image_sha256'] for r in prepared['records']}
    known_path = ROOT / 'datasets/construction/lcsd-low-light-future-r4/known-reference-files.json'
    if sha(known_path) != binding['known_reference_inventory_sha256']:
        raise ValueError('Pre-opening known-reference inventory changed')
    known = json.loads(known_path.read_text())
    references.update({r['path']: r['sha256'] for r in known})
    signatures = []
    for name, digest in references.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != digest:
            raise ValueError('Known TRAIN/materialized leakage reference changed')
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError('Unable to decode a declared leakage reference')
        signatures.append((name, digest, image_signature(frame)))
    results, previous, by_id = [], [], {}
    for record in source['records']:
        row = dict(record); reasons = []; flags = []
        image, mask = ROOT / record['image'], ROOT / record['mask']
        if sha(image) != record['image_sha256'] or sha(mask) != record['mask_sha256']:
            raise ValueError('Reserved raw bytes changed after acquisition')
        frame = cv2.imread(str(image))
        truth = np.array(Image.open(mask))
        codes, counts = np.unique(truth, return_counts=True)
        row['mask_codes'] = {str(v): int(n) for v, n in zip(codes, counts)}
        if frame is None or truth.ndim != 2 or truth.shape != frame.shape[:2] or not set(codes) <= {0, 1, 255} or not np.any(truth == 1):
            reasons.append('unresolved_expected0background1crack255ignore_or_geometry')
        if frame is not None:
            current = image_signature(frame)
            for name, digest, signature in signatures:
                if digest == record['image_sha256'] or current['native_sha256'] == signature['native_sha256']:
                    flags.append({'reference': name, 'reason': 'exact_file_or_native_pixel_duplicate'})
                    continue
                flagged, distance, correlation = near_signature(current, signature)
                if flagged:
                    flags.append({'reference': name, 'reason': 'ambiguous_near_or_crossillumination_view', 'dhash_distance': distance, 'correlation': correlation})
            for earlier, digest, signature in previous:
                flagged, distance, correlation = near_signature(current, signature)
                if digest == record['image_sha256'] or current['native_sha256'] == signature['native_sha256'] or flagged:
                    flags.append({'reference': earlier, 'reason': 'within_reserved_duplicate_or_ambiguous_near_view', 'dhash_distance': distance, 'correlation': correlation})
                    quarantine_prior_ambiguous_view(by_id[earlier], record['id'], distance, correlation)
            previous.append((record['id'], record['image_sha256'], current))
            row['native_shape'] = list(frame.shape)
        if flags:
            reasons.append('leakage_or_ambiguous_view_quarantine_before_NN')
        row.update(leakage_flags=flags, quarantine_reasons=reasons, clean=not reasons)
        results.append(row)
        by_id[record['id']] = row
    target.write_text(json.dumps({'binding_sha256': sha(binding_path), 'source_manifest_sha256': sha(source_path),
                                 'records': results, 'total': 102, 'clean': sum(r['clean'] for r in results),
                                 'quarantined': sum(not r['clean'] for r in results), 'known_reference_count': len(signatures),
                                 'known_reference_inventory_sha256': sha(known_path),
                                 'review_rule': 'All exact/near/ambiguous flags conservatively quarantined before any NN outcome; no favorable-case clearing. Same-site/pretraining independence remains unknown.',
                                 'neural_inference_started': False}, indent=2), encoding='utf-8')
    print(json.dumps({'screen_sha256': sha(target), 'clean': sum(r['clean'] for r in results), 'quarantined': sum(not r['clean'] for r in results)}, indent=2))


def pixel_counts(prediction, truth, valid):
    import numpy as np
    prediction, truth, valid = prediction.astype(bool), truth.astype(bool), valid.astype(bool)
    tp = int(np.count_nonzero(prediction & truth & valid))
    fp = int(np.count_nonzero(prediction & ~truth & valid))
    fn = int(np.count_nonzero(~prediction & truth & valid))
    return {'tp': tp, 'fp': fp, 'fn': fn, 'valid_pixels': int(valid.sum()),
            'ignored_pixels': int((~valid).sum()), 'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None,
            'iou': tp / (tp + fp + fn) if tp + fp + fn else None,
            'dice': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def aggregate_pixels(records, key):
    import numpy as np
    values = [r[key] for r in records]
    totals = {name: sum(r[name] for r in values) for name in ['tp', 'fp', 'fn', 'valid_pixels', 'ignored_pixels']}
    tp, fp, fn = totals['tp'], totals['fp'], totals['fn']
    totals.update(precision=tp / (tp + fp) if tp + fp else None,
                  recall=tp / (tp + fn) if tp + fn else None,
                  iou=tp / (tp + fp + fn) if tp + fp + fn else None,
                  dice=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None)
    macro = {name: float(np.mean([v[name] for v in values])) if values and all(v[name] is not None for v in values) else None
             for name in ['precision', 'recall', 'iou', 'dice']}
    return {'images': len(values), 'micro': totals, 'macro_all_views': macro,
            'undefined_precision_views': sum(v['precision'] is None for v in values),
            'empty_predictions_on_positive_gt': sum(v['tp'] + v['fp'] == 0 and v['fn'] > 0 for v in values)}


def audit(workload_note):
    import cv2
    import numpy as np
    from PIL import Image
    from edge_runtime.profiles.construction_lite import LiteCrackConstructionModel, decode_crack_map
    cv2.setNumThreads(1)
    binding_path, binding = checked_binding()
    screen_path = OUT / 'reserved102-leakage-encoding-freeze-before-inference.json'
    screened = json.loads(screen_path.read_text())
    source_path, source_inventory = checked_source_inventory(binding_path)
    if screened['binding_sha256'] != sha(binding_path) or screened['source_manifest_sha256'] != sha(source_path) or screened['known_reference_inventory_sha256'] != binding['known_reference_inventory_sha256']:
        raise ValueError('Pre-NN source/leakage freeze changed')
    if [r['id'] for r in screened['records']] != [r['id'] for r in source_inventory['records']] or screened['total'] != 102 or any(r['clean'] != (not r['quarantine_reasons']) or bool(r['leakage_flags']) and r['clean'] for r in screened['records']):
        raise ValueError('Pre-NN quarantine/order decisions are inconsistent with all102 sources')
    clean = [r for r in screened['records'] if r['clean']]
    if len(clean) != screened['clean'] or len(clean) + screened['quarantined'] != 102:
        raise ValueError('All102 clean/quarantine accounting changed')
    if not clean:
        raise ValueError('No clean reconciled reserved views; report quarantine counts, never select replacements')
    destination = OUT / 'reserved102-fixed-pair-diagnostic'
    if destination.exists():
        raise ValueError('Preserve complete/failed/partial fixed-pair audit; separately version any authorized retry')
    destination.mkdir()
    models, records = {}, []
    report = {'binding_sha256': sha(binding_path), 'leakage_freeze_sha256': sha(screen_path),
              'fixed_probability': .5, 'operator': 'strictly_greater_than', 'actual_workload_note': workload_note,
              'models': {}, 'failed_source_id': None,
              'unprocessed_clean_source_ids': [source['id'] for source in clean],
              'total_reserved_sources': 102, 'clean_sources': len(clean), 'quarantined_sources': screened['quarantined'],
              'all_source_decisions': screened['records'], 'records': records,
              'scope': 'Fixed TUT versus FINAL300 diagnostic only. Primary raw native pixel masks; served component masks secondary. No CAL override, promotion, video annotation, inferenceFPS or pretraining/site independence claim.'}
    try:
        # Explicit fixed order also pins the four-panel captions independently of JSON key sorting.
        for name in ['original_tut', 'final300']:
            identity = binding['fixed_pair'][name]
            path = Path(identity['path'])
            meta = json.loads(path.with_suffix('.provenance.json').read_text())
            if sha(path) != identity['sha256'] or sha(path.with_suffix('.provenance.json')) != identity['metadata_sha256']:
                raise ValueError('Fixed diagnostic pair changed')
            models[name] = LiteCrackConstructionModel({'engine': 'openvino', 'openvino_device': 'GPU', 'cpu_threads': 1,
                                                      'exposure_policy': 'low_light_gain', 'confidence': .5, 'mask_rendering': 'masks'}, meta)
        report['models'] = {name: model.describe() for name, model in models.items()}
        for index, source in enumerate(clean):
            report['failed_source_id'] = source['id']
            path, mask_path = ROOT / source['image'], ROOT / source['mask']
            if sha(path) != source['image_sha256'] or sha(mask_path) != source['mask_sha256']:
                raise ValueError('Reserved original bytes changed after pre-NN leakage freeze')
            frame = cv2.imread(str(path)); raw_mask = np.array(Image.open(mask_path))
            if frame is None or raw_mask.ndim != 2 or raw_mask.shape != frame.shape[:2] or not set(np.unique(raw_mask)) <= {0, 1, 255} or not np.any(raw_mask == 1):
                raise ValueError('Previously reconciled mask/image contract changed')
            truth, valid = raw_mask == 1, raw_mask != 255
            row = {'id': source['id'], 'source_image_sha256': source['image_sha256'],
                   'source_mask_sha256': source['mask_sha256'], 'native_shape': list(raw_mask.shape)}
            gt = frame.copy(); gt[truth] = (gt[truth] * .3 + np.array([255, 255, 0]) * .7).astype(np.uint8)
            panels = [frame, gt]
            for name, model in models.items():
                probability = model.predict_maps(frame)
                if probability.min() < 0 or probability.max() > 1:
                    raise ValueError('Sigmoid graph returned values outside0..1')
                native = cv2.resize(probability, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                prediction = native > .5
                components = decode_crack_map(probability, frame.shape[:2], .5, 4, masks=True)
                served = np.zeros(truth.shape, bool)
                for component in components:
                    served |= component['mask']
                row[name] = pixel_counts(prediction, truth, valid)
                row[name + '_served_secondary'] = pixel_counts(served, truth, valid)
                row[name + '_probability_range'] = [float(native.min()), float(native.max())]
                mask_output = destination / f'{source["id"]}-{name}-raw-native-mask.png'
                if not cv2.imwrite(str(mask_output), prediction.astype(np.uint8) * 255):
                    raise IOError('Unable to save actual native predicted mask')
                row[name + '_mask_sha256'] = sha(mask_output)
                np.savez_compressed(destination / f'{source["id"]}-{name}-actual-probabilities.npz', native_probability=native)
                overlay = frame.copy()
                overlay[prediction] = (overlay[prediction] * .3 + np.array([0, 180, 255]) * .7).astype(np.uint8)
                panels.append(overlay)
            # ALL clean views, fixed source order: no favorable-case selection.
            panel = np.hstack(panels)
            panel = cv2.copyMakeBorder(panel, 32, 0, 0, 0, cv2.BORDER_CONSTANT, value=(22, 22, 22))
            for number, caption in enumerate(['RAW PHOTO', 'ACTUAL GT CLASS1', 'ORIGINAL TUT @.50', 'FINAL300 @.50']):
                cv2.putText(panel, caption, (number * frame.shape[1] + 5, 22), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
            evidence = destination / f'{index:03d}-{source["id"]}-all-four-panels.png'
            if not cv2.imwrite(str(evidence), panel):
                raise IOError('Unable to save fixed-order photo evidence')
            row['evidence_path'] = str(evidence.relative_to(ROOT))
            records.append(row)
            if (index + 1) % 10 == 0:
                print(f'Fixed pair completed {index + 1}/{len(clean)} clean reserved photos', flush=True)
        report['primary_pixel_metrics'] = {name: aggregate_pixels(records, name) for name in models}
        report['secondary_served_component_metrics'] = {name: aggregate_pixels(records, name + '_served_secondary') for name in models}
        report['all_clean_views_completed'] = True
        report['failed_source_id'] = None
        report['first_clean_evidence_id_fixed_by_source_order'] = clean[0]['id']
        report['healthy_image_false_reject_rate'] = None
        report['healthy_image_scope'] = 'Clean sources have positive class1 crack GT; no healthy-image or good-product negative population is measured.'
    except Exception as error:
        report['all_clean_views_completed'] = False
        report['failure'] = repr(error)
        report['accuracy_claim_allowed'] = False
        raise
    finally:
        completed_ids = {row['id'] for row in records}
        report['unprocessed_clean_source_ids'] = [source['id'] for source in clean if source['id'] not in completed_ids]
        (destination / 'all-clean-reserved-fixed-pair-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['primary_pixel_metrics'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--bind', action='store_true', help='Metadata binding only after exact300/all3CAL/globalfreeze/parity')
    group.add_argument('--acquire', action='store_true', help='Guarded102 source download AFTER actual pre-opening binding; no NN')
    group.add_argument('--screen', action='store_true', help='Encoding/known-reference leakage freeze before any NN outcomes')
    group.add_argument('--audit', action='store_true', help='Fixed-pair actual GPU photo diagnostic only AFTER binding/acquisition/pre-NN screen; coordinate GPU')
    parser.add_argument('--workload-note')
    args = parser.parse_args()
    if args.bind: bind()
    elif args.acquire: acquire()
    elif args.screen: screen()
    else:
        if not args.workload_note:
            parser.error('--audit requires a truthful coordinated --workload-note')
        audit(args.workload_note)
