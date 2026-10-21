"""Actual fitted crack GPU parity/CAL and all-three strict freeze; never implicit NN."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/optimization/construction-pixel-supervision-v1'
sys.path.insert(0, str(ROOT))
from train_construction_pixel_supervision import validate_execution, MODELS, sha, calibration_ground_truth, canonical_calibration_rows


def select_threshold(points):
    eligible = [p for p in points if (p['precision'] or 0) >= .8 and p['negative_image_fpr'] <= .1]
    selected = max(eligible, key=lambda p: (p['recall'] or 0, p['precision'] or 0, p['threshold'])) if eligible else None
    diagnostic = max(points, key=lambda p: (p['f1'], p['precision'] or 0, p['threshold']))
    return selected, diagnostic


def checkpoint(step):
    path = MODELS / f'crack-pixel-step{step}.onnx'
    meta_path = path.with_suffix('.provenance.json')
    meta = json.loads(meta_path.read_text())
    receipt, execution, protocol, prepared = validate_execution()
    if meta['step'] != step or meta['sha256'] != sha(path) or meta['execution_receipt_sha256'] != sha(receipt):
        raise ValueError('Actual checkpoint/export/protocol provenance differs')
    if sha(path.with_suffix('.pt')) != meta['checkpoint_sha256'] or sha(ROOT / meta['train_parity_reference']) != meta['train_parity_reference_sha256']:
        raise ValueError('TRAIN reference or trained tensor checkpoint changed')
    if meta['prepared_manifest_sha256'] != protocol['prepared_manifest_sha256'] or meta['proposal_sha256'] != execution['proposal_sha256']:
        raise ValueError('Prepared inputs/proposal identity differs')
    if meta['classes'] != ['crack'] or meta['input_shape'] != [1, 3, 384, 384] or meta['output_shape'] != [1, 1, 384, 384] or meta['exposure_policy'] != 'low_light_gain':
        raise ValueError('Single-channel serving geometry/preprocessing changed')
    return path, meta, receipt, execution, protocol


def validate_annotation_receipts():
    """Bind publisher box GT and negative identities, not image hashes alone."""
    from evaluate_construction_round3 import BASE
    prior = json.loads((ROOT / 'output/evaluation/construction-round3/litecrackseg-calibration/preregistered-protocol.json').read_text())
    if sha(BASE / 'exposures.json') != prior['source_manifest_sha256'] or sha(BASE / 'background-import.json') != prior['background_import_sha256']:
        raise ValueError('Protected publisher GT/background annotation receipt changed')
    return {name: sha(BASE / name) for name in ['exposures.json', 'background-import.json']}


def calibrate(step, workload_note):
    import cv2
    import numpy as np
    from edge_runtime.profiles.construction_lite import LiteCrackConstructionModel, decode_crack_map
    from evaluate_construction_round3 import sampled, score_frame, summarize
    path, meta, receipt, execution, protocol = checkpoint(step)
    annotation_pins = validate_annotation_receipts()
    target = OUT / f'checkpoint{step}-calibration.json'
    protected_artifacts = [target, OUT / f'checkpoint{step}-fitted-gpu-parity.json',
                           OUT / f'checkpoint{step}-actual-calibration-maps.npz',
                           OUT / f'checkpoint{step}-selected-calibration-records.json']
    if any(p.exists() for p in protected_artifacts):
        raise ValueError('Refusing to overwrite completed/failed/partial CAL evidence; retry needs separately versioned authorization')
    cv2.setNumThreads(1)
    model = LiteCrackConstructionModel({'engine': 'openvino', 'openvino_device': 'GPU', 'cpu_threads': 1,
                                        'exposure_policy': 'low_light_gain', 'mask_rendering': 'masks'}, meta)
    # GPU guard is fixed TRAIN+zero only, BEFORE any CAL pixels/NN are read.
    reference = np.load(ROOT / meta['train_parity_reference'])
    source = meta['train_parity_source']
    if sha(ROOT / source['image']) != source['image_sha256']:
        raise ValueError('Fixed TRAIN parity source changed')
    raw = cv2.imread(str(ROOT / source['image']))
    observed = model.predict_maps(raw)
    zero = model.predict_maps(np.zeros((384, 384, 3), np.uint8))
    parity = {'model_sha256': sha(path), 'metadata_sha256': sha(path.with_suffix('.provenance.json')),
              'reference_sha256': meta['train_parity_reference_sha256'], 'execution_receipt_sha256': sha(receipt),
              'fixed_train_source': source['image'], 'max_probability_error_train': float(np.abs(observed - reference['probability'][0, 0]).max()),
              'max_probability_error_zero': float(np.abs(zero - reference['zero_probability'][0, 0]).max()),
              'all_checks_passed': bool(np.isfinite(observed).all() and np.isfinite(zero).all()
                                        and np.allclose(observed, reference['probability'][0, 0], atol=1e-4, rtol=1e-3)
                                        and np.allclose(zero, reference['zero_probability'][0, 0], atol=1e-4, rtol=1e-3)),
              'guard': 'CPU Torch reference versus actual OpenVINO GPU same384 sigmoid, TRAIN+zero; atol1e-4/rtol1e-3 unchanged beforeCAL',
              'actual_workload_note': workload_note, 'capacity_claim': False}
    parity_path = OUT / f'checkpoint{step}-fitted-gpu-parity.json'
    parity_path.write_text(json.dumps(parity, indent=2), encoding='utf-8')
    if not parity['all_checks_passed']:
        raise ValueError('Fitted GPU parity failed; no CAL inference permitted')
    rows = sampled('calibration')
    actual_gt = {**canonical_calibration_rows(rows), 'annotation_json_pins': annotation_pins}
    if actual_gt != execution['calibration_ground_truth_identity']:
        raise ValueError('Actual publisher boxes/annotation IDs/frame indices/negative scope differ from pre-fit canonical GT')
    identity = [{'source': r['source'], 'image': str(r['image'].relative_to(ROOT)), 'sha256': sha(r['image'])} for r in rows]
    declared = protocol['calibration']['ordered_input_metadata']
    if len(rows) != 122 or len(identity) != len(declared) or any(a['source'] != b['source'] or Path(a['image']) != Path(b['image']) or a['sha256'] != b['sha256'] for a, b in zip(identity, declared)):
        raise ValueError('Actual122 CAL images differ from pre-fit source-only receipt')
    maps, shapes = [], []
    for row in rows:
        frame = cv2.imread(str(row['image']))
        shapes.append(frame.shape[:2]); maps.append(model.predict_maps(frame))
    maps_path = OUT / f'checkpoint{step}-actual-calibration-maps.npz'
    np.savez_compressed(maps_path, probabilities=np.stack(maps))
    points = []
    all_records = {}
    for threshold in protocol['calibration']['threshold_grid']:
        records = []
        for row, probability, shape in zip(rows, maps, shapes):
            predictions = decode_crack_map(probability, shape, threshold, 4)
            record = score_frame(predictions, row['expected'], {'crack'}, negative=row['negative'])
            record.update(source=row['source'], negative=row['negative'], frame_index=row['frame_index'])
            records.append(record)
        summary = summarize(records)['by_class']['crack']
        negatives = [r for r in records if r['negative']]
        if len(negatives) != 10:
            raise ValueError('Fixed10 true target-negative CAL photos changed')
        fpr = sum(bool(r['predictions']) for r in negatives) / len(negatives)
        p, r = summary['precision'] or 0, summary['recall'] or 0
        thin_ids = {e['id'] for record in records for e in record['expected'] if e['label'] == 'crack'
                    and min(e['bbox'][2:]) <= 30 and max(e['bbox'][2:]) >= 2 * min(e['bbox'][2:])}
        thin_matches = sum(p.get('matched_annotation_id') in thin_ids for record in records for p in record['predictions'])
        points.append({'threshold': threshold, **summary, 'negative_image_fpr': fpr, 'f1': 2 * p * r / (p + r) if p + r else 0,
                       'thin_geometry_annotations': len(thin_ids), 'thin_geometry_localized': thin_matches,
                       'thin_geometry_recall': thin_matches / len(thin_ids) if thin_ids else None})
        all_records[threshold] = records
    selected, diagnostic = select_threshold(points)
    verification_point = selected or diagnostic  # Diagnostic never becomes an eligible winner.
    model.threshold = verification_point['threshold']
    def signature(predictions):
        return [(r['label'], tuple(r['bbox']), round(float(r['score']), 6), hashlib.sha256(r['mask'].tobytes()).hexdigest()) for r in predictions]
    # Every selected/diagnostic mask and box must be exactly the actual serving decoder.
    for row, probability, shape in zip(rows, maps, shapes):
        cached = decode_crack_map(probability, shape, model.threshold, 4, masks=True)
        actual = model.infer(cv2.imread(str(row['image'])))
        if signature(cached) != signature(actual):
            raise ValueError('Actual selected serving versus cached mask/box/score parity failed; no freeze allowed')
    report = {'step': step, 'model_path': str(path), 'model_sha256': sha(path),
              'model_description': model.describe(),
              'metadata_sha256': sha(path.with_suffix('.provenance.json')), 'execution_receipt_sha256': sha(receipt),
              'proposal_sha256': execution['proposal_sha256'], 'prepared_manifest_sha256': execution['prepared_manifest_sha256'],
              'annotation_receipt_pins': annotation_pins,
              'calibration_ground_truth_identity': actual_gt,
              'parity_report_sha256': sha(parity_path), 'actual_maps_sha256': sha(maps_path),
              'ordered_calibration_sources': identity, 'calibration_source_identity_sha256': hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
              'calibration_protocol': protocol['calibration'], 'threshold_candidates': points,
              'eligible_selected': selected, 'failed_diagnostic': diagnostic if selected is None else None,
              'serving_parity_all122_masks_boxes_scores': True, 'actual_workload_note': workload_note,
              'scope': 'Actual GPU CAL only; publisher-box IoU.5 primary, unsupported spall explicitly unscored; no video pixelGT/inferenceFPS/heldout claim',
              'heldout_images_opened': 0, 'reserved_lcsd102_pixels_opened': 0}
    records_path = OUT / f'checkpoint{step}-selected-calibration-records.json'
    records_path.write_text(json.dumps(all_records[verification_point['threshold']], indent=2), encoding='utf-8')
    report['selected_records_sha256'] = sha(records_path)
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'step': step, 'eligible': selected, 'failed_diagnostic': report['failed_diagnostic'], 'report_sha256': sha(target)}, indent=2))


def validate_report_set(reports):
    if sorted(r['step'] for r in reports) != [100, 200, 300]:
        raise ValueError('All three fixed100/200/300 actual CAL reports are mandatory')
    identities = {r['calibration_source_identity_sha256'] for r in reports}
    if len(identities) != 1 or any(not r['serving_parity_all122_masks_boxes_scores'] for r in reports):
        raise ValueError('Same122 CAL sources and actual serving parity required for every checkpoint')
    for row in reports:
        sources = row['ordered_calibration_sources']
        if len(sources) != 122 or len({r['image'] for r in sources}) != 122 or hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest() != row['calibration_source_identity_sha256']:
            raise ValueError('Actual122 ordered source identities/digest differ')
        point = row['eligible_selected']
        if point is not None and ((point['precision'] or 0) < .8 or point['negative_image_fpr'] > .1):
            raise ValueError('A purported eligible point violates fixed strict guards')


def freeze():
    receipt, execution, protocol, _ = validate_execution()
    completed_path = OUT / 'training-complete.json'
    completed = json.loads(completed_path.read_text())
    if completed['fixed_steps_completed'] != 300 or completed['execution_receipt_sha256'] != sha(receipt) or completed['history'][-1]['step'] != 300:
        raise ValueError('The complete fixed300-step fit must finish before global selection')
    if completed['calibration_images_opened'] or completed['heldout_images_opened'] or completed['reserved_lcsd102_pixels_opened']:
        raise ValueError('Training data separation failed')
    reports = [json.loads((OUT / f'checkpoint{s}-calibration.json').read_text()) for s in [100, 200, 300]]
    validate_report_set(reports)
    annotation_pins = validate_annotation_receipts()
    current_gt = calibration_ground_truth()
    if current_gt != execution['calibration_ground_truth_identity']:
        raise ValueError('Canonical publisher GT changed since before first optimizer')
    if sorted(c['step'] for c in completed['checkpoints']) != [100, 200, 300]:
        raise ValueError('Completed fit lacks exactly the three protected exported checkpoints')
    choices = []
    for report in reports:
        path, meta, _, _, _ = checkpoint(report['step'])
        if report['model_path'] != str(path) or report['model_sha256'] != sha(path) or report['metadata_sha256'] != sha(path.with_suffix('.provenance.json')):
            raise ValueError('Calibrated checkpoint/model metadata identity changed')
        if report['proposal_sha256'] != execution['proposal_sha256'] or report['execution_receipt_sha256'] != sha(receipt) or report['prepared_manifest_sha256'] != execution['prepared_manifest_sha256'] or report['calibration_protocol'] != protocol['calibration']:
            raise ValueError('Actual CAL report differs from protected protocol/source')
        if report['annotation_receipt_pins'] != annotation_pins or [r['threshold'] for r in report['threshold_candidates']] != protocol['calibration']['threshold_grid']:
            raise ValueError('Publisher GT or actual fixed threshold grid differs')
        saved_gt = report['calibration_ground_truth_identity']
        if saved_gt != current_gt or hashlib.sha256(json.dumps(saved_gt['ordered_records'], sort_keys=True).encode()).hexdigest() != saved_gt['sha256']:
            raise ValueError('Actual all-three CAL expected boxes/IDs/negative/source scope differ')
        trained = next(c for c in completed['checkpoints'] if c['step'] == report['step'])
        if trained['sha256'] != report['model_sha256'] or trained['metadata_sha256'] != report['metadata_sha256'] or trained['checkpoint_sha256'] != meta['checkpoint_sha256']:
            raise ValueError('Actual trained/exported checkpoint differs from calibrated graph')
        parity_path = OUT / f'checkpoint{report["step"]}-fitted-gpu-parity.json'
        parity = json.loads(parity_path.read_text())
        if sha(parity_path) != report['parity_report_sha256'] or not parity['all_checks_passed'] or parity['model_sha256'] != sha(path) or parity['reference_sha256'] != meta['train_parity_reference_sha256']:
            raise ValueError('Actual fitted GPU parity missing or stale')
        if sha(OUT / f'checkpoint{report["step"]}-actual-calibration-maps.npz') != report['actual_maps_sha256'] or sha(OUT / f'checkpoint{report["step"]}-selected-calibration-records.json') != report['selected_records_sha256']:
            raise ValueError('Actual CAL map/record evidence changed')
        expected_point, _ = select_threshold(report['threshold_candidates'])
        if expected_point != report['eligible_selected']:
            raise ValueError('Selected point is not the fixed actual-grid winner')
        if expected_point is not None:
            choices.append({'step': report['step'], 'model_path': str(path), 'model_sha256': sha(path),
                            'metadata_sha256': sha(path.with_suffix('.provenance.json')), 'point': expected_point})
    choices.sort(key=lambda row: (row['point']['recall'] or 0, row['point']['precision'] or 0,
                                  -row['point']['negative_image_fpr'], -row['step']), reverse=True)
    result = {'frozen_utc': datetime.now(timezone.utc).isoformat(), 'accepted': bool(choices),
              'winner': choices[0] if choices else None, 'ranked_eligible': choices,
              'proposal_sha256': execution['proposal_sha256'], 'execution_receipt_sha256': sha(receipt),
              'training_complete_sha256': sha(completed_path),
              'calibration_report_sha256': {str(s): sha(OUT / f'checkpoint{s}-calibration.json') for s in [100, 200, 300]},
              'calibration_source_identity_sha256': reports[0]['calibration_source_identity_sha256'],
              'scope': 'All-three CAL-only selection. Rejected if none meet precision.8/negativeFPR.1. No heldout/photo-test/video generalization or real-time qualification follows automatically.',
              'heldout_images_used_for_selection': 0, 'reserved_lcsd102_pixels_opened': 0}
    target = OUT / 'winner-frozen-before-future-evaluation.json'
    if target.exists():
        old = json.loads(target.read_text())
        if any(old[k] != result[k] for k in ['winner', 'execution_receipt_sha256', 'calibration_report_sha256']):
            raise ValueError('Refusing to replace immutable selection')
    else:
        target.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--calibrate', action='store_true')
    group.add_argument('--freeze', action='store_true')
    parser.add_argument('--step', type=int, choices=[100, 200, 300])
    parser.add_argument('--workload-note')
    args = parser.parse_args()
    if args.calibrate:
        if args.step is None or not args.workload_note:
            parser.error('--calibrate requires --step and truthful --workload-note; coordinate GPU before launch')
        calibrate(args.step, args.workload_note)
    else:
        freeze()
