"""One-shot official SDD2 test audit of two already frozen factory profiles.

No fitting, threshold search, domain-gate change, source crop or test selection.
The official ZIP stays intact; test members are decoded directly when run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import time
import zipfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge_runtime.contracts import FrameContext
from edge_runtime.profiles.factory import FactoryProfile
from factory_localization_audit import aggregate, mask_metrics, retained_mask, runtime_box_metrics, save_visual_evidence
from fetch_factory_sdd2 import EXPECTED_SHA256, sha256


DEFAULT_PROTOCOL = ROOT/'output/optimization/factory-round3/sdd2-test-protocol-before-opening.json'


def validate_protocol_selection(protocol, freeze_sha256, after_model_sha256):
    """Pure fail-closed binding, before any archive access or model construction.

    Earlier declarations without a specific freeze pin cannot authorize opening
    a later winner. A new additive declaration must identify its exact freeze.
    """
    expected_freeze = protocol.get('after_freeze_sha256')
    if not isinstance(expected_freeze, str) or len(expected_freeze) != 64 or any(c not in '0123456789abcdef' for c in expected_freeze):
        raise RuntimeError('Official-test protocol requires an explicit canonical after_freeze_sha256 pin; preserve old declarations and supply a new additive --protocol')
    if expected_freeze != freeze_sha256:
        raise RuntimeError('Selected freeze differs from the exact after freeze pinned before official-test opening')
    if 'after_model_sha256' in protocol:
        expected_model = protocol['after_model_sha256']
        if not isinstance(expected_model, str) or len(expected_model) != 64 or any(c not in '0123456789abcdef' for c in expected_model):
            raise RuntimeError('Official-test protocol after_model_sha256 must be an explicit canonical SHA256 when present')
        if expected_model != after_model_sha256:
            raise RuntimeError('After model differs from the optional exact model pin declared before official-test opening')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL,
                        help='Pre-opening declaration pinned to the exact selected freeze; preserved v1 default fails closed without that pin')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Refusing to overwrite a completed fresh-test audit')
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes.decode('utf-8'))
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    freeze_bytes = args.freeze.read_bytes()
    freeze_sha = hashlib.sha256(freeze_bytes).hexdigest()
    after_sha = sha256(args.after)
    validate_protocol_selection(protocol, freeze_sha, after_sha)
    freeze = json.loads(freeze_bytes.decode('utf-8'))
    winner = freeze['winner']
    after_meta_path = args.after.with_suffix('.onnx.json')
    after_meta = json.loads(after_meta_path.read_text(encoding='utf-8'))
    if after_sha != winner['model_sha256'] or sha256(after_meta_path) != winner['metadata_sha256']:
        raise RuntimeError('After model/metadata differs from the frozen selected winner')
    point = winner['point']
    if (after_meta['pixel_threshold'],after_meta['minimum_component_pixels'],after_meta.get('hysteresis_growth_threshold')) != (point['high'],point['area'],point['growth']):
        raise RuntimeError('After serving policy differs from the frozen winner')
    if sha256(args.before) != protocol['before_model_sha256'] or sha256(args.before.with_suffix('.onnx.json')) != protocol['before_metadata_sha256']:
        raise RuntimeError('Before model/metadata differs from the preserved baseline')
    archive = ROOT/'datasets/factory/KolektorSDD2/KolektorSDD2.zip'
    if sha256(archive) != EXPECTED_SHA256:
        raise RuntimeError('Author archive bytes differ from the pinned release')
    cv2.setNumThreads(2)
    profiles = {tag: FactoryProfile({'defect_model_path':str(path.resolve()),
                'defect_backend':'auto', 'confirmation_frames':1})
                for tag,path in [('before',args.before),('after',args.after)]}
    captured = {}
    for tag,profile in profiles.items():
        original = profile._defect_model.predict
        def capture(frame, original=original, tag=tag):
            result = original(frame)
            captured[tag] = result[0]
            return result
        profile._defect_model.predict = capture
    records = {tag: [] for tag in profiles}
    raw_records = {tag: [] for tag in profiles}
    saved = {tag: {'defective':0,'normal_fp':0} for tag in profiles}
    journal_path = args.output.with_name(args.output.stem+'-opening-journal.json')
    journal = {'opening_requested_utc':datetime.now(timezone.utc).isoformat(),
               'archive_sha256':EXPECTED_SHA256,'winner_model_sha256':after_sha,
               'winner_metadata_sha256':winner['metadata_sha256'],
               'freeze_sha256':freeze_sha,
               'protocol_sha256':protocol_sha,'completed':False}
    if journal_path.exists():
        previous = json.loads(journal_path.read_text(encoding='utf-8'))
        for key in ('archive_sha256','winner_model_sha256','winner_metadata_sha256','freeze_sha256','protocol_sha256'):
            if previous.get(key) != journal[key]:
                raise RuntimeError('A previous official-test opening used different frozen inputs')
        journal['previous_opening_requested_utc'] = previous['opening_requested_utc']
    journal_path.parent.mkdir(parents=True,exist_ok=True)
    journal_path.write_text(json.dumps(journal,indent=2)+'\n',encoding='utf-8')
    with zipfile.ZipFile(archive) as source:
        names = set(source.namelist())
        images = sorted(name for name in names if PurePosixPath(name).parts[0]=='test'
                        and name.endswith('.png') and not name.endswith('_GT.png'))
        if len(images) != 1004:
            raise RuntimeError(f'Unexpected official test image count: {len(images)}')
        for index,name in enumerate(images):
            mask_name = name[:-4]+'_GT.png'
            raw,gt_raw = source.read(name),source.read(mask_name)
            frame = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
            gt = cv2.imdecode(np.frombuffer(gt_raw,np.uint8),cv2.IMREAD_GRAYSCALE)
            if frame is None or gt is None or frame.shape[:2] != gt.shape:
                raise RuntimeError(f'Invalid original image/mask geometry: {name}')
            truth = gt > 0
            for tag,profile in profiles.items():
                captured.pop(tag,None)
                result = profile.process(frame,FrameContext(index,time.perf_counter(),unit_id=name))
                abstained = result.status not in {'fault','review','clear','no_candidate'}
                model = profile._defect_model
                prediction = np.zeros(gt.shape,bool) if abstained else retained_mask(
                    captured[tag],model.pixel_threshold,model.area_threshold,gt.shape,
                    model.metadata.get('hysteresis_growth_threshold'))
                metrics = mask_metrics(prediction,truth)
                if not abstained:
                    assert metrics['flagged'] == (result.status in {'fault','review'})
                row = {'path':name,'sha256':hashlib.sha256(raw).hexdigest(),
                       'mask_sha256':hashlib.sha256(gt_raw).hexdigest(),
                       'defective':bool(truth.any()),'abstained':abstained,
                       'status':result.status,'message':result.title,
                       'mask_shape_hw':list(gt.shape),
                       'prediction_mask_sha256':hashlib.sha256(prediction.tobytes()).hexdigest(),**metrics,
                       **runtime_box_metrics(result.findings,truth)}
                kind = 'defective' if row['defective'] else 'normal_fp' if row['flagged'] else None
                if kind and saved[tag][kind] < 3:
                    row['visual_evidence'] = save_visual_evidence(args.output.parent/f'sdd2-{tag}-masks',
                        Path(name),frame,truth,prediction,metrics)
                    row['visual_evidence']['selection'] = 'First3 defective and first3 flagged normal images in sorted filename order, per model; all1004 images scored'
                    saved[tag][kind] += 1
                records[tag].append(row)
                # Predeclared diagnostic distinguishes domain/quality rejection from NN transfer.
                # It cannot be used as the deployed profile's result or select a replacement.
                if abstained:
                    model.predict(frame)
                raw_prediction = retained_mask(captured[tag],model.pixel_threshold,model.area_threshold,
                                               gt.shape,model.metadata.get('hysteresis_growth_threshold'))
                raw_records[tag].append({'path':name,'defective':bool(truth.any()),
                                         'abstained':False,**mask_metrics(raw_prediction,truth)})
            if (index+1)%100 == 0:
                print(f'Frozen paired SDD2 audit: {index+1}/{len(images)}',flush=True)
    results = {}
    for tag,rows in records.items():
        metrics = aggregate(rows)
        metrics['abstained_defective_views'] = sum(r['defective'] and r['abstained'] for r in rows)
        metrics['abstained_normal_views'] = sum(not r['defective'] and r['abstained'] for r in rows)
        metrics['non_abstained_views'] = sum(not r['abstained'] for r in rows)
        metrics['eligible_normal_views'] = sum(not r['defective'] and not r['abstained'] for r in rows)
        metrics['normal_false_alarm_rate_eligible_views'] = (metrics['normal_false_alarm_views']/metrics['eligible_normal_views']
                                                           if metrics['eligible_normal_views'] else None)
        results[tag] = metrics
    assert results['after']['defective_views']==110 and results['after']['normal_views']==894
    report = {'scope':'Fresh official SDD2 split, opened only after fixed-model selection; different source domain. '
                       'All 1004 images, with defective abstentions counted as misses. '
                       'Physical item identities unavailable; no independent factory or video claim. '
                       'NN execution times intentionally excluded from speed conclusions.',
              'protocol':protocol,'protocol_sha256':protocol_sha,
              'freeze_sha256':freeze_sha,'archive_sha256':EXPECTED_SHA256,
              'preprocessing':'Exact installed FactoryProfile, frozen quality/aspect gates and policy; '
                              'no test-driven crop/rotation/resize change. Retained score masks projected to original geometry.',
              'raw_diagnostic_scope':'Separate frozen-NN mask diagnostic bypasses quality/aspect rejection only; '
                                     'same original pixels, NN resize, threshold/area/growth. Not a deployed profile result, '
                                     'not used for model/threshold selection and cannot qualify production reliability.',
              'visual_selection':'First three defective images and first three flagged normal images in sorted filename order, per model; all records scored.',
              'models':{tag:p.describe() for tag,p in profiles.items()},
              'results':results,'records':records,
              'raw_diagnostic_results':{tag:aggregate(rows) for tag,rows in raw_records.items()},
              'raw_diagnostic_records':raw_records}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    journal.update(completed=True,completed_utc=datetime.now(timezone.utc).isoformat(),report_sha256=sha256(args.output))
    journal_path.write_text(json.dumps(journal,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
