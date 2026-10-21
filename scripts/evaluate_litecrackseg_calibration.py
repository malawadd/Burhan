"""Preregister then audit a local thin-crack specialist on unchanged calibration only."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge_runtime.profiles.construction_lite import LiteCrackConstructionModel, decode_crack_map, camera_exposure
from evaluate_construction_round3 import BASE, sampled, score_frame, summarize

OUT = ROOT / 'output/evaluation/construction-round3/litecrackseg-calibration'
GRID = [.10, .15, .20, .25, .35, .45, .55, .65, .75, .85]
EXPOSURE_POLICY = 'off'
ORIGINAL_OUT = OUT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol():
    rows = sampled('calibration')
    return {'schema_version': 1, 'frozen_before_specialist_forward': True,
            'scope': 'Unchanged original fit/calibration split only; no heldout or exposed recording predictions',
            'architecture': 'Author LiteCrackSeg TUT; pinned local source and tensor-only checkpoint',
            'source_revision': '082d33111f1a3b9ac1d78e984b06fd3530bf3533',
            'checkpoint_sha256': 'c12fb1ce991e479a32e5a26d7648ce9d20a440b9af4fd0411d321aa6a2eed497',
            'initial_full_frame_square_size': 384,
            **({'exposure_policy':EXPOSURE_POLICY,
                 'exposure_rule':'If original native grayscale p99<64, multiply all BGR by min(8,128/max(p99,1)), clip0..255 and uint8 truncation; otherwise byte-identical input',
                 'original_calibration_protocol_sha256':digest(ORIGINAL_OUT/'preregistered-protocol.json'),
                 'cache_rule':'Unchanged native input + identical model uses prior probability map; all122 images still scored. Changed inputs recomputed. Original candidate artifacts remain intact',
                 'known_gate_limitation':'Original negative FPR is1.0; any unchanged flagged negative stays a failure. This is a low-light diagnostic, not presumed release success'} if EXPOSURE_POLICY!='off' else {}),
            'preprocessing': 'Actual author path: RGB float0..255; no /255 and no ImageNet normalization; fullframe stretch only',
            'threshold_grid': GRID, 'minimum_native_component_pixels': 4, 'connectivity': 8,
            'projection': 'Bilinear probability map to native frame; strict probability>threshold',
            'postprocessing': 'No morphology, dilation, closing, GT-dependent crop, fragment grouping or box inflation; maximum100 mean-score-ranked components',
            'matching_iou_threshold': .5,
            'localization_scope': 'Derived component bounding-box IoU proxy versus publisher boxes; no pixel mask IoU/recall is available for ev-CIVIL',
            'thin_geometry': 'Native GT box short side<=30 pixels and long side>=2*short side; not physical crack width',
            'selection': 'Maximize crack localization recall under precision>=.8 and negative-image FPR<=.1; if none pass, bestF1 is retained explicitly FAILED, not a deployable fix',
            'spalling': 'Unsupported by this specialist; never relabel crack probabilities as spalling',
            'timing': 'Accuracy run may overlap CPU training; no capacity or real-time claim',
            'source_manifest_sha256': digest(BASE/'exposures.json'),
            'background_import_sha256': digest(BASE/'background-import.json'),
            'calibration_images': len(rows), 'negative_images': sum(r['negative'] for r in rows),
            'ordered_image_hashes': [{'source': r['source'], 'image': str(r['image'].relative_to(ROOT)),
                                      'sha256': digest(r['image'])} for r in rows],
            'license': 'Whole-model/source license undeclared; experimental evaluation only; commercial and redistribution clearance unresolved'}


def frozen_protocol():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT/'preregistered-protocol.json'
    expected = protocol()
    if target.exists():
        if json.loads(target.read_text(encoding='utf8')) != expected:
            raise RuntimeError('Frozen protocol or calibration source pixels changed')
    else:
        target.write_text(json.dumps(expected, indent=2), encoding='utf8')
    return target


def thin(box):
    w, h = box['bbox'][2:]
    return min(w, h) <= 30 and max(w, h) >= 2 * min(w, h)


def audit(path, engine, device):
    proto = frozen_protocol()
    meta = json.loads(path.with_suffix('.provenance.json').read_text(encoding='utf8'))
    if digest(path) != meta['sha256']:
        raise RuntimeError('Export hash changed')
    model = LiteCrackConstructionModel({'engine': engine, 'openvino_device': device, 'cpu_threads': 1,
                                       'exposure_policy':EXPOSURE_POLICY}, meta)
    rows = sampled('calibration')
    maps = []; exposure_records=[]
    original_maps = np.load(ORIGINAL_OUT/f'{path.stem}-probabilities.npz')['probabilities'] if EXPOSURE_POLICY!='off' else None
    for index, row in enumerate(rows):
        frame = cv2.imread(str(row['image']))
        _,gain = camera_exposure(frame,EXPOSURE_POLICY)
        use_cache=original_maps is not None and gain==1.0
        maps.append(original_maps[index] if use_cache else model.predict_maps(frame))
        exposure_records.append({'index':index,'image':str(row['image'].relative_to(ROOT)),'gain':gain,'probabilities_reused_byte_identical_input':use_cache})
        if index % 20 == 0:
            print(f'Calibration probability maps {index+1}/{len(rows)}', flush=True)
    np.savez_compressed(OUT/f'{path.stem}-probabilities.npz', probabilities=np.stack(maps))
    (OUT/'exposure-records.json').write_text(json.dumps(exposure_records,indent=2),encoding='utf8')
    candidates = []
    all_records = {}
    for threshold in GRID:
        scored = []
        for row, probability in zip(rows, maps):
            frame = cv2.imread(str(row['image']))
            predictions = decode_crack_map(probability, frame.shape[:2], threshold, 4)
            # Same class-annotated frame scope; spall-only publisher frames are
            # unsupported/unscored, not assumed crack-free. True negatives score.
            expected = [e for e in row['expected'] if e['label'] == 'crack']
            record = score_frame(predictions, expected, {'crack'}, negative=row['negative'])
            record.update(source=row['source'], frame_index=row['frame_index'], negative=row['negative'])
            scored.append(record)
        summary = summarize(scored)['by_class']['crack']
        p, r = summary['precision'] or 0, summary['recall'] or 0
        negatives = [r for r in scored if r['negative']]
        fpr = sum(bool(r['predictions']) for r in negatives) / len(negatives)
        candidates.append({'threshold': threshold, **summary, 'negative_image_fpr': fpr,
                           'f1': 2*p*r/(p+r) if p+r else 0,
                           'precision_and_negative_gate': p >= .8 and fpr <= .1})
        all_records[threshold] = scored
    passed = [c for c in candidates if c['precision_and_negative_gate']]
    selected = max(passed, key=lambda c: (c['recall'] or 0, c['precision'] or 0, c['threshold'])) if passed else max(candidates, key=lambda c: (c['f1'], c['precision'] or 0, c['threshold']))
    records = all_records[selected['threshold']]
    # Thin recall uses full one-to-one matching, then selects native geometric
    # thin GT; other annotations cannot be discarded to make a component match.
    thin_count = thin_matches = 0
    for record in records:
        thin_ids = {e['id'] for e in record['expected'] if thin(e)}
        thin_count += len(thin_ids)
        thin_matches += sum(p.get('matched_annotation_id') in thin_ids for p in record['predictions'])
    selected['thin_geometry_annotations'] = thin_count
    selected['thin_geometry_localized'] = thin_matches
    selected['thin_geometry_recall'] = thin_matches/thin_count if thin_count else None
    selected['status'] = 'calibration_gate_passed' if passed else 'FAILED_calibration_gate'
    summary = {'model': model.describe(), 'protocol_sha256': digest(proto), 'selected': selected,
               'threshold_candidates': candidates, 'pixel_metrics': None,
               'pixel_metric_limitation': 'Publisher has boxes, not pixel masks; saved model masks are predictions, never ground truth',
               'mask_and_box_scope': 'Actual connected segmentation regions; component boxes are only a localization proxy',
               'validation_scope': 'Calibration only, correlated video frames; author pretraining overlap unknown; no heldout/exposed inference',
               'timing_scope': 'CPU fit may overlap; no speed/capacity result'}
    prefix = OUT/path.stem
    prefix.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    prefix.with_suffix('.records.json').write_text(json.dumps(records, indent=2), encoding='utf8')
    selected_masks = []
    preview_ids = []
    for index, (row, probability, record) in enumerate(zip(rows, maps, records)):
        frame = cv2.imread(str(row['image']))
        native = cv2.resize(probability, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        components=decode_crack_map(probability,frame.shape[:2],selected['threshold'],4,masks=True)
        retained=np.zeros(frame.shape[:2],dtype=np.uint8)
        for component in components:retained[component['mask']]=1
        selected_masks.append(retained)
        # Fixed first crack/spall/negative plus first four crack frames: no
        # outcome selection and no heldout inputs. GT and MODEL are separate.
        group = 'negative' if row['negative'] else 'crack' if any(e['label']=='crack' for e in row['expected']) else 'spalling_unsupported'
        if preview_ids.count(group) >= (4 if group=='crack' else 1):
            continue
        preview_ids.append(group)
        gt = frame.copy(); prediction = frame.copy()
        for e in row['expected']:
            x,y,w,h = map(round,e['bbox']); cv2.rectangle(gt,(x,y),(x+w,y+h),(255,255,0),1)
        mask = selected_masks[-1].astype(bool)
        prediction[mask] = (prediction[mask]*.35 + np.array([0,0,255])*.65).astype(np.uint8)
        panel = np.hstack([frame,gt,prediction])
        panel = cv2.copyMakeBorder(panel,35,0,0,0,cv2.BORDER_CONSTANT,value=(25,25,25))
        for j, caption in enumerate(['RAW CAMERA','PUBLISHER BOXES (not mask GT)','MODEL CRACK MASK']):
            cv2.putText(panel,caption,(j*frame.shape[1]+5,23),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1)
        cv2.imwrite(str(OUT/f'{path.stem}-{group}-{index}.jpg'),panel)
    np.savez_compressed(OUT/f'{path.stem}-selected-native-masks.npz', **{f'frame_{i}':m for i,m in enumerate(selected_masks)})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-only', action='store_true')
    parser.add_argument('--model', type=Path, default=ROOT/'models/construction/thin-contingency/litecrackseg-tut-384.onnx')
    parser.add_argument('--engine', choices=['onnxruntime','openvino'], default='openvino')
    parser.add_argument('--device', choices=['CPU','GPU'], default='GPU')
    parser.add_argument('--exposure-policy',choices=['off','low_light_gain'],default='off')
    args = parser.parse_args()
    EXPOSURE_POLICY=args.exposure_policy
    if EXPOSURE_POLICY!='off':
        OUT=ORIGINAL_OUT/'low-light-gain'
    if args.freeze_only:
        print(frozen_protocol())
    else:
        audit(args.model.resolve(), args.engine, args.device)
