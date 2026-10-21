"""Protect the predeclared attempt3 calibration winner; no inference or test IO."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json

ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'output/optimization/factory-round3'
NAME='spatial_round3d_packed_refine'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(row):
    p=row['point']
    return p['meaningful_iou_0_1_recall'],p['worst_positive_iou'],p['coverage_50pct_recall'],p['worst_positive_coverage'],p['micro_pixel_iou'],-p['normal_fp'],p['foreground_pixel_recall']


def main():
    protocol_path=DIRECTORY/(NAME+'-protocol-before-fit.json');report_path=DIRECTORY/(NAME+'-calibration.json')
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'));reports=json.loads(report_path.read_text(encoding='utf-8'))
    if sorted(r['step'] for r in reports['reports'])!=[200,400,600]:raise ValueError('All three planned attempt3 calibrations must finish')
    choices=[]
    for r in reports['reports']:
        point=r['selected']
        if point is None:continue
        if point['normal_fp']>1 or point['foreground_pixel_precision']<.65:raise ValueError('Candidate fails predeclared calibration guard')
        path=Path(r['checkpoint']);meta_path=path.with_suffix('.onnx.json');metadata=json.loads(meta_path.read_text(encoding='utf-8'))
        if sha(path)!=r['model_sha256'] or sha(meta_path)!=r['metadata_sha256'] or metadata['model_sha256']!=r['model_sha256']:raise ValueError('Artifact differs from calibrated hash')
        if (metadata['pixel_threshold'],metadata['minimum_component_pixels'],metadata.get('hysteresis_growth_threshold'))!=(point['high'],point['area'],point['growth']):raise ValueError('Calibrated policy changed')
        if metadata['input_shape']!=[1,3,1280,512] or metadata['output_shape']!=[1,1,1280,512]:raise ValueError('Graph description changed')
        choices.append({'step':r['step'],'model_path':str(path),'model_sha256':r['model_sha256'],'metadata_sha256':r['metadata_sha256'],'point':point})
    if not choices:raise ValueError('No attempt3 candidate met the precision/false-alarm guards')
    choices.sort(key=rank,reverse=True)
    value={'frozen_utc':datetime.now(timezone.utc).isoformat(),'attempt':3,'variant':'packed-pixelB','winner':choices[0],'ranked_candidates':choices,
           'selection_gate':protocol['selection_gate'],'selection_rank':protocol['selection_rank'],
           'protocol_sha256':sha(protocol_path),'calibration_report_sha256':sha(report_path),
           'preserved_attempt1_freeze_sha256':sha(DIRECTORY/'winner-freeze-before-comparison.json'),
           'preserved_attempt2_freeze_sha256':sha(DIRECTORY/'winner-attempt2-freeze-before-comparison.json'),
           'comparison_images_used_for_selection':0,'official_sdd2_test_still_sealed':True,
           'capacity_gate':'Provisional contended full-profile calibration>15/s may permit accuracy-only original80; final isolated paired capacity and sustained runtime remain mandatory',
           'scope':'Original80 already-exposed development comparison; no blind-test or production assurance claim'}
    target=DIRECTORY/'winner-attempt3-packed-freeze-before-comparison.json'
    if target.exists():
        old=json.loads(target.read_text(encoding='utf-8'))
        if old['winner']!=value['winner'] or old['calibration_report_sha256']!=value['calibration_report_sha256']:raise ValueError('Different attempt3 winner already frozen')
        value=old
    else:target.write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps({'freeze_report':str(target),'winner':value['winner'],'capacity_gate':value['capacity_gate']},indent=2))


if __name__=='__main__':main()
