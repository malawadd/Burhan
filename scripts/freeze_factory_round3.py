"""Freeze the predeclared calibration-only winner before capacity/comparison.

This reads only completed training/calibration reports and model files, never
dataset comparison pixels. It performs no inference and changes no weights.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json

ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'output/optimization/factory-round3'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    candidates=[];protocols=[];inputs={}
    for variant in ['context','shallow']:
        stem=f'spatial_round3_{variant}'
        protocol_path=DIRECTORY/(stem+'-protocol-before-fit.json')
        report_path=DIRECTORY/(stem+'-calibration.json')
        protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
        report=json.loads(report_path.read_text(encoding='utf-8'))
        protocols.append(protocol)
        if sorted(row['step'] for row in report['reports'])!=[200,400,600]:
            raise ValueError(f'All predeclared {variant} checkpoints must finish calibration before freeze')
        inputs[variant]={'training_protocol_sha256':sha(protocol_path),'calibration_report_sha256':sha(report_path)}
        for row in report['reports']:
            if row['selected'] is None:continue
            point=row['selected']
            if point['normal_fp']>1:raise ValueError('Candidate exceeds the predeclared calibration false-alarm budget')
            path=Path(row['checkpoint']);meta_path=path.with_suffix('.onnx.json')
            metadata=json.loads(meta_path.read_text(encoding='utf-8'))
            if sha(path)!=row['model_sha256'] or metadata['model_sha256']!=row['model_sha256'] or sha(meta_path)!=row['metadata_sha256']:
                raise ValueError('Calibrated model or metadata changed before freeze')
            if metadata['pixel_threshold']!=point['high'] or metadata['minimum_component_pixels']!=point['area'] or metadata.get('hysteresis_growth_threshold')!=point['growth']:
                raise ValueError('Model policy differs from its selected calibration point')
            candidates.append({'variant':variant,'step':row['step'],'model_path':str(path),
                               'model_sha256':row['model_sha256'],'metadata_sha256':row['metadata_sha256'],'point':point})
    for key in ['seed','steps','save_at','warmstart_sha256','pretrained_sha256','split_items','raw_high_grid','area_grid','growth_grid']:
        if protocols[0][key]!=protocols[1][key]:raise ValueError(f'Controlled protocol differs at {key}')
    def rank(row):
        p=row['point']
        return p['meaningful_iou_0_1_recall'],-p['normal_fp'],p['micro_pixel_iou'],p['foreground_pixel_recall']
    ranked=sorted(candidates,key=rank,reverse=True)
    if not ranked:raise ValueError('No eligible calibrated candidate')
    value={'frozen_utc':datetime.now(timezone.utc).isoformat(),'selection':'Meaningful IoU>=.1 view recall, fewer normal FP, microIoU, foreground recall; same predeclared rule across both architectures',
           'selection_inputs':inputs,'winner':ranked[0],'ranked_candidates':ranked,
           'comparison_images_opened_for_selection':0,
           'capacity_gate':'Pending isolated complete-profile calibration capacity >15/s before original80 scoring; final end-to-end runtime qualification remains separate',
           'comparison_scope':'Original80 is already exposed development comparison, never blind or selection data',
           'official_sdd2_test_used':False,'weights_or_policy_changed_by_freeze':False}
    target=DIRECTORY/'winner-freeze-before-comparison.json'
    if target.exists():
        previous=json.loads(target.read_text(encoding='utf-8'))
        if previous['winner']!=value['winner'] or previous['selection_inputs']!=inputs:
            raise ValueError('A different winner/report is already frozen; preserve history')
        value=previous
    else:target.write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps({'freeze_report':str(target),'winner':value['winner'],'capacity_gate':value['capacity_gate']},indent=2))


if __name__=='__main__':main()
