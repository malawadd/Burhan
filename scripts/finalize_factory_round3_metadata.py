"""Correct descriptive NN geometry before freeze; preserve all old metadata.

No inference, checkpoint/policy change or comparison-image access occurs.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json

ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'output/optimization/factory-round3'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    history=DIRECTORY/'metadata-geometry-finalization.json'
    if history.exists():
        saved=json.loads(history.read_text(encoding='utf-8'))
        for row in saved['models']:
            if sha(Path(row['metadata']))!=row['new_metadata_sha256']:raise ValueError('Finalized metadata changed')
        print('Geometry metadata already finalized and hashes verified');return
    if (DIRECTORY/'winner-freeze-before-comparison.json').exists():raise ValueError('Cannot change frozen metadata')
    # Reject an early invocation before modifying any sidecar.
    for variant in ['context','shallow']:
        report=json.loads((DIRECTORY/f'spatial_round3_{variant}-calibration.json').read_text(encoding='utf-8'))
        if sorted(row['step'] for row in report['reports'])!=[200,400,600]:raise ValueError('Wait for all planned calibrations')
    changes=[];reports=[]
    for variant in ['context','shallow']:
        report_path=DIRECTORY/f'spatial_round3_{variant}-calibration.json'
        data=json.loads(report_path.read_text(encoding='utf-8'))
        if sorted(row['step'] for row in data['reports'])!=[200,400,600]:raise ValueError('Wait for all planned calibrations')
        old_report_sha=sha(report_path)
        for row in data['reports']:
            if row['selected'] is None:continue
            model=Path(row['checkpoint']);path=model.with_suffix('.onnx.json')
            old=path.read_bytes();old_sha=hashlib.sha256(old).hexdigest()
            if old_sha!=row['metadata_sha256'] or sha(model)!=row['model_sha256']:raise ValueError('Artifact changed before metadata finalization')
            metadata=json.loads(old.decode('utf-8'))
            policy=(metadata['pixel_threshold'],metadata['minimum_component_pixels'],metadata.get('hysteresis_growth_threshold'))
            metadata.update(input_width=512,input_height=1280,input_shape=[1,3,1280,512],
                            output_shape=[1,1,640,256],model_grid_width=256,model_grid_height=640,
                            architecture=('MobileNetV3Small blocks0:7 encoder; trainable ImageNet blocks7:12 context; 96-to-40 residual fusion at stride16; depthwise spatial U-Net skip decoder; stride2 logits' if variant=='context' else
                                          'MobileNetV3Small blocks0:7 encoder; depthwise spatial U-Net skip decoder at strides2/8/16; stride2 logits'),
                            metadata_geometry_note='Legacy input_width256 denoted analysis grid; explicit NN input512x1280 and output256x640 now distinguished. Descriptive correction only; model and policy unchanged.')
            assert policy==(metadata['pixel_threshold'],metadata['minimum_component_pixels'],metadata.get('hysteresis_growth_threshold'))
            (DIRECTORY/f'{model.stem}-metadata-before-geometry-finalization.json').write_bytes(old)
            path.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
            row['metadata_sha256']=sha(path)
            changes.append({'metadata':str(path),'old_metadata_sha256':old_sha,'new_metadata_sha256':row['metadata_sha256'],'model_sha256':row['model_sha256'],'decision_policy_changed':False})
        report_path.write_text(json.dumps(data,indent=2),encoding='utf-8')
        reports.append({'report':str(report_path),'old_sha256':old_report_sha,'new_sha256':sha(report_path)})
    history.write_text(json.dumps({'recorded_utc':datetime.now(timezone.utc).isoformat(),'models':changes,'reports':reports,'model_weights_changed':False,'comparison_images_opened':0},indent=2),encoding='utf-8')
    print(json.dumps({'history':str(history),'models_finalized':len(changes)},indent=2))


if __name__=='__main__':main()
