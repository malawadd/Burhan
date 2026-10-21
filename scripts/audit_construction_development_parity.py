"""Read-only JSON audit: full recordings and selected CAL predictions agree.

Video encoding requires the isolated export environment, which has NumPy2.4.6;
serving/CAL has NumPy2.2.6. OpenCV5.0/OpenVINO2026.1 builds are identical.
Compare actual overlapping native-source predictions, not version assumptions.
"""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/evaluation/construction-round3'
STEMS=['evcivil-yolov8n-r4-head-640-gain','evcivil-yolov8n-r4-real-lowlight-640-gain']
CASES=['crack_89','crack_129','spalling_75','ros_crack_21']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(predictions):
    return [(p['label'],[round(float(v),6)for v in p['bbox']],round(float(p['score']),6))for p in predictions]


def main():
    results=[];before={}
    for stem in STEMS:
        cal_path=OUT/stem/'calibration-runtime-records.json'
        cal=json.loads(cal_path.read_text(encoding='utf8'));overlap=0
        for case in CASES:
            path=ROOT/'output'/f'construction-development-{stem}-{case}.json'
            report=json.loads(path.read_text(encoding='utf8'));frames=report['frames'];count=report['source']['original_camera_frames']
            if len(frames)!=count or [r['source_frame_index']for r in frames]!=list(range(count)) or report['derived_video']['encoded_frames']!=count or report['derived_video']['max_timestamp_error_s']!=0:
                raise RuntimeError('Complete source/encoded frame/PTS contract differs')
            identities=[{'index':r['source_frame_index'],'time':r['video_time_s'],'timestamp':r['source_timestamp_us'],
                         'raw':r['raw_native_bgr_sha256'],'input':r['model_input_native_bgr_sha256'],
                         'expected':r['expected'],'gain':r['input_exposure_gain']}for r in frames]
            if stem==STEMS[0]:before[case]=identities
            elif before[case]!=identities:raise RuntimeError('Before/after raw/native input/GT/exposure timeline differs')
            selected=[r for r in cal if r['source']==case]
            for row in selected:
                frame=frames[row['original_frame_index']]
                if signature(frame['predictions'])!=signature(row['predictions']):raise RuntimeError('Isolated export versus serving CAL prediction parity differs')
                if frame['input_exposure_gain']!=row['input_exposure_gain']:raise RuntimeError('Actual input gain differs across environments')
            overlap+=len(selected)
            results.append({'stem':stem,'sequence':case,'original_frames':count,'matching_calibration_predictions':len(selected),'report_sha256':sha(path)})
        if overlap!=48:raise RuntimeError(f'Expected16x3 overlapping declared CAL views; found{overlap}')
    result={'all_checks_passed':True,'scope':'Read-only actual JSON predictions at the fixed overlapping CAL sources; no NN/image decoding, no invented pixel ground truth or inference-speed claim.',
            'model_order':STEMS,'records':results,'all_source_timelines_raw_inputs_gained_inputs_gt_identical_before_after':True,
            'serving_vs_export_cal_prediction_signature_parity':'Exact label/order/bbox/score at1e-6 on48 shared sampled CAL sources per model; exposedcrack89 has no CAL overlap.',
            'runtime_versions':{'opencv':'5.0.0 both','openvino':'2026.1.0-21367-63e31528c62-releases/2026/1 both','numpy_serving':'2.2.6','numpy_export':'2.4.6'}}
    target=OUT/'four-development-video-environment-parity.json';target.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
