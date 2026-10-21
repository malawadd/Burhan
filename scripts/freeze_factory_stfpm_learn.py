"""L7 all-three terminal outcomes: numeric exclusions or completed guarded CAL."""
import json
from datetime import datetime,timezone
from factory_stfpm_learn_common import (OUT,NAME,SAVE_AT,CAL_SHA,sha,identity,write_new,verify_protocol,protocol_path,
    model_path,rank,validate_point,validate_search_report,validate_bn_rows,require_complete_training,validate_numeric_receipt)

def validate_terminal(outcome,parity):
    validate_numeric_receipt(parity)
    if outcome['step']!=parity['step']:raise ValueError('Terminal checkpoint mismatch')
    if outcome['status']=='numeric_failed_no_CAL':
        if parity['all_checks_passed'] or outcome['CAL_inputs_opened']!=0 or 'calibration_report_sha256' in outcome:raise ValueError('Numeric exclusion must open zero CAL inputs')
    elif outcome['status']=='calibration_complete':
        if not parity['all_checks_passed'] or outcome['CAL_inputs_opened']!=80 or not outcome.get('calibration_report_sha256'):raise ValueError('CAL requires passing numeric guard and complete80')
    else:raise ValueError('Every checkpoint needs an explicit terminal outcome')

def main():
    _,protocol=verify_protocol();target=OUT/'winner-frozen-before-comparison.json'
    if target.exists():raise ValueError('Preserve first global freeze')
    complete_path,complete=require_complete_training(protocol)
    from prepare_factory_stfpm_learn import verify_bn_fit
    fit_receipt,_,fit=verify_bn_fit(protocol)
    execution=OUT/'training-execution-before-optimizer.json';x=json.loads(execution.read_text(encoding='utf-8'))
    if complete['teacher_state_sha256']!=x['initial_teacher_state_sha256']:raise ValueError('Teacher lineage changed')
    for r in [complete,x]:
        if r['BN_fit_receipt_sha256']!=sha(fit_receipt) or r['BN_fitted_state_sha256']!=fit['state_sha256']:raise ValueError('BN initialization changed')
    if x['protocol_sha256']!=sha(protocol_path()) or x['initial_teacher_state_sha256']!=fit['teacher_sha256']:raise ValueError('Exact initial teacher/protocol changed')
    validate_bn_rows(x['initial_BN'],protocol['actual_BN_factory']['student_BN'],64)
    choices=[];outcomes={}
    for step in SAVE_AT:
        path=model_path(step);terminal_path=OUT/f'checkpoint{step}-terminal-outcome.json'
        t=json.loads(terminal_path.read_text(encoding='utf-8'))
        train_path=OUT/f'checkpoint{step}-training.json';train=json.loads(train_path.read_text(encoding='utf-8'))
        export_path=OUT/f'{NAME}-step{step}-export.json';export=json.loads(export_path.read_text(encoding='utf-8'))
        parity_path=OUT/f'{NAME}-step{step}-gpu-parity.json';parity=json.loads(parity_path.read_text(encoding='utf-8'))
        validate_terminal(t,parity)
        for row in [t,export,parity]:
            if row['protocol_sha256']!=sha(protocol_path()) or row['model_sha256']!=sha(path):raise ValueError('Terminal/export/parity source changed')
        if (t['step']!=step or t['parity_report_sha256']!=sha(parity_path) or t['training_report_sha256']!=sha(train_path)
            or train['state_sha256']!=sha(path.with_suffix('.pt')) or train['onnx_sha256']!=sha(path)
            or train['execution_receipt_sha256']!=sha(execution) or not train['teacher_unchanged']
            or train['teacher_state_sha256']!=complete['teacher_state_sha256']):raise ValueError('Checkpoint training lineage changed')
        if parity['export_receipt_sha256']!=sha(export_path) or export['reference_sha256']!=sha(OUT/f'{NAME}-step{step}-torch-reference.npz'):raise ValueError('Export/reference drift')
        for row in [train,export,parity]:
            if row['BN_fit_receipt_sha256']!=sha(fit_receipt) or row['BN_fitted_state_sha256']!=fit['state_sha256']:raise ValueError('Checkpoint BN lineage drift')
        validate_bn_rows(train['BN_rows'],protocol['actual_BN_factory']['student_BN'],64+step)
        outcomes[str(step)]={'status':t['status'],'terminal_sha256':sha(terminal_path)}
        if t['status']=='numeric_failed_no_CAL':
            if (OUT/f'checkpoint{step}-calibration.json').exists() or (OUT/f'checkpoint{step}-actualGPU-CAL-scoremaps.npz').exists():raise ValueError('Numeric-failed checkpoint cannot have CAL evidence')
            if sha(path.with_suffix('.onnx.json'))!=export['initial_metadata_sha256']:raise ValueError('Excluded metadata changed')
            continue
        report_path=OUT/f'checkpoint{step}-calibration.json';r=json.loads(report_path.read_text(encoding='utf-8'))
        if (t['calibration_report_sha256']!=sha(report_path) or r['model_sha256']!=sha(path) or r['state_sha256']!=sha(path.with_suffix('.pt'))
            or r['protocol_sha256']!=sha(protocol_path()) or r['parity_report_sha256']!=sha(parity_path)
            or r['training_report_sha256']!=sha(train_path) or not r['actual_GPU_maps']
            or r['scorecache_sha256']!=sha(OUT/f'checkpoint{step}-actualGPU-CAL-scoremaps.npz')):raise ValueError('Completed CAL identity changed')
        sources=r['calibration_source_identity']['ordered_sources']
        if len(sources)!=80 or sum(s['defective'] for s in sources)!=10 or identity(sources)!=CAL_SHA or r['calibration_source_identity']['sha256']!=CAL_SHA:raise ValueError('Protected CAL80 drift')
        validate_search_report(r,protocol);point=r['selected']
        if point is None:
            if sha(path.with_suffix('.onnx.json'))!=export['initial_metadata_sha256']:raise ValueError('Ineligible metadata drift')
            continue
        validate_point(point,protocol);meta_path=path.with_suffix('.onnx.json');meta=json.loads(meta_path.read_text(encoding='utf-8'))
        if (r['metadata_sha256']!=sha(meta_path) or meta['model_sha256']!=sha(path) or meta['training_protocol']!=protocol
            or meta['calibration_selected_point']!=point or meta['calibration_source_identity_sha256']!=CAL_SHA
            or meta['output_logits'] is not False or meta['openvino_device']!='GPU' or meta['openvino_inference_precision']!='f16'
            or meta['openvino_disable_winograd_convolution'] is not True or 'openvino_cpu_threads' in meta):raise ValueError('Actual frozen deployment metadata differs')
        if (meta['pixel_threshold'],meta['minimum_component_pixels'],meta['hysteresis_growth_threshold'])!=(point['high'],point['area'],point['growth']):raise ValueError('Frozen serving point drift')
        if sha(OUT/f'checkpoint{step}-initial-metadata.json')!=export['initial_metadata_sha256']:raise ValueError('Preserved initial metadata differs')
        choices.append({'step':step,'model_path':str(path),'model_sha256':sha(path),'metadata_sha256':sha(meta_path),'point':point})
    choices.sort(key=lambda r:rank(r['point']),reverse=True)
    value={'frozen_utc':datetime.now(timezone.utc).isoformat(),'attempt':NAME,'accepted_for_independent_audit':bool(choices),
        'winner':choices[0] if choices else None,'ranked_candidates':choices,'all_three_terminal_outcomes':outcomes,
        'all_three_required_and_complete':True,'training_complete_sha256':sha(complete_path),'protocol_sha256':sha(protocol_path()),
        'comparison_images_used_for_selection':0,'official_sdd2_test_still_sealed':True,
        'scope':'Only CAL-ranked GPU-numeric-passing checkpoint may reach root exposed audit. NO speed/adoption claim; fitted>=15 and sustained runtime still mandatory.'}
    write_new(target,value);print(json.dumps(value,indent=2))

if __name__=='__main__':main()
