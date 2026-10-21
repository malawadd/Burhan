"""L global all-three freeze only, no model inference and no comparison access."""
import json
from datetime import datetime,timezone
from factory_stfpm_bn_common import OUT,SAVE_AT,CAL_SHA,NAME,sha,identity,write_new,verify_protocol,protocol_path,model_path,rank,validate_point,validate_search_report,validate_bn_rows


def main():
    _,protocol=verify_protocol();target=OUT/'winner-frozen-before-comparison.json'
    from prepare_factory_stfpm_bn import verify_bn_fit
    fit_receipt,_,fit=verify_bn_fit(protocol)
    if target.exists():raise ValueError('Preserve existing global outcome')
    completed=OUT/'training-complete.json';complete=json.loads(completed.read_text(encoding='utf-8'))
    execution=OUT/'training-execution-before-optimizer.json';x=json.loads(execution.read_text(encoding='utf-8'))
    if complete['fixed_steps_completed']!=600 or [r['step'] for r in complete['history']]!=list(range(1,601)) or complete['save_at']!=SAVE_AT:
        raise ValueError('Complete fixed600/all3 required before globalfreeze')
    if not complete['teacher_unchanged'] or complete['teacher_state_sha256']!=x['initial_teacher_state_sha256']:
        raise ValueError('Frozen teacher drift')
    if complete['protocol_sha256']!=sha(protocol_path()) or complete['execution_receipt_sha256']!=sha(execution):raise ValueError('Completed fit lineage changed')
    for row in [x,complete]:
        if row['BN_fit_receipt_sha256']!=sha(fit_receipt) or row['BN_fitted_state_sha256']!=fit['state_sha256']:raise ValueError('Original BN fitting provenance changed')
    validate_bn_rows(x['initial_BN'],protocol['actual_BN_factory']['student_BN'],64)
    reports=[];choices=[];hashes={}
    for step in SAVE_AT:
        report_path=OUT/f'checkpoint{step}-calibration.json';r=json.loads(report_path.read_text(encoding='utf-8'));path=model_path(step)
        train_path=OUT/f'checkpoint{step}-training.json';train=json.loads(train_path.read_text(encoding='utf-8'))
        parity_path=OUT/f'{NAME}-step{step}-gpu-parity.json';parity=json.loads(parity_path.read_text(encoding='utf-8'))
        if r['step']!=step or r['protocol_sha256']!=sha(protocol_path()) or r['model_sha256']!=sha(path) or r['state_sha256']!=sha(path.with_suffix('.pt')):
            raise ValueError('Calibrated checkpoint source/protocol changed')
        if complete['checkpoint_reports'][str(step)]!=sha(train_path) or r['training_report_sha256']!=sha(train_path):raise ValueError('All3 training provenance changed')
        if train['teacher_state_sha256']!=complete['teacher_state_sha256'] or not train['teacher_unchanged']:raise ValueError('Teacher must remain identical for all3')
        export_path=OUT/f'{NAME}-step{step}-export.json'
        export=json.loads(export_path.read_text(encoding='utf-8'))
        for row in [train,export,parity]:
            if row['BN_fit_receipt_sha256']!=sha(fit_receipt) or row['BN_fitted_state_sha256']!=fit['state_sha256']:raise ValueError('Every fitted checkpoint must bind the same BN initialization')
        validate_bn_rows(train['BN_rows'],protocol['actual_BN_factory']['student_BN'],64+step)
        if (r['parity_report_sha256']!=sha(parity_path) or not parity['all_checks_passed'] or parity['model_sha256']!=sha(path)
                or parity['protocol_sha256']!=sha(protocol_path()) or parity['export_receipt_sha256']!=sha(export_path)):
            raise ValueError('Numeric parity changed')
        cache=OUT/f'checkpoint{step}-actualGPU-CAL-scoremaps.npz'
        if r['scorecache_sha256']!=sha(cache) or not r['actual_GPU_maps']:raise ValueError('ActualGPU map evidence changed')
        sources=r['calibration_source_identity']['ordered_sources']
        if identity(sources)!=CAL_SHA or r['calibration_source_identity']['sha256']!=CAL_SHA or len(sources)!=80 or sum(i['defective'] for i in sources)!=10:
            raise ValueError('Protected same80 CAL sources required')
        validate_search_report(r,protocol)
        point=r['selected'];hashes[str(step)]=sha(report_path);reports.append(r)
        if point is None:continue
        validate_point(point,protocol)
        meta=json.loads(path.with_suffix('.onnx.json').read_text(encoding='utf-8'))
        if r['metadata_sha256']!=sha(path.with_suffix('.onnx.json')) or meta['model_sha256']!=sha(path) or meta['training_protocol']!=protocol:
            raise ValueError('Selected metadata/source drift')
        if meta['calibration_selected_point']!=point or meta['calibration_source_identity_sha256']!=CAL_SHA or meta['output_logits'] is not False:
            raise ValueError('Actual boundedscore policy mismatch')
        if (meta['pixel_threshold'],meta['minimum_component_pixels'],meta['hysteresis_growth_threshold'])!=(point['high'],point['area'],point['growth']):
            raise ValueError('Serving point mismatch')
        choices.append({'step':step,'model_path':str(path),'model_sha256':sha(path),'metadata_sha256':r['metadata_sha256'],'point':point})
    choices.sort(key=lambda c:rank(c['point']),reverse=True)
    value={'frozen_utc':datetime.now(timezone.utc).isoformat(),'attempt':NAME,'accepted_for_independent_audit':bool(choices),
           'winner':choices[0] if choices else None,'ranked_candidates':choices,'calibration_report_sha256':hashes,
           'training_complete_sha256':sha(completed),'protocol_sha256':sha(protocol_path()),'all_three_required_and_complete':True,
           'BN_fit_receipt_sha256':sha(fit_receipt),'BN_fitted_state_sha256':fit['state_sha256'],
           'comparison_images_used_for_selection':0,'official_sdd2_test_still_sealed':True,
           'scope':'Global CAL selection only; rootONLYwinner originalaudit. No adoption/15FPS/production claim.'}
    write_new(target,value);print(json.dumps(value,indent=2))


if __name__=='__main__':main()
