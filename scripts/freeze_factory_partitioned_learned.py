"""All3 learned mixed-backend outcomes, ranking only completed numeric-pass CAL."""
import json
from datetime import datetime,timezone
from prepare_factory_partitioned_learned import OUT,STEPS,PROTOCOL,model_path,sha,write_new,verify_protocol,gpu_configuration_passes
from calibrate_factory_partitioned_learned import transformed,terminal_numeric_pass
from factory_stfpm_common import identity,CAL_SHA,rank,validate_search_report

def validate_CAL_config(audit):
    if audit.get('actual_configuration_passed') is not True or not gpu_configuration_passes(audit.get('properties',{})):
        raise ValueError('Actual CAL configuration failed')

def validate_terminal(terminal,numeric):
    if terminal['step']!=numeric['step']:raise ValueError('Terminal step differs')
    passed=terminal_numeric_pass(numeric)
    if terminal['status']=='numeric_failed_no_CAL':
        if passed or terminal['CAL_inputs_opened']!=0 or 'calibration_sha256' in terminal:raise ValueError('Numeric failure must exclude all CAL')
    elif terminal['status']=='calibration_complete':
        if not passed or terminal['CAL_inputs_opened']!=80 or not terminal.get('calibration_sha256'):raise ValueError('Completed CAL requires passing numeric/full80')
    else:raise ValueError('Every checkpoint needs explicit terminal outcome')

def main():
    protocol=verify_protocol();target=OUT/'winner-frozen-before-root-capacity.json'
    if target.exists():raise ValueError('Preserve global result')
    choices=[];outcomes={}
    for step in STEPS:
        path,ref,transform_path,t=transformed(step,protocol,selected=True)
        numeric_path=OUT/f'checkpoint{step}-numeric.json';numeric=json.loads(numeric_path.read_text())
        terminal_path=OUT/f'checkpoint{step}-terminal.json';terminal=json.loads(terminal_path.read_text())
        validate_terminal(terminal,numeric)
        if (numeric['step']!=step or numeric['protocol_sha256']!=sha(PROTOCOL) or numeric['model_sha256']!=sha(path)
            or numeric['initial_metadata_sha256']!=t['initial_metadata_sha256'] or numeric['reference_sha256']!=sha(ref)
            or numeric['transform_receipt_sha256']!=sha(transform_path) or terminal['protocol_sha256']!=sha(PROTOCOL)
            or terminal['numeric_sha256']!=sha(numeric_path)):raise ValueError('Numeric/terminal exact checkpoint lineage changed')
        outcomes[str(step)]={'status':terminal['status'],'terminal_sha256':sha(terminal_path)}
        meta_path=path.with_suffix('.onnx.json')
        if terminal['status']=='numeric_failed_no_CAL':
            if any((OUT/name).exists() for name in [f'checkpoint{step}-CAL-attempt.json',f'checkpoint{step}-calibration.json',f'checkpoint{step}-actual-mixed-scoremaps.npz']):raise ValueError('Excluded numeric checkpoint cannot have CAL')
            if sha(meta_path)!=t['initial_metadata_sha256']:raise ValueError('Excluded sidecar changed')
            continue
        report_path=OUT/f'checkpoint{step}-calibration.json';r=json.loads(report_path.read_text())
        if (terminal['calibration_sha256']!=sha(report_path) or r['numeric_sha256']!=sha(numeric_path)
            or r['protocol_sha256']!=sha(PROTOCOL) or r['model_sha256']!=sha(path) or r['transform_receipt_sha256']!=sha(transform_path)
            or r['actual_partitioned_maps'] is not True or r['execution']!='GPU raw features + CPU NumPyfloat32 score head'
            or r['scorecache_sha256']!=sha(OUT/f'checkpoint{step}-actual-mixed-scoremaps.npz')):raise ValueError('Mixed CAL/source/cache lineage changed')
        validate_CAL_config(r['compiled_provenance'])
        sources=r['calibration_source_identity']['ordered_sources']
        if (len(sources)!=80 or sum(s['defective'] for s in sources)!=10 or identity(sources)!=CAL_SHA
            or r['calibration_source_identity']['sha256']!=CAL_SHA):raise ValueError('Protected CAL80 identity changed')
        validate_search_report(r,protocol);point=r['selected']
        if point is None:
            if sha(meta_path)!=t['initial_metadata_sha256']:raise ValueError('Unselected metadata changed')
            continue
        meta=json.loads(meta_path.read_text())
        if (r['metadata_sha256']!=sha(meta_path) or sha(OUT/f'checkpoint{step}-initial-metadata.json')!=t['initial_metadata_sha256']
            or meta['model_sha256']!=sha(path) or meta['partitioned_protocol_sha256']!=sha(PROTOCOL)
            or meta['partitioned_score_head']!='GPUfeatures_plus_CPU_Numpy_float32' or meta['all_outputs_retained']!=6
            or meta['split_items']!=protocol['split_items'] or meta['openvino_device']!='GPU'
            or meta['openvino_inference_precision']!='f16' or meta['openvino_disable_winograd_convolution'] is not True
            or meta['output_logits'] is not False or meta['calibration_selected_point']!=point
            or meta['calibration_source_identity_sha256']!=CAL_SHA):raise ValueError('Frozen mixed deployment metadata changed')
        if (meta['pixel_threshold'],meta['minimum_component_pixels'],meta['hysteresis_growth_threshold'])!=(point['high'],point['area'],point['growth']):raise ValueError('Frozen point differs')
        choices.append({'step':step,'model_path':str(path),'model_sha256':sha(path),'metadata_sha256':sha(meta_path),'point':point})
    choices.sort(key=lambda row:rank(row['point']),reverse=True)
    result={'frozen_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(PROTOCOL),
        'all_three_terminal_outcomes':outcomes,'all_three_required_and_complete':True,'winner':choices[0] if choices else None,
        'ranked_candidates':choices,'accepted_for_root_fitted_capacity':bool(choices),'accepted_for_independent_audit':False,
        'original_audit_prerequisite':'Root fresh fitted-policy complete-profile>=15 PASS required first; use bound standalone mixed adapter, never default shared runtime',
        'production_qualified':False,'sustained_runtime_prerequisite':'Actual fitted16-inputFPS/end-to-end counts and tails required beforeacceptance',
        'actual_score_backend':'GPU raw features + CPU NumPyfloat32 score head','comparison_images_used_for_selection':0,'official_SDD2_test_still_sealed':True}
    write_new(target,result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
