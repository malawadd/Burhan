"""Frozen L9 mixed-adapter capacity/runtime only; no model or point selection.

Does not call historical live preparation verifiers: their fixture/test state is
historical evidence. Serving math/runtime identities and frozen artifacts remain
strictly pinned. All execution requires a separate root quiet-window release.
Source-only draft: final fixtures/peer review/plan have not been recorded.
"""
import argparse,json,sys,time,hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
OLD=ROOT/'output/optimization/factory-round3l9-learned-partitioned'
FREEZE=OLD/'winner-frozen-before-root-capacity.json'
OUT=ROOT/'output/optimization/factory-round3l9-winner-capacity'
PLAN=OUT/'plan-before-first64-and-runtime.json'
INITIAL=ROOT/'output/optimization/factory-round3l9-partitioned/protocol-before-numeric-and-conditional-all3.json'
INITIAL_SHA='522a1f549ca08e964fa46c2787f32370da0b4787dba3fe1ab3809efc1f87b32c'
SERVING=['scripts/calibrate_factory_partitioned_learned.py','scripts/factory_partitioned_segmentation.py',
         'scripts/factory_stfpm_partition_math.py','scripts/prepare_factory_stfpm_partition.py',
         'scripts/factory_tapped_segmentation.py','edge_runtime/profiles/factory.py',
         'edge_runtime/profiles/factory_defect.py','edge_runtime/profiles/factory_encoder.py']
PINS=[*SERVING,'scripts/benchmark_factory_partitioned_winner.py','scripts/benchmark_runtime.py',
      'scripts/diagnose_factory_stfpm_stages.py','edge_runtime/runtime.py','edge_runtime/sources.py',
      'edge_runtime/contracts.py','edge_runtime/buffer.py','edge_runtime/media.py','edge_runtime/storage.py',
      'tests/test_factory_partitioned_winner_benchmark.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def frozen_binding():
    frozen=read(FREEZE);protocol_path=OLD/'protocol-before-all3-transforms.json';protocol=read(protocol_path)
    if (frozen['all_three_required_and_complete'] is not True or frozen['winner'] is None
            or frozen['protocol_sha256']!=sha(protocol_path) or frozen['accepted_for_independent_audit'] is not False):raise ValueError('One completed fixed global winner required')
    winner=frozen['winner'];path=Path(winner['model_path']);meta=read(path.with_suffix('.onnx.json'));step=winner['step']
    if (sha(path)!=winner['model_sha256'] or sha(path.with_suffix('.onnx.json'))!=winner['metadata_sha256']
            or meta['calibration_selected_point']!=winner['point'] or meta['model_sha256']!=sha(path)
            or (meta['pixel_threshold'],meta['minimum_component_pixels'],meta['hysteresis_growth_threshold'])
               !=(winner['point']['high'],winner['point']['area'],winner['point']['growth'])
            or meta['partitioned_score_head']!='GPUfeatures_plus_CPU_Numpy_float32' or meta['output_logits'] is not False):raise ValueError('Actualselectedmodel/sidecar/point drift')
    archive=OLD/'executed-source-snapshot-before-transform/manifest.json';a=read(archive)
    if a['protocol_sha256']!=sha(protocol_path):raise ValueError('Executed source archive differs')
    recorded={row['path']:row['sha256'] for row in a['sources']}
    # Preserve historical prep/test receipts without requiring their live files.
    for relative in SERVING:
        if recorded.get(relative)!=protocol['source_pins'].get(relative) or sha(ROOT/relative)!=recorded[relative]:raise ValueError('Executedserving source changed: '+relative)
        archived=archive.parent/relative
        if sha(archived)!=recorded[relative]:raise ValueError('Archivedserving bytes changed')
    terminal_paths=[]
    for candidate in [200,400,600]:
        tpath=OLD/f'checkpoint{candidate}-terminal.json';t=read(tpath)
        if (frozen['all_three_terminal_outcomes'][str(candidate)]['terminal_sha256']!=sha(tpath)
                or t['protocol_sha256']!=sha(protocol_path)):raise ValueError('All3 terminal freeze changed')
        terminal_paths.append(tpath)
    numeric=OLD/f'checkpoint{step}-numeric.json';cal=OLD/f'checkpoint{step}-calibration.json';r=read(cal);n=read(numeric)
    terminal=read(OLD/f'checkpoint{step}-terminal.json');cache=OLD/f'checkpoint{step}-actual-mixed-scoremaps.npz'
    if (terminal['status']!='calibration_complete' or terminal['numeric_sha256']!=sha(numeric) or terminal['calibration_sha256']!=sha(cal)
            or r['selected']!=winner['point'] or r['numeric_sha256']!=sha(numeric) or r['scorecache_sha256']!=sha(cache)
            or n['all_checks_passed'] is not True or n['model_sha256']!=sha(path) or r['model_sha256']!=sha(path)):
        raise ValueError('Selected numeric/CAL/cache identities changed')
    paths=[FREEZE,protocol_path,archive,*terminal_paths,numeric,cal,cache,path,path.with_suffix('.onnx.json')]
    sources=r['calibration_source_identity']['ordered_sources']
    if len(sources)!=80 or sum(v['defective'] for v in sources)!=10:raise ValueError('Protected CAL80 source inventory required')
    return {'winner':winner,'model_path':str(path),'identities':{str(p.relative_to(ROOT)):sha(p) for p in paths},
            'source_archive_sha256':sha(archive),'calibration_sources':sources,'model_split':meta['split_items']}


def plan_value():
    if sha(INITIAL)!=INITIAL_SHA:raise ValueError('Original16normalTRAIN protocol changed')
    binding=frozen_binding();fixture=OUT/'preparation-fixtures.json';f=read(fixture)
    if f['all_checks_passed'] is not True:raise ValueError('Pure benchmark fixture pass required')
    for relative,digest in f['source_pins'].items():
        if sha(ROOT/relative)!=digest:raise ValueError('Benchmark fixture source drift')
    return {'binding':binding,'initial_protocol_sha256':INITIAL_SHA,'normal16_sources':read(INITIAL)['capacity']['real_sources'],
        'capacity':{'initialization_calls':3,'real_warmup_calls':4,'measured_calls':64,'minimum_mean_calls_per_second':15,
            'all64must_infer':True,'policy':'Exactgloballyselectedfittedpoint; no alternative checkpoint/threshold','all_stage_and_total_tails':True},
        'runtime':{'script':'scripts/benchmark_runtime.py','requested_seconds':60,'input_FPS':16,'expected_frames':960,'deadline_ms':100,'preview_FPS':6,
            'source_partition':'calibration','source_count':80,'source_kind':'Predecodedrealpublicstills cycling; never a physicalproductionvideo/coverage claim',
            'initialization':'ExistingEdgeRuntime3zero workerwarmups beforecapture; firstcapturedrealframesretained, no retiming',
            'pass':'Actualcaptured=processed=960; tracedsequence1..960exactonce; dropped/source_skipped/inspection_errors/coverage_gaps=0; deadline late=0. Reportallfailures/tails without changingclock.'},
        'order':'Rootquietfirst64; FAIL=>noruntime/original. PASS=>rootmayrelease originalaudit orruntime in eitherorder. Runtime mustpass beforeacceptance.',
        'override':'Onlyprocess-localrt.create_profile returns verifiedbuild_profile with mixed adapter; restoreallmodulebindings at exit. No liveprepverify/testsdependency.',
        'source_pins':{p:sha(ROOT/p) for p in PINS},'fixture_sha256':sha(fixture),'official_SDD2_test_still_sealed':True}


def verify_plan():
    value=read(PLAN)
    if value!=plan_value():raise ValueError('Frozenwinner/benchmarkplan/source changed')
    return value


def summarize_ms(values):
    x=np.asarray(values,dtype=np.float64)
    if len(x)!=64 or not np.isfinite(x).all() or (x<=0).any():raise ValueError('Exactly64valid firsttimings required')
    return {'mean':float(x.mean()),'p50':float(np.median(x)),'p95':float(np.percentile(x,95)),
            'p99':float(np.percentile(x,99)),'max':float(x.max()),'serial_calls_per_second':1000/float(x.mean())}


def capacity_acceptance(records,audit):
    from factory_stfpm_learn_common import gpu_configuration_passes
    times=summarize_ms([r['profile_ms'] for r in records])
    passed=(all(r.get('neural_stage_ms') is not None for r in records) and audit.get('actual_configuration_passed') is True
            and gpu_configuration_passes(audit.get('properties',{})) and times['serial_calls_per_second']>=15)
    return bool(passed),times


def capacity(plan,scope):
    import cv2
    from edge_runtime.contracts import FrameContext
    from calibrate_factory_partitioned_learned import build_profile
    from diagnose_factory_stfpm_stages import configuration_audit
    path=Path(plan['binding']['model_path']);target=OUT/'selected-winner-first64-capacity.json';attempt=OUT/'capacity-attempt.json'
    if target.exists() or attempt.exists():raise ValueError('Preserve firstselectedcapacity attempt')
    write_new(attempt,{'plan_sha256':sha(PLAN),'model_sha256':sha(path),'scope':scope})
    frames=[];rows=plan['normal16_sources']
    if len(rows)!=16:raise ValueError('Exactly16predeclarednormalTRAIN views')
    for row in rows:
        source=ROOT/'datasets/factory/KolektorSDD'/row['image']
        if sha(source)!=row['image_sha256']:raise ValueError('ProtectednormalTRAIN image changed')
        frame=cv2.imread(str(source))
        if frame is None:raise ValueError('ProtectednormalTRAIN decode failed')
        frames.append(frame)
    started=time.perf_counter();profile=build_profile(path);profile.prepare_worker();startup=time.perf_counter()-started
    if profile._defect_model.neural.initialization_report['warmup_iterations']!=3:raise ValueError('Fixed3initializationcalls required')
    for i,frame in enumerate(frames[:4]):profile.process(frame,FrameContext(i,time.perf_counter(),unit_id=f'L9selectedwarm{i}'))
    profile.reset();records=[]
    for i in range(64):
        started=time.perf_counter();result=profile.process(frames[i%16],FrameContext(i,started,unit_id=f'L9selected{i}'))
        records.append({'source':rows[i%16]['image'],'profile_ms':1000*(time.perf_counter()-started),'status':result.status,
            'neural_stage_ms':result.metrics.get('factory_neural_stage_ms'),'partition_stage_ms':profile._defect_model.neural.request.last_timing.copy()})
    audit=configuration_audit(profile._defect_model.neural.compiled,'GPU');passed,times=capacity_acceptance(records,audit)
    report={'plan_sha256':sha(PLAN),'model_sha256':sha(path),'metadata_sha256':sha(path.with_suffix('.onnx.json')),'startup_seconds':startup,
        'records':records,'profile_ms':times,'compiled_provenance':audit,'capacity_gate_passed':passed,'scope':scope,
        'policy':plan['binding']['winner']['point'],'architecture':'ActualGPU6rawfeatures+CPUf32head+inheritedprofilepostprocess','sourcepins_unchanged_after':verify_plan()==plan}
    write_new(target,report);print(json.dumps({k:v for k,v in report.items() if k not in ['records','compiled_provenance']},indent=2))
    if not passed:raise ValueError('Selectedfittedcapacity<15 or othergatefailed; noruntime/original')


def capacity_passed(plan):
    path=OUT/'selected-winner-first64-capacity.json';r=read(path);model=Path(plan['binding']['model_path'])
    passed,times=capacity_acceptance(r['records'],r['compiled_provenance'])
    if (r['plan_sha256']!=sha(PLAN) or r['model_sha256']!=sha(model) or r['metadata_sha256']!=sha(model.with_suffix('.onnx.json'))
            or r['capacity_gate_passed']!=passed or r['profile_ms']!=times):raise ValueError('First64identity/gate changed')
    if not passed:raise ValueError('No runtime/original afterfailedcapacity')
    return path


def runtime_acceptance(report):
    state=report['state'];counts=state['metrics'];trace=report['diagnostic_trace'];sequences=[r['sequence'] for r in trace]
    checks={'source_clock':'requested_seconds' in report and report['requested_seconds']==60 and report['requested_fps']==16,
        'captured960':counts['captured']==960,'processed960':counts['processed']==960,
        'no_holes_or_duplicate_profile_calls':sequences==list(range(1,961)),
        'no_drops_source_skips_errors_gaps':all(counts[k]==0 for k in ['dropped','source_skipped','inspection_errors','coverage_gaps']),
        'no_deadline_misses':counts['late']==0,'eof_without_error':state['state']=='eof' and state['error'] is None}
    return all(checks.values()),checks


def runtime(plan,scope):
    import edge_runtime.runtime as rt
    import benchmark_runtime as bench
    from calibrate_factory_partitioned_learned import build_profile
    from factory_partitioned_segmentation import FactoryPartitionedSegmentation
    capacity_path=capacity_passed(plan);target=OUT/'selected-winner-runtime16-60s.json';attempt=OUT/'runtime-attempt.json'
    if target.exists() or attempt.exists():raise ValueError('Preserve first60secondruntime attempt')
    path=Path(plan['binding']['model_path']);write_new(attempt,{'plan_sha256':sha(PLAN),'capacity_sha256':sha(capacity_path),'scope':scope})
    # Preserve established source clock/runtime/queue/media/storage behavior.
    # The wrapper changes only profile creation and diagnostic stage collection.
    original_create=rt.create_profile;original_source=rt.LocalSource;stages=[];created=[]
    def mixed_profile(name,options):
        if name!='factory' or Path(options['defect_model_path']).resolve()!=path.resolve():raise ValueError('Unexpectedruntime profile/model')
        profile=build_profile(path)
        if not isinstance(profile._defect_model.neural,FactoryPartitionedSegmentation):raise ValueError('Runtime mustuse mixedadapter')
        original_process=profile.process
        def process(frame,context):
            result=original_process(frame,context)
            stages.append({'sequence':context.sequence,'partition_stage_ms':profile._defect_model.neural.request.last_timing.copy()})
            return result
        profile.process=process;created.append(profile);return profile
    try:
        with patch.object(rt,'create_profile',mixed_profile):
            bench.main(['--model',str(path),'--backend','auto','--varied','--partition','calibration','--fps','16','--seconds','60',
                        '--deadline','100','--preview-fps','6','--trace','--output',str(target)])
    finally:
        rt.LocalSource=original_source
    if rt.create_profile is not original_create or len(created)!=1:raise ValueError('Runtime constructor override/restoration failed')
    report=read(target);passed,checks=runtime_acceptance(report)
    expected=plan['binding']['calibration_sources']
    if report['input_sha256']!=[r['image_sha256'] for r in expected]:raise ValueError('RuntimeCALsource order/identity changed')
    result={'plan_sha256':sha(PLAN),'raw_runtime_report_sha256':sha(target),'capacity_sha256':sha(capacity_path),'model_sha256':sha(path),
        'metadata_sha256':sha(path.with_suffix('.onnx.json')),'runtime_gate_passed':passed,'checks':checks,'actual_partition_stages':stages,
        'actual_mixed_adapter':True,'sourcepins_unchanged_after':verify_plan()==plan,'scope':scope,
        'source_note':'Publicrealstillclockedreplayonly; no movingproductionvideo/physicallinecoverage claim',
        'all_late_calls':[r for r in report['diagnostic_trace'] if r['host_to_profile_return_ms']>100]}
    write_new(OUT/'runtime-gate-and-partition-stages.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['actual_partition_stages','all_late_calls']},indent=2))
    if not passed:raise ValueError('Actual60secondruntime gatefailed; retainallcounters/tails')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['record-plan','capacity','runtime'],required=True)
    parser.add_argument('--root-quiet-released',action='store_true');parser.add_argument('--contention-note',default='Rootconfirmedallprojectcomputequiet; ordinaryhostservicesuncontrolled');args=parser.parse_args()
    if args.stage=='record-plan':write_new(PLAN,plan_value());print(json.dumps({'plan_sha256':sha(PLAN),'NN_executed':False}));return
    if not args.root_quiet_released:raise ValueError('Explicitrootquietrelease required')
    import cv2
    cv2.setNumThreads(1);plan=verify_plan()
    try:(capacity if args.stage=='capacity' else runtime)(plan,args.contention_note)
    except Exception as error:
        failure=OUT/f'first-failure-{args.stage}.json'
        if not failure.exists():write_new(failure,{'plan_sha256':sha(PLAN),'error':str(error),'no_retry':True})
        raise


if __name__=='__main__':main()
