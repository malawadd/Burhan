"""Root-only frozen round4 benchmarks; preparation is metadata-only, NN opt-in.

No training, point search, runtime monkeypatch, process suspension or test opening.
Original-mask scoring delegates to the existing independent auditor only after
the selected fitted policy passes its first64 complete-profile capacity gate.
"""
import argparse,hashlib,json,math,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
R4=ROOT/'output/optimization/factory-round4-supervised'
OUT=ROOT/'output/optimization/factory-round4-root-benchmark'
PLAN=OUT/'plan-before-any-benchmark.json'
FIXTURE=OUT/'source-preparation-fixtures-v4-post-CAL-exploratory.json'
NORMAL_PROTOCOL=ROOT/'output/optimization/factory-round3l9-partitioned/protocol-before-numeric-and-conditional-all3.json'
NORMAL_SHA='522a1f549ca08e964fa46c2787f32370da0b4787dba3fe1ab3809efc1f87b32c'
PROPOSAL_SHA='89adfa6e24de1c4dd3b2fbe13c2644b3948dc521cec520e840ceadf9ab7d1a2e'
CAL_SHA='7b09d8cbf39129dad144fcad6ecbad0ade0bb19e39de6e8fbdfc634eb6ed6d07'
RECOVERY=R4/'CAL-identity-v2-correction-before-CAL.json'
RECOVERY_DECISION=R4/'parent-approved-CAL-identity-v2-decision.json'
RECOVERY_WRAPPER=ROOT/'scripts/evaluate_factory_round4_cal_identity_v2.py'
RECOVERY_REFERENCE=ROOT/'output/optimization/factory-round3l9-learned-partitioned/checkpoint200-calibration.json'
ORIGINAL_EVALUATOR_SHA='34bec29e967394e46a0f7503501749244659589723299d6155e3dda71afce160'
EXPLORATORY_NAME='lower_alarm_exploratory'
EXPLORATORY_DIR=ROOT/'output/optimization/factory-round4-lower-alarm'
EXPLORATORY_PROPOSAL_SHA='02d6260ed04337370e03115dc484092e2a2c852d514a8bf29d6a78fc3782577f'
EXPLORATORY_FREEZE_SHA='1e7ae79cc097f7e0bd1bd5938eda690cc76524e64d9e621a097fdcb761263bc1'
ARMS={'primary':1,'secondary':3}
PINS=['scripts/benchmark_factory_round4_root.py','scripts/benchmark_runtime.py','scripts/factory_localization_audit.py',
 'scripts/train_factory_surface.py','edge_runtime/profiles/factory.py','edge_runtime/profiles/factory_defect.py',
 'edge_runtime/profiles/factory_encoder.py','edge_runtime/runtime.py','edge_runtime/sources.py',
 'edge_runtime/contracts.py','edge_runtime/buffer.py','edge_runtime/media.py','edge_runtime/storage.py',
 'tests/test_factory_round4_root_benchmark.py']

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while b:=f.read(1024*1024):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def write_new(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def historical_cal_identity(value):
 # Protected7b09... was recorded with default JSON separators, including spaces.
 # This serialization check changes no source path/hash/shape/order/content.
 return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def policy(meta):return {k:meta[k] for k in ['pixel_threshold','minimum_component_pixels','hysteresis_growth_threshold']}

def recovery_binding(protocol,freeze):
 # Verify durable executed bytes, not old live preparation/test dependencies.
 r=read(RECOVERY);decision=read(RECOVERY_DECISION);digest=sha(RECOVERY)
 rows=r['ordered_CAL_sources'];reference=read(RECOVERY_REFERENCE)
 if (decision.get('approved_CAL_identity_metadata_only_recovery') is not True
     or decision.get('original_evaluator_sha256')!=ORIGINAL_EVALUATOR_SHA
     or r['parent_decision_sha256']!=sha(RECOVERY_DECISION) or r['original_evaluator_sha256']!=ORIGINAL_EVALUATOR_SHA
     or sha(ROOT/'scripts/evaluate_factory_round4.py')!=ORIGINAL_EVALUATOR_SHA
     or sha(RECOVERY_REFERENCE)!=r['reference_report_sha256']
     or rows!=reference['calibration_source_identity']['ordered_sources'] or historical_cal_identity(rows)!=CAL_SHA
     or len(rows)!=80 or sum(v['defective'] for v in rows)!=10
     or r['historical_CAL_identity_sha256']!=CAL_SHA or r['CAL_source_count']!=80 or r['CAL_defective_count']!=10
     or r['current_ordered_CAL_content_exactly_equals_protected_reference'] is not True
     or r['all_three_numeric_PASS_preserved'] is not True or r['CAL_model_initializations_before_correction']!=0
     or r['original_fit_protocol_unchanged'] is not True or r['no_new_training'] is not True):
  raise ValueError('Exact metadata-only protected CAL serialization recovery required')
 wrapper_key=str(RECOVERY_WRAPPER.relative_to(ROOT))
 wrapper_digest=r['source_pins'][wrapper_key]
 if decision.get('wrapper_sha256')!=wrapper_digest or sha(RECOVERY_WRAPPER)!=wrapper_digest or freeze.get('CAL_identity_v2_correction_sha256')!=digest or freeze.get('CAL_identity_v2_wrapper_source_sha256')!=wrapper_digest:
  raise ValueError('Global freeze recovery wrapper/receipt lineage differs')
 archive=ROOT/r['executed_source_archive'];manifest=archive/'manifest.json';a=read(manifest)
 if sha(manifest)!=r['executed_source_archive_manifest_sha256'] or a['source_pins']!=r['source_pins'] or a['original_fit_protocol_sha256']!=sha(protocol):
  raise ValueError('Executed CAL recovery source archive differs')
 paths=[RECOVERY,RECOVERY_DECISION,RECOVERY_WRAPPER,RECOVERY_REFERENCE,manifest]
 for name,expected in a['source_pins'].items():
  path=archive/name
  if sha(path)!=expected:raise ValueError('Executed CAL recovery source bytes changed')
  paths.append(path)
 for name,expected in r['failure_archive_pins'].items():
  path=ROOT/name
  if sha(path)!=expected:raise ValueError('Original pre-CAL failure archive changed')
  paths.append(path)
 if set(r['preserved_numeric_reports'])!={'300','600','900'}:raise ValueError('All3 preserved numeric identities required')
 for step in [300,600,900]:
  path=R4/f'checkpoint{step}-numeric.json'
  if sha(path)!=r['preserved_numeric_reports'][str(step)] or read(path)['all_checks_passed'] is not True:raise ValueError('Preserved numeric PASS changed')
  paths.append(path)
 return {'receipt_sha256':digest,'wrapper_source_sha256':wrapper_digest,'protected_ordered_CAL_sources':rows},paths

def exploratory_binding(original_freeze):
 proposal_path=EXPLORATORY_DIR/'post-CAL-exploratory-proposal-before-any-benchmark.json'
 frozen_path=EXPLORATORY_DIR/'post-CAL-lower-alarm-candidate-freeze.json'
 decision_path=EXPLORATORY_DIR/'parent-approved-post-CAL-exploratory-decision.json'
 if sha(proposal_path)!=EXPLORATORY_PROPOSAL_SHA or sha(frozen_path)!=EXPLORATORY_FREEZE_SHA:raise ValueError('Exact separately approved POST-CAL exploratory comparison required')
 p,f,d=read(proposal_path),read(frozen_path),read(decision_path)
 if (d.get('approved_post_CAL_lower_alarm_exploratory_comparison') is not True or d['proposal_sha256']!=sha(proposal_path)
     or any(f.get(k)!=v for k,v in p.items()) or f['proposal_sha256']!=sha(proposal_path) or f['parent_decision_sha256']!=sha(decision_path)
     or f['model_name']!=EXPLORATORY_NAME or f['accepted_for_original_audit'] is not False or f['production_qualified'] is not False
     or f['binding']['original_global_freeze_sha256']!=sha(original_freeze)
     or f['original_prospective_two_arms_unchanged'] is not True or f['all_three_candidate_outcomes_mandatory'] is not True
     or f['new_threshold_searches']!=0 or f['new_fits']!=0 or f['new_source_scoring']!=0):raise ValueError('Exploratory approval/scope/original-freeze lineage differs')
 archive=ROOT/f['executed_source_archive'];manifest=archive/'manifest.json'
 if sha(manifest)!=f['executed_source_manifest_sha256'] or read(manifest)['source_pins']!=f['source_pins'] or read(manifest)['proposal_sha256']!=sha(proposal_path):raise ValueError('Exploratory executed source archive differs')
 paths=[proposal_path,frozen_path,decision_path,manifest]
 for name,digest in f['source_pins'].items():
  path=archive/name
  if sha(path)!=digest:raise ValueError('Exploratory archived source bytes changed')
  paths.append(path)
 for name,digest in f['binding']['artifact_pins'].items():
  path=ROOT/name
  if sha(path)!=digest:raise ValueError('Exploratory original all3 candidate artifacts changed')
  paths.append(path)
 candidate=f['binding']['candidate']
 if f['accepted_for_root_first64_capacity']!=(candidate is not None):raise ValueError('Exploratory capacity eligibility differs')
 return candidate,paths

def binding():
 protocol=R4/'protocol-before-fit.json';freeze=R4/'winner-frozen-before-root-capacity.json'
 if sha(protocol)!=PROPOSAL_SHA:raise ValueError('Exact approved round4 protocol required')
 p=read(protocol);f=read(freeze)
 if f['protocol_sha256']!=sha(protocol) or f['all_three_required_and_complete'] is not True or f['both_arms_frozen_before_any_original'] is not True or list(f['arms'])!=list(ARMS):raise ValueError('Complete fixed all3/both-arm freeze required')
 completion=R4/'training-complete.json';done=read(completion)
 if done['protocol_sha256']!=sha(protocol) or done['transfer_steps']!=4096 or done['fine_steps']!=900 or done['saved_all_three']!=[300,600,900]:raise ValueError('All4996 fixed updates must be complete')
 recovery,recovery_paths=recovery_binding(protocol,f)
 paths=[protocol,freeze,completion,R4/'executed-sources-before-optimizer/manifest.json']
 archive=read(paths[-1])
 if archive['protocol_sha256']!=sha(protocol) or archive['source_pins']!=p['source_pins']:raise ValueError('Executed source archive binding differs')
 for relative,digest in archive['source_pins'].items():
  archived=paths[-1].parent/relative
  if sha(archived)!=digest:raise ValueError('Executed source archive bytes changed')
 paths.extend(recovery_paths)
 # Historical prep/test live files are NOT prerequisites. Only actual serving
 # code remains live-pinned here; final benchmark sources are pinned separately.
 if sha(ROOT/'edge_runtime/profiles/factory_encoder.py')!=p['source_pins']['edge_runtime/profiles/factory_encoder.py']:raise ValueError('Executed serving encoder changed')
 for step in [300,600,900]:
  terminal=R4/f'checkpoint{step}-terminal.json';t=read(terminal)
  if t['protocol_sha256']!=sha(protocol) or f['all_three_terminal_outcomes'][str(step)]['terminal_sha256']!=sha(terminal):raise ValueError('Three terminal outcome identities differ')
  paths.append(terminal)
 models={}
 for name,path in [('v10h',ROOT/'models/factory/spatial_unet_v10h_sdd2-step300.onnx'),('C200',ROOT/'models/factory/spatial_round3e_phase_max_refine-step200.onnx')]:
  meta=read(path.with_suffix('.onnx.json'));models[name]={'path':str(path),'sha256':sha(path),'metadata_sha256':sha(path.with_suffix('.onnx.json')),'policy':policy(meta),'split_items':meta['split_items'],'role':'retained historical comparator'}
 if models['v10h']['sha256']!='dd5019cbcbc859c4eeb994a7639fb0c520ee3e3175e47a3f1544ec4c5dd60753' or models['v10h']['metadata_sha256']!='25d8d250c3abc0deab82994c3cc48f6f5d356058bbd74ab4903b9cd497f1ba78':raise ValueError('Original v10h baseline changed')
 cfreeze=ROOT/'output/optimization/factory-round3/winner-attempt3-phase-max-freeze-before-comparison.json';c=read(cfreeze)['winner'];paths.append(cfreeze)
 if (models['C200']['sha256'],models['C200']['metadata_sha256'])!=(c['model_sha256'],c['metadata_sha256']):raise ValueError('Original C200 comparator changed')
 eligible=[];cal_sources=None
 for arm,budget in ARMS.items():
  entry=f['arms'][arm]
  if entry['normal_FP_budget']!=budget:raise ValueError('Prospective arm budget differs')
  w=entry['winner']
  if w is None:continue
  if w['point']['normal_fp']>budget or w['point']['foreground_pixel_precision']<.65:raise ValueError('Frozen winner fails its declared arm guards')
  step=w['step'];model=Path(w['model_path']);meta=read(model.with_suffix('.onnx.json'));cal=R4/f'checkpoint{step}-calibration.json';r=read(cal)
  numeric=R4/f'checkpoint{step}-numeric.json';n=read(numeric);terminal=read(R4/f'checkpoint{step}-terminal.json')
  if (terminal['status']!='calibration_complete' or terminal['calibration_sha256']!=sha(cal) or terminal['numeric_sha256']!=sha(numeric)
      or n['all_checks_passed'] is not True or r['numeric_sha256']!=sha(numeric) or r['protocol_sha256']!=sha(protocol)
      or r['scorecache_sha256']!=sha(R4/f'checkpoint{step}-scorecache.npz') or r['arms'][arm]['selected']!=w['point']
      or r['arms'][arm]['deployment']['model_path']!=str(model)):raise ValueError('Selected actual numeric/CAL/cache/terminal lineage differs')
  if (sha(model)!=w['model_sha256'] or sha(model.with_suffix('.onnx.json'))!=w['metadata_sha256'] or meta['model_sha256']!=sha(model)
      or meta['calibration_selected_point']!=w['point'] or meta['round4_policy_arm']!=arm or meta['round4_normal_FP_budget']!=budget
      or policy(meta)!={'pixel_threshold':w['point']['high'],'minimum_component_pixels':w['point']['area'],'hysteresis_growth_threshold':w['point']['growth']}):raise ValueError('Selected weights/metadata/frozen policy differs')
  current=r['calibration_source_identity']['ordered_sources']
  if len(current)!=80 or sum(v['defective'] for v in current)!=10 or historical_cal_identity(current)!=CAL_SHA or r['calibration_source_identity']['sha256']!=CAL_SHA:raise ValueError('Protected calibration input identity differs')
  if current!=recovery['protected_ordered_CAL_sources'] or r.get('CAL_identity_v2_correction_sha256')!=recovery['receipt_sha256'] or r.get('CAL_identity_v2_wrapper_source_sha256')!=recovery['wrapper_source_sha256']:raise ValueError('Actual CAL report recovery lineage differs')
  if cal_sources is not None and current!=cal_sources:raise ValueError('Both arms must use identical fixed CAL inputs')
  cal_sources=current;eligible.append(arm);models[arm]={'path':str(model),'sha256':sha(model),'metadata_sha256':sha(model.with_suffix('.onnx.json')),'policy':policy(meta),'split_items':meta['split_items'],'step':step,'role':'CAL-frozen round4 '+arm}
  paths.extend([cal,numeric,R4/f'checkpoint{step}-scorecache.npz'])
 if f['accepted_for_root_fitted_capacity']!=bool(eligible) or f['accepted_for_independent_audit'] is not False:raise ValueError('Selected prospective freeze capacity/accuracy eligibility differs')
 exploratory,extra_paths=exploratory_binding(freeze);paths.extend(extra_paths)
 if exploratory is not None:
  step=exploratory['step'];r=read(R4/f'checkpoint{step}-calibration.json');deployment=r['arms']['primary']['deployment'];point=r['arms']['primary']['selected']
  if (exploratory!={'step':step,'point':point,**deployment} or r['calibration_source_identity']['ordered_sources']!=cal_sources
      or r['scorecache_sha256']!=sha(R4/f'checkpoint{step}-scorecache.npz') or r['protocol_sha256']!=sha(protocol)
      or r.get('CAL_identity_v2_correction_sha256')!=recovery['receipt_sha256'] or r.get('CAL_identity_v2_wrapper_source_sha256')!=recovery['wrapper_source_sha256']):raise ValueError('Exploratory candidate must be exact preselected PRIMARY deployment/cache on same CAL80')
  model=Path(exploratory['model_path']);meta=read(model.with_suffix('.onnx.json'))
  if (sha(model)!=exploratory['model_sha256'] or sha(model.with_suffix('.onnx.json'))!=exploratory['metadata_sha256'] or meta['model_sha256']!=sha(model)
      or meta['calibration_selected_point']!=point or meta['round4_policy_arm']!='primary' or meta['round4_normal_FP_budget']!=1
      or policy(meta)!={'pixel_threshold':point['high'],'minimum_component_pixels':point['area'],'hysteresis_growth_threshold':point['growth']}):raise ValueError('Exploratory exact existing PRIMARY weights/metadata/policy differs')
  models[EXPLORATORY_NAME]={'path':str(model),'sha256':sha(model),'metadata_sha256':sha(model.with_suffix('.onnx.json')),'policy':policy(meta),'split_items':meta['split_items'],'step':step,'role':'POST-CAL exploratory lower-alarm comparison; not prospective confirmatory'}
  eligible.append(EXPLORATORY_NAME);paths.extend([model,model.with_suffix('.onnx.json'),R4/f'checkpoint{step}-scorecache.npz'])
 if any(v['split_items']!=p['split_items'] for v in models.values()):raise ValueError('Comparators/winners differ from protected physical item partitions')
 return {'models':models,'eligible_round4_arms':eligible,'calibration_sources':cal_sources,'CAL_identity_recovery':recovery,'artifact_pins':{str(v.relative_to(ROOT)):sha(v) for v in paths}}

def prepare():
 fixture=read(FIXTURE)
 if fixture['all_checks_passed'] is not True or fixture['tests']!=7:raise ValueError('Reviewed7 pure benchmark fixtures required')
 for relative,digest in fixture['source_pins'].items():
  if sha(ROOT/relative)!=digest:raise ValueError('Benchmark preparation fixture source changed')
 if sha(NORMAL_PROTOCOL)!=NORMAL_SHA:raise ValueError('Fixed normal16 source protocol changed')
 b=binding();normal=read(NORMAL_PROTOCOL)['capacity']['real_sources']
 if len(normal)!=16 or any(r['defective'] is not False for r in normal):raise ValueError('Exact16 real normal TRAIN sources required')
 train=set(b['models']['C200']['split_items']['train'])
 if any(Path(r['image'].replace('\\','/')).parts[0] not in train for r in normal):raise ValueError('Normal source outside protected TRAIN')
 plan={'binding':b,'normal16_sources':normal,'normal_source_protocol_sha256':NORMAL_SHA,'preparation_fixture_sha256':sha(FIXTURE),'source_pins':{r:sha(ROOT/r) for r in PINS},
  'status':'READY_FOR_ROOT_QUIET_RELEASE' if b['eligible_round4_arms'] else 'NO_ELIGIBLE_ROUND4_WINNER_NO_BENCHMARK',
  'capacity':{'init_zero_calls':3,'real_warmups':4,'first_measured_calls':64,'normal_source_cycle':16,'minimum_mean_full_profile_FPS':15,'all64must_infer':True},
  'paired':{'partition':'calibration','views':80,'repeats':3,'order':'At view i/repeat r rotate model order by(i+r)%n; every model gets same frame, no extra model calls or favorable snapshot selection'},
  'runtime':{'fps':16,'seconds':60,'expected_submitted_and_processed':960,'deadline_ms':100,'preview_fps':6,'zero_drop_skip_error_gap_late_required':True},
  'accuracy':'Root existing factory_localization_audit.py on exposed80 only after EACH learned policy passes first64; saveALL11defects+ALLflaggednormals with raw/GT/model/difference. No checkpoint/point/arm selection from original.',
  'scope':'Capacity excludes capture/queue/UI/storage; separate runtime includes existing edge capture/queue/evidence path. CAL/original are reused development, not blind. Clocked real-still replay is not a physical moving-line video.',
  'quiet_rule':'Root must confirm all project fits, exports, NN, data I/O, media encoding and tests exited; station stopped/browser playback paused. Ordinary host services remain uncontrolled; no process suspension/termination.',
  'first_result_rule':'Every initial attempt/tail/failure is preserved. No clock/retry/fps/source/config/threshold changes. No losing checkpoint fallback. Failed arm is ineligible, other arm remains independently reported.'}
 write_new(PLAN,plan);print(json.dumps({'plan':str(PLAN),'sha256':sha(PLAN),'eligible_arms':b['eligible_round4_arms']},indent=2))

def verify_plan():
 p=read(PLAN)
 if p['binding']!=binding() or sha(NORMAL_PROTOCOL)!=p['normal_source_protocol_sha256'] or sha(FIXTURE)!=p['preparation_fixture_sha256']:raise ValueError('Frozen plan/artifact/model binding changed')
 for relative,digest in p['source_pins'].items():
  if sha(ROOT/relative)!=digest:raise ValueError('Frozen benchmark/serving source changed: '+relative)
 if not p['binding']['eligible_round4_arms']:raise ValueError('No eligible new winner; no benchmark/original workload')
 return p

def summarize(values,expected=None):
 import numpy as np
 if any(type(v) not in (float,int) for v in values):raise ValueError('Timings must be actual numbers, not booleans')
 x=np.asarray(values,dtype=np.float64)
 if not len(x) or expected is not None and len(x)!=expected or not np.isfinite(x).all() or (x<=0).any():raise ValueError('All positive finite first-call timings required')
 return {'count':len(x),'mean_ms':float(x.mean()),'p50_ms':float(np.median(x)),'p95_ms':float(np.percentile(x,95)),'p99_ms':float(np.percentile(x,99)),'max_ms':float(x.max()),'serial_full_profile_FPS':1000/float(x.mean())}
def runtime_checks(report):
 m=report['state']['metrics'];trace=report['diagnostic_trace']
 return {'clock':report['requested_seconds']==60 and report['requested_fps']==16,'captured960':m['captured']==960,'processed960':m['processed']==960,
  'sequence_exact_once': [r['sequence'] for r in trace]==list(range(1,961)),
  'zero_drop_skip_errors_gaps':all(m[k]==0 for k in ['dropped','source_skipped','inspection_errors','coverage_gaps']),
  'zero_deadline_late':m['late']==0,'clean_eof':report['state']['state']=='eof' and report['state']['error'] is None}
def compiled_properties(model):
 result={}
 for key in ['EXECUTION_DEVICES','INFERENCE_PRECISION_HINT','NUM_STREAMS','PERFORMANCE_HINT','GPU_DISABLE_WINOGRAD_CONVOLUTION']:
  value=model.compiled.get_property(key)
  result[key]=list(value) if key=='EXECUTION_DEVICES' else value if type(value) in (bool,int,float,str) else str(value)
 return result
def recipe_passed(p):
 return (isinstance(p.get('EXECUTION_DEVICES'),list) and len(p['EXECUTION_DEVICES'])==1 and str(p['EXECUTION_DEVICES'][0]).split('.')[0]=='GPU'
  and p.get('INFERENCE_PRECISION_HINT') in ['f16','float16',"<Type: 'float16'>"] and str(p.get('NUM_STREAMS'))=='1'
  and str(p.get('PERFORMANCE_HINT')) in ['LATENCY','PerformanceMode.LATENCY'] and p.get('GPU_DISABLE_WINOGRAD_CONVOLUTION') is False)
def capacity_records_valid(records,sources):
 def stages_valid(row):
  s=row.get('neural_stage_ms')
  return (isinstance(s,dict) and set(s)=={'preprocess','infer_readback','postprocess','total'}
   and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in s.values()) and s['total']>0 and s['infer_readback']>0
   and abs(s['total']-s['preprocess']-s['infer_readback']-s['postprocess'])<=.005
   and type(row.get('profile_ms')) in (int,float) and s['total']<=row['profile_ms']+.01)
 return (len(records)==64 and len(sources)==16 and all(type(r.get('index')) is int and r['index']==i and r.get('source')==sources[i%16]['image']
  and stages_valid(r) for i,r in enumerate(records)))
def start_attempt(label):
 target=OUT/(label+'.json');attempt=OUT/(label+'-attempt.json')
 if target.exists() or attempt.exists():raise ValueError('Preserve first benchmark attempt: '+label)
 write_new(attempt,{'plan_sha256':sha(PLAN),'started_utc':datetime.now(timezone.utc).isoformat()});return target
def predecode_normal(plan):
 import cv2
 frames=[]
 for r in plan['normal16_sources']:
  path=ROOT/'datasets/factory/KolektorSDD'/r['image']
  if sha(path)!=r['image_sha256']:raise ValueError('Fixed TRAIN input changed')
  frame=cv2.imread(str(path))
  if frame is None or list(frame.shape)!=r['native_shape']:raise ValueError('Fixed TRAIN source decode/shape differs')
  frames.append(frame)
 return frames
def new_profile(record,warm):
 from edge_runtime.profiles.factory import FactoryProfile
 from edge_runtime.contracts import FrameContext
 profile=FactoryProfile({'defect_model_path':record['path'],'defect_backend':'auto','confirmation_frames':1});profile.prepare_worker()
 if profile._defect_model.neural.initialization_report['warmup_iterations']!=3:raise ValueError('Exactly3 existing zero worker-init calls required')
 props=compiled_properties(profile._defect_model.neural)
 if not recipe_passed(props):raise ValueError('Actual GPU/f16/1stream/LATENCY/default Winograd recipe differs')
 for i,frame in enumerate(warm[:4]):profile.process(frame,FrameContext(i,time.perf_counter(),unit_id='root-r4-warm'+str(i)))
 profile.reset();return profile,props
def capacity(plan,name):
 import cv2
 from edge_runtime.contracts import FrameContext
 target=start_attempt(name+'-first64-capacity');frames=predecode_normal(plan);profile,props=new_profile(plan['binding']['models'][name],frames);records=[]
 for i in range(64):
  started=time.perf_counter();result=profile.process(frames[i%16],FrameContext(i,started,unit_id='root-r4-cap'+str(i)))
  records.append({'index':i,'source':plan['normal16_sources'][i%16]['image'],'profile_ms':1000*(time.perf_counter()-started),'status':result.status,'neural_stage_ms':result.metrics.get('factory_neural_stage_ms')})
 timing=summarize([r['profile_ms'] for r in records],64);passed=timing['serial_full_profile_FPS']>=15 and capacity_records_valid(records,plan['normal16_sources'])
 write_new(target,{'plan_sha256':sha(PLAN),'model':plan['binding']['models'][name],'configuration':props,'records':records,'timing':timing,
  'initialization':dict(profile._defect_model.neural.initialization_report),'extra_real_warmups':4,'capacity_gate_passed':passed,'pins_unchanged_after':verify_plan()==plan})
 print(json.dumps({'name':name,'capacity_gate_passed':passed,'timing':timing},indent=2))
def passed_capacity(plan,name):
 r=read(OUT/(name+'-first64-capacity.json'))
 expected=summarize([v['profile_ms'] for v in r['records']],64)
 passed=(recipe_passed(r['configuration']) and expected['serial_full_profile_FPS']>=15 and capacity_records_valid(r['records'],plan['normal16_sources'])
  and r['initialization']['warmup_iterations']==3 and r['extra_real_warmups']==4)
 if r['model']!=plan['binding']['models'][name] or r['plan_sha256']!=sha(PLAN) or r['timing']!=expected or r['capacity_gate_passed']!=passed or not r['pins_unchanged_after']:raise ValueError('First64 capacity identity/gate differs')
 return passed
def active_names(plan):
 arms=[n for n in plan['binding']['eligible_round4_arms'] if passed_capacity(plan,n)]
 if not arms:raise ValueError('No new winner passed first64; no original/paired/runtime workload')
 return ['v10h','C200',*arms]
def paired(plan):
 import cv2
 from edge_runtime.contracts import FrameContext
 names=active_names(plan);target=start_attempt('paired-CAL80');warm=predecode_normal(plan);profiles={};props={}
 for name in names:profiles[name],props[name]=new_profile(plan['binding']['models'][name],warm)
 sources=plan['binding']['calibration_sources'];frames=[]
 for r in sources:
  path=ROOT/'datasets/factory/KolektorSDD'/r['image']
  if sha(path)!=r['image_sha256']:raise ValueError('Pinned CAL source changed')
  frame=cv2.imread(str(path))
  if frame is None:raise ValueError('CAL decode failed')
  frames.append(frame)
 records=[]
 for repeat in range(3):
  for i,frame in enumerate(frames):
   offset=(i+repeat)%len(names);order=names[offset:]+names[:offset];row={'repeat':repeat,'source':sources[i]['image'],'order':order,'models':{}}
   for name in order:
    started=time.perf_counter();result=profiles[name].process(frame,FrameContext(i,started,unit_id=f'root-r4-pair-{repeat}-{i}'))
    row['models'][name]={'profile_ms':1000*(time.perf_counter()-started),'status':result.status,'neural_stage_ms':result.metrics.get('factory_neural_stage_ms')}
   records.append(row)
 write_new(target,{'plan_sha256':sha(PLAN),'models':{n:plan['binding']['models'][n] for n in names},'configuration':props,'records':records,
  'timings':{n:summarize([r['models'][n]['profile_ms'] for r in records],240) for n in names},
  'all_calls_inferred':{n:all(isinstance(r['models'][n]['neural_stage_ms'],dict) for r in records) for n in names},'pins_unchanged_after':verify_plan()==plan,
  'scope':'Balanced complete-profile timings on predecoded CAL80; three repetitions add timing observations, not accuracy evidence. No extra selected snapshot/model calls.'})
 print('PAIRED',target)
def legacy(plan,name,stage):
 names=active_names(plan)
 if name not in names:raise ValueError('Model not active after first64 capacity')
 target=start_attempt(name+'-'+stage);model=plan['binding']['models'][name]['path']
 if stage=='original':
  command=[sys.executable,'-X','utf8',str(ROOT/'scripts/factory_localization_audit.py'),'--model',model,'--partition','test','--output',str(target),'--visuals-dir',str(OUT/(name+'-ALL-original-localization-visuals'))]
 else:
  command=[sys.executable,'-X','utf8',str(ROOT/'scripts/benchmark_runtime.py'),'--model',model,'--backend','auto','--varied','--partition','calibration','--fps','16','--seconds','60','--deadline','100','--preview-fps','6','--trace','--output',str(target)]
 subprocess.run(command,cwd=ROOT,check=True)
 report=read(target);gate={'plan_sha256':sha(PLAN),'raw_report_sha256':sha(target),'model':plan['binding']['models'][name],'command':command,'pins_unchanged_after':verify_plan()==plan}
 if stage=='runtime':
  expected=[v['image_sha256'] for v in plan['binding']['calibration_sources']]
  if report['input_sha256']!=expected:raise ValueError('Runtime source order/identity differs')
  gate['checks']=runtime_checks(report);gate['runtime_gate_passed']=all(gate['checks'].values());gate['profile_tails']=summarize([r['profile_ms'] for r in report['diagnostic_trace']])
  gate['host_to_profile_return_tails']=summarize([r['host_to_profile_return_ms'] for r in report['diagnostic_trace']]);gate['all_over100ms_trace_rows']=[r for r in report['diagnostic_trace'] if r['host_to_profile_return_ms']>100]
 else:gate['accuracy_only_no_model_selection']=True
 write_new(OUT/(name+'-'+stage+'-gate.json'),gate);print('COMPLETED',name,stage)

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['prepare','capacity','paired','runtime','original'],required=True);parser.add_argument('--name',choices=['v10h','C200','primary','secondary',EXPLORATORY_NAME]);parser.add_argument('--root-quiet-release',action='store_true');parser.add_argument('--root-original-release',action='store_true');args=parser.parse_args()
 if args.stage=='prepare':prepare()
 else:
  if not args.root_quiet_release:raise ValueError('Separate explicit root quiet-window release required for any NN')
  if args.stage=='original' and not args.root_original_release:raise ValueError('Root alone releases exposed original scoring after capacity')
  import cv2
  cv2.setNumThreads(2);plan=verify_plan()
  if args.stage=='paired':paired(plan)
  else:
   if args.name not in plan['binding']['models']:raise ValueError('Only present frozen policy/comparator names allowed')
   if args.stage=='capacity':capacity(plan,args.name)
   else:legacy(plan,args.name,args.stage)
