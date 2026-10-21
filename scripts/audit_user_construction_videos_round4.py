"""Read-only private-video evidence audit. No NN, upload, training or GT creation."""
import argparse, collections, hashlib, json, math, sys, time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
import av
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/evaluation/user-construction-videos-round4'
PLAN=OUT/'baseline-plan-v2-before-inference.json'
INVENTORY=OUT/'original-video-inventory.json'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while block:=f.read(1024*1024):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def write_new(path,value):
 with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def fraction(row):return Fraction(row['pts'])*Fraction(row['time_base'])
def finite(value):return type(value) in (float,int) and math.isfinite(value)
def protected_path(relative,folder):
 p=(ROOT/relative).resolve()
 if not p.is_relative_to(folder.resolve()):raise ValueError('Evidence path escapes registered folder')
 return p

class Checks:
 def __init__(self):self.total=0;self.failures=[]
 def check(self,condition,context):
  self.total+=1
  if not condition:self.failures.append(context)

def encoded_timeline(path):
 times=[];rotation=None
 with av.open(str(path)) as video:
  stream=video.streams.video[0];stream.codec_context.thread_count=1
  meta={'width':stream.width,'height':stream.height,'average_rate':str(stream.average_rate),'codec':stream.codec_context.name}
  for frame in video.decode(video=0):
   if rotation is None:rotation=int(getattr(frame,'rotation',0))
   times.append({'pts':frame.pts,'time_base':str(frame.time_base)})
 meta.update(rotation_degrees=rotation,frames=len(times))
 return times,meta

def audit_source(source,plan,checks):
 name=Path(source['file']).stem;folder=OUT/name;original=Path(source['file']).resolve()
 if not original.is_relative_to((ROOT.parent/'videos').resolve()):raise ValueError('Original is outside user-provided local videos folder')
 report_path=folder/'before-three-models-v2.json';report=read(report_path);records=report['frames']
 original_sha=sha(original);checks.check(original_sha==source['sha256'],name+': original file SHA')
 checks.check(original.stat().st_size==source['bytes'],name+': original file byte count')
 checks.check(report['source']['sha256']==original_sha and Path(report['source']['original_path']).resolve()==original,name+': report source identity')
 checks.check(report['selection_plan_sha256']==sha(PLAN),name+': report plan SHA')
 checks.check(report['independent_accuracy_credit'] is False and report['purpose']=='failure_diagnostic',name+': diagnostic-only scope')
 for key in ['annotated_frames','localized_matches','missed_defects','false_alerts','unsupported_annotations']:
  checks.check(key in report['summary'] and report['summary'][key] is None,name+': no fabricated '+key)
 checks.check(len(records)==source['decoded_frames']==report['source']['original_camera_frames'],name+': report/inventory count')
 checks.check(read(folder/'source-inventory.json')==source,name+': original inventory copy')
 completion=read(folder/'baseline-v2-completed.json')
 checks.check(completion['report_sha256']==sha(report_path) and completion['frames']==len(records),name+': completion report identity')
 model_names=list(plan['models']);checks.check(list(report['models'])==model_names,name+': exact three baseline models')
 counts={n:collections.Counter() for n in model_names};present=collections.Counter();class_present={n:collections.Counter() for n in model_names}
 mask_counts=collections.Counter();total_masks=0;npz_bytes=0
 times=source['all_frame_times'];origin=fraction(times[0])
 checks.check([r['index'] for r in times]==list(range(len(times))),name+': inventory timeline contiguous')
 # OpenCV5 FFmpeg supplies the same native auto-oriented BGR as serving. A
 # single decoder thread bounds this audit; no processing model is imported.
 cap=cv2.VideoCapture(str(original),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
 if not cap.isOpened():raise ValueError('OpenCV5 cannot open original private video')
 backend=cap.getBackendName();raw_matches=0
 try:
  for i,row in enumerate(records):
   context=f'{name}/frame{i}'
   ok,raw=cap.read()
   if not ok:checks.check(False,context+': missing native source exposure');break
   height,width=raw.shape[:2];expected_size=([source['native_height'],source['native_width']] if source['rotation_degrees']%180 else [source['native_width'],source['native_height']])
   checks.check(row['frame_size_wh']==[width,height]==expected_size,context+': native auto-oriented dimensions')
   raw_ok=hashlib.sha256(raw.tobytes()).hexdigest()==row['raw_native_bgr_sha256'];raw_matches+=int(raw_ok);checks.check(raw_ok,context+': actual native BGR SHA')
   checks.check(row['source_frame_index']==i and row['source_pts']==times[i]['pts'] and row['source_time_base']==times[i]['time_base'],context+': source index/PTS/timebase')
   checks.check(finite(row['video_time_s']) and abs(row['video_time_s']-float(fraction(times[i])-origin))<=1e-9,context+': source relative timestamp')
   checks.check(row['evaluated'] is False and row['expected']==[],context+': explicitly unlabelled/unscored')
   checks.check(read(folder/'baseline-v2-masks'/f'frame_{i:05d}.json')==row,context+': saved frame-record copy')
   checks.check(list(row['models'])==model_names,context+': exact model order/presence')
   flattened=[p for n in model_names for p in row['models'][n]['predictions']]
   checks.check(flattened==row['predictions'],context+': flattened predictions preserved')
   mask_file=protected_path(row['actual_masks']['path'],folder/'baseline-v2-masks')
   checks.check(mask_file==folder/'baseline-v2-masks'/f'frame_{i:05d}.npz',context+': fixed mask asset identity')
   checks.check(sha(mask_file)==row['actual_masks']['sha256'],context+': NPZ file SHA');npz_bytes+=mask_file.stat().st_size
   expected_keys=[]
   with np.load(mask_file,allow_pickle=False) as archive:
    for model_name in model_names:
     predictions=row['models'][model_name]['predictions'];counts[model_name].update(p['label'] for p in predictions)
     present[model_name]+=bool(predictions);class_present[model_name].update(set(p['label'] for p in predictions))
     spec=plan['models'][model_name]['specification'];allowed=plan['models'][model_name]['options'].get('damage_classes',spec['classes'])
     for pi,prediction in enumerate(predictions):
      pc=context+'/'+model_name+'/'+str(pi)
      checks.check(prediction['evaluation_status']=='unscored' and prediction['model']==model_name,pc+': unscored prediction source')
      checks.check(not any(k in prediction for k in ['matched','matched_iou','gt_id','annotation_id','true_positive','false_positive']),pc+': no invented match or GT identity')
      cls=prediction['class_id'];checks.check(type(cls) is int and 0<=cls<len(spec['classes']) and spec['classes'][cls]==prediction['label'] and prediction['label'] in allowed,pc+': declared class mapping/filter')
      checks.check(finite(prediction['score']) and 0<=prediction['score']<=1,pc+': finite score')
      box=prediction['bbox'];checks.check(len(box)==4 and all(finite(v) for v in box) and box[0]>=0 and box[1]>=0 and box[2]>0 and box[3]>0 and box[0]+box[2]<=width+1e-5 and box[1]+box[3]<=height+1e-5,pc+': native box geometry')
      metadata=prediction.get('actual_mask')
      if metadata is None:
       checks.check(model_name=='wall_appearance',pc+': expected mask-bearing model');continue
      checks.check(model_name!='wall_appearance',pc+': no fake box-only mask')
      key=metadata['key'];expected_keys.append(key)
      checks.check(key==f'{model_name}_{pi}' and metadata['shape_hw']==[height,width],pc+': mask native shape/key')
      packed=archive[key];size=height*width
      checks.check(packed.dtype==np.uint8 and packed.ndim==1 and len(packed)==(size+7)//8,pc+': packed native mask schema')
      checks.check(hashlib.sha256(packed.tobytes()).hexdigest()==metadata['packed_sha256'],pc+': packed mask SHA')
      bits=np.unpackbits(packed)
      checks.check(int(bits[:size].sum())==metadata['positive_pixels'] and not bits[size:].any(),pc+': actual mask pixel count/tail bits')
      total_masks+=1;mask_counts[model_name]+=1
    checks.check(sorted(archive.files)==sorted(expected_keys) and len(expected_keys)==len(set(expected_keys)),context+': exact NPZ keys/no hidden GT arrays')
  extra,_=cap.read();checks.check(not extra,name+': no missing final native exposure')
 finally:cap.release()
 checks.check(len(list((folder/'baseline-v2-masks').glob('frame_*.npz')))==len(records),name+': no NPZ frame holes/extras')
 checks.check(len(list((folder/'baseline-v2-masks').glob('frame_*.json')))==len(records),name+': no frame-record holes/extras')
 checks.check({n:dict(v) for n,v in counts.items()}==report['summary']['unscored_prediction_counts'],name+': declared class counts')
 original_times,original_meta=encoded_timeline(original)
 checks.check(original_times==[{k:r[k] for k in ['pts','time_base']} for r in times],name+': independent original frame PTS/count')
 checks.check(original_meta['width']==source['native_width'] and original_meta['height']==source['native_height'] and original_meta['rotation_degrees']==source['rotation_degrees'],name+': original encoded dimensions/rotation')
 derived=protected_path(report['derived_video']['path'],folder)
 checks.check(sha(derived)==report['derived_video']['sha256']==completion['video_sha256'],name+': derived MP4 SHA')
 derived_times,derived_meta=encoded_timeline(derived)
 checks.check([fraction(r) for r in derived_times]==[fraction(r)-origin for r in times],name+': full MP4 exact rational relative PTS')
 checks.check(len(derived_times)==len(records)==report['derived_video']['encoded_frames'] and report['derived_video']['all_original_PTS_exact'] is True,name+': full MP4 count/claim')
 checks.check((derived_meta['width'],derived_meta['height'])==(1400,1240),name+': derived panel geometry')
 checks.check(sha(original)==original_sha,name+': source remained unchanged')
 return {'source_name':name,'original_sha256':original_sha,'report_sha256':sha(report_path),'derived_video_sha256':sha(derived),
  'original_frames':len(times),'raw_BGR_hashes_matching':raw_matches,'original_PTS_frames':len(original_times),'derived_PTS_frames':len(derived_times),
  'model_class_prediction_counts':{n:dict(v) for n,v in counts.items()},'model_prediction_present_frames':dict(present),
  'model_class_prediction_present_frames':{n:dict(v) for n,v in class_present.items()},'native_masks_verified':total_masks,
  'native_masks_per_model':dict(mask_counts),'mask_NPZ_file_bytes_read':npz_bytes,'decoder':{'opencv':cv2.__version__,'backend':backend,'ffmpeg_decode_threads_requested':1,'pyav':av.__version__,'pyav_decode_threads':1}}

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUT/'independent-private-baseline-audit-v1.json');args=parser.parse_args()
 if args.output.exists():raise ValueError('Preserve prior independent audit; use an explicit new output path')
 cv2.setNumThreads(1)
 if not cv2.__version__.startswith('5.'):raise ValueError('Use existing export environment with serving OpenCV5')
 checks=Checks();plan=read(PLAN);inventory=read(INVENTORY);start=time.perf_counter();results=[]
 checks.check(len(inventory)==4 and sum(r['decoded_frames'] for r in inventory)==878,'Exact four originals/878 inventory frames')
 checks.check(plan['inventory_sha256']==sha(INVENTORY),'Frozen plan inventory SHA')
 completion=read(OUT/'all-four-before-v2-completed.json')
 checks.check(completion=={'completed':True,'original_exposures':878,'models_per_exposure':3,'NN_calls_excluding_initialization':2634,'ground_truth_metrics':False,'plan_sha256':sha(PLAN)},'Global completion counts/scope')
 archive=OUT/'baseline-v2-executed-source-archive';snapshot=read(archive/'manifest.json')
 checks.check(snapshot['baseline_plan_sha256']==sha(PLAN) and snapshot['source_pins']==plan['source_pins'],'Executed source archive plan binding')
 for relative,digest in snapshot['source_pins'].items():checks.check(sha(protected_path((archive/relative).relative_to(ROOT),archive))==digest,'Archived source bytes: '+relative)
 for name,row in plan['models'].items():checks.check(sha(ROOT/'models/construction'/row['specification']['file'])==row['specification']['sha256'],'Pinned model asset: '+name)
 error=None
 try:
  for source in inventory:
   print('AUDIT',Path(source['file']).name,flush=True);results.append(audit_source(source,plan,checks));print('DONE',results[-1]['source_name'],results[-1]['original_frames'],flush=True)
 except Exception as exc:
  error={'type':type(exc).__name__,'message':str(exc)};checks.check(False,'Audit interrupted by schema/asset/decoder error; incomplete output is NOT a pass')
 report={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'Read-only integrity/count audit of actual private-video baseline artifacts. No NN, training, upload or ground-truth fabrication.',
  'source_pins':{'auditor_sha256':sha(__file__),'baseline_plan_sha256':sha(PLAN),'inventory_sha256':sha(INVENTORY),'source_archive_manifest_sha256':sha(archive/'manifest.json')},
  'all_checks_passed':not checks.failures and error is None and len(results)==4,'checks':checks.total,'failures':checks.failures,'interruption':error,'videos':results,
  'completed_original_frames':sum(r['original_frames'] for r in results),'native_masks_verified':sum(r['native_masks_verified'] for r in results),
  'accuracy_metrics':None,'evaluation_status':'UNSCORED: output presence/encoding integrity only; no accuracy or verified defect count',
  'timing_scope':'Single-thread media/hash audit alongside TRAIN workloads; elapsed is NOT inference speed','elapsed_seconds':time.perf_counter()-start}
 write_new(args.output,report);print(json.dumps({'all_checks_passed':report['all_checks_passed'],'checks':checks.total,'failures':checks.failures[:10],'interruption':error,'receipt':str(args.output),'sha256':sha(args.output)},indent=2))
 if not report['all_checks_passed']:sys.exit(1)

if __name__=='__main__':main()
