"""Pure benchmark integrity fixtures; no model/images/NN or runtime start."""
import copy,sys,unittest,hashlib,json
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark_factory_round4_root import summarize,runtime_checks,recipe_passed,capacity_records_valid,historical_cal_identity
import benchmark_factory_round4_root as benchmark

def runtime_report():
 return {'requested_seconds':60,'requested_fps':16,'state':{'state':'eof','error':None,
  'metrics':{'captured':960,'processed':960,'dropped':0,'source_skipped':0,'inspection_errors':0,'coverage_gaps':0,'late':0}},
  'diagnostic_trace':[{'sequence':i} for i in range(1,961)]}

class RootBenchContracts(unittest.TestCase):
 def test_post_CAL_exploratory_entry_has_same_first64_speed_gate(self):
  name=benchmark.EXPLORATORY_NAME;sources=[{'image':str(i)} for i in range(16)]
  model={'role':'POST-CAL exploratory comparison'};plan={'binding':{'models':{name:model}},'normal16_sources':sources}
  rows=[{'index':i,'source':str(i%16),'profile_ms':40.,'neural_stage_ms':{'preprocess':1.,'infer_readback':20.,'postprocess':1.,'total':22.}} for i in range(64)]
  recipe={'EXECUTION_DEVICES':['GPU.0'],'INFERENCE_PRECISION_HINT':'f16','NUM_STREAMS':1,'PERFORMANCE_HINT':'LATENCY','GPU_DISABLE_WINOGRAD_CONVOLUTION':False}
  report={'records':rows,'model':model,'plan_sha256':'fixed','configuration':recipe,'initialization':{'warmup_iterations':3},'extra_real_warmups':4,'capacity_gate_passed':True,'pins_unchanged_after':True,'timing':summarize([40.]*64,64)}
  with patch.object(benchmark,'read',return_value=report),patch.object(benchmark,'sha',return_value='fixed'):
   self.assertTrue(benchmark.passed_capacity(plan,name))
   report['records'][-1]['profile_ms']=4000.;report['timing']=summarize([40.]*63+[4000.],64);report['capacity_gate_passed']=False
   self.assertFalse(benchmark.passed_capacity(plan,name))
 def test_historical_CAL_serialization_keeps_exact_rows_order_and_content(self):
  rows=[{'image':'item/Part0.jpg','image_sha256':'abc','shape':[1280,512],'defective':False},
        {'image':'item/Part1.jpg','image_sha256':'def','shape':[1280,512],'defective':True}]
  expected=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
  compact=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
  self.assertEqual(historical_cal_identity(rows),expected);self.assertNotEqual(expected,compact)
  self.assertNotEqual(historical_cal_identity(rows[::-1]),expected)
  changed=copy.deepcopy(rows);changed[0]['image_sha256']='changed'
  self.assertNotEqual(historical_cal_identity(changed),expected)
 def test_first64_preserves_outlier_and_rejects_missing_or_invalid_timing(self):
  t=summarize([50.]*63+[2000.],64)
  self.assertEqual(t['max_ms'],2000.);self.assertAlmostEqual(t['mean_ms'],(63*50+2000)/64)
  self.assertLess(t['serial_full_profile_FPS'],15)
  for value in ([50.]*63,[50.]*63+[float('nan')],[50.]*63+[0],[50.]*63+[True]):
   with self.assertRaises(ValueError):summarize(value,64)
 def test_runtime_exact_clock_and_all_counters_required(self):
  self.assertTrue(all(runtime_checks(runtime_report()).values()))
  for metric,value in [('captured',959),('processed',959),('dropped',1),('source_skipped',1),('inspection_errors',1),('coverage_gaps',1),('late',1)]:
   r=runtime_report();r['state']['metrics'][metric]=value;self.assertFalse(all(runtime_checks(r).values()))
  r=runtime_report();r['requested_fps']=15;self.assertFalse(all(runtime_checks(r).values()))
 def test_runtime_equal_count_still_rejects_duplicate_and_missing_sequence(self):
  r=runtime_report();r['diagnostic_trace'][500]['sequence']=500
  self.assertEqual(len(r['diagnostic_trace']),960);self.assertFalse(runtime_checks(r)['sequence_exact_once'])
 def test_capacity_requires_fixed_real_source_cycle_and_actual_nn_every_call(self):
  sources=[{'image':str(i)} for i in range(16)]
  rows=[{'index':i,'source':str(i%16),'profile_ms':40.,'neural_stage_ms':{'preprocess':1.,'infer_readback':20.,'postprocess':1.,'total':22.}} for i in range(64)]
  self.assertTrue(capacity_records_valid(rows,sources))
  for key,value in [('index',0),('source','wrong'),('neural_stage_ms',None),('neural_stage_ms',{})]:
   bad=copy.deepcopy(rows);bad[30][key]=value;self.assertFalse(capacity_records_valid(bad,sources))
  for key,value in [('total',float('nan')),('infer_readback',float('inf')),('postprocess',-1.),('total',39.),('preprocess',True)]:
   bad=copy.deepcopy(rows);bad[30]['neural_stage_ms'][key]=value;self.assertFalse(capacity_records_valid(bad,sources))
 def test_actual_gpu_recipe_is_checked_without_fallback(self):
  p={'EXECUTION_DEVICES':['GPU.0'],'INFERENCE_PRECISION_HINT':'f16','NUM_STREAMS':1,'PERFORMANCE_HINT':'LATENCY','GPU_DISABLE_WINOGRAD_CONVOLUTION':False}
  self.assertTrue(recipe_passed(p))
  for key,value in [('EXECUTION_DEVICES',['CPU']),('INFERENCE_PRECISION_HINT','f32'),('NUM_STREAMS',2),('GPU_DISABLE_WINOGRAD_CONVOLUTION',True)]:
   bad=dict(p);bad[key]=value;self.assertFalse(recipe_passed(bad))

if __name__=='__main__':unittest.main()
