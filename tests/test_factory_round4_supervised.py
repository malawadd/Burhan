"""TRAIN geometry/loss contracts only; no model, weights, or source-image decode."""
import ast,copy,importlib.util,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from factory_round4_supervised import projected,crop_observed,coverage_crops,supervised_loss
from evaluate_factory_round4 import recipe_passes,numeric_passes,select_round4,raw_candidates,validate_search
from factory_mask_metrics import original_pixel_weights,component_mask_statistics,at_area,aggregate
from edge_runtime.profiles.factory_encoder import seeded_growth_mask
from factory_stfpm_common import rank

def recipe():return {'EXECUTION_DEVICES':['GPU.0'],'INFERENCE_PRECISION_HINT':'f16','NUM_STREAMS':1,'PERFORMANCE_HINT':'LATENCY','GPU_DISABLE_WINOGRAD_CONVOLUTION':False}

def search_fixture():
    rows=[]
    for i in range(6):
        truth=np.zeros((8,8),np.uint8);scores=np.full((8,8),.1,np.float32)
        if i<2:truth[2:6,2:6]=1;scores[2:6,2:6]=.95 if i==0 else .6
        elif i<4:scores[0,0]=.6
        rows.append({'image':str(i),'defective':i<2,'truth':truth,'scores':scores,'projection_weights':original_pixel_weights(truth,scores.shape)})
    return rows,{'raw_high_grid':[.5,.8],'area_grid':[1],'growth_grid':[None,.25]}

class Round4Contracts(unittest.TestCase):
    def test_all_foreground_is_covered_once_across_native_tile_boundaries(self):
        truth=np.zeros((701,512),np.uint8);truth[319:322,0:10]=1;truth[-1,-1]=1;truth[7,7]=1
        tiles=coverage_crops(truth,np.random.default_rng(1));recovered=np.zeros_like(truth)
        for tile in tiles:recovered[tile['top']:tile['top']+320]=truth[tile['top']:tile['top']+320]
        np.testing.assert_array_equal(recovered,truth);self.assertEqual(sum(t['foreground_pixels'] for t in tiles),int(truth.sum()))

    def test_outside_observed_pixels_are_ignored_not_certified_background(self):
        image=np.full((19,512,3),17,np.uint8);truth=np.zeros((19,512),np.uint8);truth[-1,-1]=1
        frame,target,valid=crop_observed(image,truth,0,320)
        self.assertEqual(frame.shape,(320,512,3));self.assertEqual(int(valid.sum()),19*512)
        self.assertEqual(int(target.sum()),1);self.assertFalse(valid[19:].any());self.assertFalse(frame[19:].any())

    def test_new_domains_preserve_aspect_and_real_pixel_gt_occupancy(self):
        image=np.zeros((63,23,3),np.uint8);truth=np.zeros((63,23),np.uint8);truth[-1,-1]=255
        frame,target=projected(image,truth,'SDD2')
        self.assertEqual(frame.shape[:2],(round(63*512/23),512));self.assertTrue(target[-1,-1]);self.assertEqual(set(np.unique(target)),{0,1})

    def test_SDD1_uses_exact_unchanged_serving_input_geometry(self):
        image=np.zeros((128,50,3),np.uint8);truth=np.zeros((128,50),np.uint8);truth[40:45,49]=85
        frame,target=projected(image,truth,'SDD1');self.assertEqual(frame.shape,(1280,512,3));self.assertEqual(target.shape,(1280,512))
        self.assertTrue(target[:,511].any());self.assertFalse(target[:,0].any())

    @unittest.skipUnless(importlib.util.find_spec('torch'),'Loss fixture requires isolated export environment')
    def test_ignored_padding_has_exact_zero_loss_gradient_and_no_loss_influence(self):
        import torch
        base=torch.tensor([[[[-2.,1.,3.],[.1,-.2,.4]]]],requires_grad=True);target=torch.tensor([[[[0.,1.,0.],[1.,0.,0.]]]])
        valid=torch.ones_like(target,dtype=torch.bool);first=supervised_loss(base,target,valid)
        padded=torch.cat([base.detach(),torch.full_like(base,50.)],dim=2).requires_grad_()
        padded_target=torch.cat([target,torch.ones_like(target)],dim=2);padded_valid=torch.cat([valid,torch.zeros_like(valid)],dim=2)
        second=supervised_loss(padded,padded_target,padded_valid);torch.testing.assert_close(first,second,rtol=0,atol=0)
        second.backward();self.assertEqual(int(torch.count_nonzero(padded.grad[:,:,2:])),0)

    def test_compiled_configuration_recomputed_not_aggregate_boolean(self):
        self.assertTrue(recipe_passes(recipe()))
        for key,value in [('EXECUTION_DEVICES',['CPU']),('INFERENCE_PRECISION_HINT','f32'),('NUM_STREAMS',2),('GPU_DISABLE_WINOGRAD_CONVOLUTION',True)]:
            broken=recipe();broken[key]=value;self.assertFalse(recipe_passes(broken))

    def test_numerical_failures_exclude_CAL_without_guard_relaxation(self):
        r={'configuration':recipe(),'cases':[{'max_abs_error':.001,'score_range':[.1,.8],'numeric_failure':None} for _ in range(3)]}
        self.assertTrue(numeric_passes(r))
        for change in [{'max_abs_error':.01001},{'max_abs_error':-1},{'max_abs_error':True},{'score_range':[0,float('nan')]},{'numeric_failure':'nonfinite'}]:
            bad=copy.deepcopy(r);bad['cases'][0].update(change);self.assertFalse(numeric_passes(bad))

    def test_primary_budget1_is_exact_old_search_on_same_cached_maps(self):
        rows,protocol=search_fixture()
        # Execute only the original two pure functions, never its runtime imports.
        tree=ast.parse((ROOT/'scripts/calibrate_factory_stfpm_learn.py').read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'select','summarize'}]
        namespace={'np':np,'aggregate':aggregate,'component_mask_statistics':component_mask_statistics,'at_area':at_area,'seeded_growth_mask':seeded_growth_mask,'rank':rank}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'original_pure_select','exec'),namespace)
        raw,grown,point,records=namespace['select'](rows,protocol);actual=select_round4(rows,protocol,1)
        self.assertEqual(raw,raw_candidates(rows,protocol));self.assertEqual(grown,actual['grown_grid']);self.assertEqual(point,actual['selected']);self.assertEqual(records,actual['selected_records'])

    def test_secondary_budget_is_independent_prospective_arm_not_primary_relaxation(self):
        rows,protocol=search_fixture();raw=raw_candidates(rows,protocol)
        primary=select_round4(rows,protocol,1,raw);secondary=select_round4(rows,protocol,3,raw)
        validate_search(raw,primary,protocol,1);validate_search(raw,secondary,protocol,3)
        self.assertEqual(primary['selected']['normal_fp'],0);self.assertEqual(primary['selected']['meaningful_iou_0_1_recall'],.5)
        self.assertEqual(secondary['selected']['normal_fp'],2);self.assertEqual(secondary['selected']['meaningful_iou_0_1_recall'],1.)
        with self.assertRaises(ValueError):validate_search(raw,secondary,protocol,1)

if __name__=='__main__':unittest.main()
