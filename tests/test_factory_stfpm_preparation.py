"""Pure NumPy/AST/source-metadata L fixtures. No Torch, images or NN."""
import ast,copy,json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT))
import factory_stfpm_common as c
from factory_stfpm_math import unit_feature,disagreement,geometric_mean


class Preparation(unittest.TestCase):
    def test_half_constant_extremes_and_zero_have_finite_bounded_vectors(self):
        for channels in [16,24,40]:
            for value in [0,1e-6,-1e-6,1,-1,65504,-65504]:
                x=np.full((1,channels,3,4),value,dtype=np.float16)
                with np.errstate(over='raise',divide='raise',invalid='raise'):
                    normalized=unit_feature(x);score=disagreement(x,-x)
                self.assertTrue(np.isfinite(normalized).all());self.assertTrue(np.isfinite(score).all())
                self.assertLessEqual(float(score.max()),1);self.assertGreaterEqual(float(score.min()),0)
                self.assertLessEqual(float((normalized.astype(np.float64)**2).sum(1).max()),1.003)

    def test_matching_and_opposite_directions_not_intensity_scale(self):
        rng=np.random.default_rng(4);x=rng.normal(size=(1,40,4,5)).astype(np.float32)
        np.testing.assert_allclose(disagreement(x,x),0,atol=1e-7)
        np.testing.assert_allclose(disagreement(x,-x),1,atol=2e-6)
        np.testing.assert_allclose(disagreement(x,4*x),0,atol=1e-7)
        y=x.copy();y[:,0]*=-1
        self.assertGreater(float(disagreement(x,y).max()),0)

    def test_geometric_mean_floor_is_half_representable_and_does_not_underflow(self):
        for value in [c.NORM_SCALE_FLOOR,c.NORM_SQUARED_FLOOR,c.MAP_FLOOR]:self.assertEqual(float(np.float16(value)),value)
        for value in [0,c.MAP_FLOOR,.01,1]:
            maps=[np.full((1,1,2,3),value,dtype=np.float16) for _ in range(3)]
            result=geometric_mean(maps)
            self.assertTrue(np.isfinite(result).all());self.assertGreaterEqual(float(result.min()),c.MAP_FLOOR)
        self.assertAlmostEqual(float(geometric_mean([np.array(.01),np.array(.1),np.array(1.)])),.1,places=6)

    def test_actual_sampler_600_source_coverage_and_observed_bounds(self):
        tree=ast.parse((ROOT/'scripts/train_factory_stfpm.py').read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='sample_plan')
        scope={'np':np,'SEED':c.SEED};exec(compile(ast.Module(body=[function],type_ignores=[]),'sampler-only','exec'),scope)
        plan=scope['sample_plan'](208)
        self.assertEqual(plan,scope['sample_plan'](208));self.assertEqual(len(plan),600)
        counts=np.bincount([i for i,_ in plan],minlength=208)
        self.assertEqual(int(counts.min()),2);self.assertEqual(int(counts.max()),3)
        self.assertTrue(all(0<=y<=640 and y+640<=1280 for _,y in plan))
        self.assertGreater(sum(y==0 for _,y in plan),0);self.assertGreater(sum(y==640 for _,y in plan),0)

    def test_normal_source_hash_and_physical_split_guards(self):
        old=json.loads(c.OLD.read_text(encoding='utf-8'));rows=old['normal_sources'];split=old['split_items']
        c.validate_normal_sources(rows,split)
        bad=copy.deepcopy(rows);bad[0]['defective']=True
        with self.assertRaises(ValueError):c.validate_normal_sources(bad,split)
        bad=copy.deepcopy(rows);bad[0]['mask_sha256']='0'*64
        with self.assertRaises(ValueError):c.validate_normal_sources(bad,split)
        bad=copy.deepcopy(split);bad['train'][0]=bad['calibration'][0]
        with self.assertRaises(ValueError):c.validate_normal_sources(rows,bad)

    def test_guard_rejects_falsealarm_lowprecision_nonfinite_and_offgrid(self):
        p={'high':c.HIGH_GRID[5],'area':4,'growth':None,'normal_fp':1,'foreground_pixel_precision':.65,
           'meaningful_iou_0_1_recall':1,'worst_positive_iou':.1,'coverage_50pct_recall':.9,
           'worst_positive_coverage':.5,'micro_pixel_iou':.4,'foreground_pixel_recall':.7}
        protocol={'raw_high_grid':c.HIGH_GRID,'area_grid':c.AREA_GRID,'growth_grid':c.GROWTH_GRID}
        c.validate_point(p,protocol)
        for key,value in [('normal_fp',2),('foreground_pixel_precision',.649),('micro_pixel_iou',float('nan')),('high',.123456)]:
            with self.assertRaises(ValueError):c.validate_point(p|{key:value},protocol)

    def test_complete_search_and_stable_winner_required(self):
        protocol={'raw_high_grid':[.1,.2],'area_grid':[1,2],'growth_grid':[None,.05]}
        def point(high,area,growth=False):
            r={'high':high,'area':area,'normal_fp':0,'foreground_pixel_precision':.8,
               'meaningful_iou_0_1_recall':1.,'worst_positive_iou':high,'coverage_50pct_recall':.9,
               'worst_positive_coverage':.6,'micro_pixel_iou':.5,'foreground_pixel_recall':.7}
            if growth is not False:r['growth']=growth
            return r
        raw=[point(h,a) for h in [.1,.2] for a in [1,2]]
        grown=[point(h,1,g) for h in [.2,.1] for g in [None,.05]]
        report={'raw_grid':raw,'grown_grid':grown,'selected':grown[0]}
        c.validate_search_report(report,protocol)
        bad=copy.deepcopy(report);bad['raw_grid'].pop()
        with self.assertRaises(ValueError):c.validate_search_report(bad,protocol)
        bad=copy.deepcopy(report);bad['grown_grid'].pop()
        with self.assertRaises(ValueError):c.validate_search_report(bad,protocol)
        bad=copy.deepcopy(report);bad['selected']=grown[-1]
        with self.assertRaises(ValueError):c.validate_search_report(bad,protocol)

    def test_graph_source_has_independent_random_student_and_frozen_teacher(self):
        source=(ROOT/'scripts/factory_stfpm_graph.py').read_text();tree=ast.parse(source)
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        constructor_calls=[n for n in ast.walk(init) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='mobilenet_v3_small']
        self.assertEqual(len(constructor_calls),2)
        loaders=[n for n in ast.walk(init) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='load_state_dict']
        self.assertEqual(len(loaders),1)
        self.assertIn('strict=True',ast.unparse(loaders[0]))
        self.assertIn('weights_only=True',source)
        train=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='train_normals')
        self.assertEqual(ast.unparse(train.body[-1]),'self.teacher.eval()')
        self.assertIn('parameter.requires_grad_(False)',source)
        self.assertNotIn('factory_c_density',source)

    def test_runtime_score_semantics_passthrough_preserves_old_default_and_scores(self):
        from edge_runtime.profiles.factory_defect import SurfaceDefectModel
        default='Learned_neural_sigmoid_score_not_calibrated_probability'
        class FakeNeural:
            def predict(self,frame,threshold,area):
                return frame,[{'score':.2}],{'model_score_semantics':default}
        model=object.__new__(SurfaceDefectModel);model.neural=FakeNeural();model.pixel_threshold=.05;model.area_threshold=4
        array=np.array([[.2]],dtype=np.float32)
        model.metadata={}
        old,candidates,metrics=model.predict(array)
        self.assertIs(old,array);self.assertEqual(candidates,[{'score':.2}]);self.assertEqual(metrics['model_score_semantics'],default)
        model.metadata={'score_semantics':c.SCORE_SEMANTICS}
        new,actual,metrics=model.predict(array)
        self.assertIs(new,array);self.assertEqual(actual,candidates);self.assertEqual(metrics['model_score_semantics'],c.SCORE_SEMANTICS)
        model.metadata={'score_semantics':None}
        self.assertEqual(model.predict(array)[2]['model_score_semantics'],default)


if __name__=='__main__':unittest.main()
