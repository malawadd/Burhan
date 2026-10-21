"""Pure numeric/AST guards only; never imports Torch/OV or reads dataset pixels."""
import ast
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import factory_c_density_common as common
import factory_c_density_math as math


class Preparation(unittest.TestCase):
    def test_paired_c_forward_preserves_source_operations(self):
        old=ast.parse((ROOT/'scripts/train_factory_round3d.py').read_text())
        new=ast.parse((ROOT/'scripts/factory_c_density_graph.py').read_text())
        source=next(n for n in ast.walk(old) if isinstance(n,ast.ClassDef) and n.name=='PackedPixelRefiner')
        source=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        paired=next(n for n in ast.walk(new) if isinstance(n,ast.FunctionDef) and n.name=='parts')
        class Rename(ast.NodeTransformer):
            def visit_Name(self,node):
                return ast.copy_location(ast.Name(id='self' if node.id=='base' else node.id,ctx=node.ctx),node)
        actual=Rename().visit(copy.deepcopy(paired))
        self.assertEqual([ast.dump(n,include_attributes=False) for n in source.body[:-1]],
                         [ast.dump(n,include_attributes=False) for n in actual.body[1:-1]])
        expected=ast.parse('def f():\n return F.max_pool2d(logits+residual,2),fine').body[0].body[0]
        self.assertEqual(ast.dump(actual.body[-1],include_attributes=False),ast.dump(expected,include_attributes=False))

    def test_density_is_added_after_pooled_semantic(self):
        tree=ast.parse((ROOT/'scripts/factory_c_density_graph.py').read_text())
        klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CNormalDensityGraph')
        method=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        expected=ast.parse('def f():\n logits,fine=self.parts(bgr)\n return logits*self.alpha if self.beta==0. else self.alpha*logits+self.beta*self.evidence(fine)').body[0]
        self.assertEqual([ast.dump(n,include_attributes=False) for n in method.body], [ast.dump(n,include_attributes=False) for n in expected.body])

    def test_exact_nine_plan_and_geometry(self):
        value={'candidates':[{'id':i,'alpha':a,'beta':b} for i,(a,b) in enumerate(common.CANDIDATES)],'save_at':list(range(9)), 'input_shape':[1,3,1280,512],'output_shape':[1,1,640,256]}
        common.validate_fusion_plan(value)
        bad=copy.deepcopy(value);bad['candidates'][0]['beta']=1.
        with self.assertRaises(ValueError):common.validate_fusion_plan(bad)
        bad=copy.deepcopy(value);bad['output_shape']=[1,1,1280,512]
        with self.assertRaises(ValueError):common.validate_fusion_plan(bad)
        bad=copy.deepcopy(value);bad['save_at']=list(range(8))
        with self.assertRaises(ValueError):common.validate_fusion_plan(bad)

    def test_quantized_mean_covariance_matches_actual_residuals(self):
        rng=np.random.default_rng(74)
        features=(rng.normal(0,.17,(16,7,256))+np.linspace(30.001,35.249,256)[None,None,:])
        mean=features.mean(1);flat=features.reshape(16,-1);second=flat@flat.T/flat.shape[1]
        muq=mean.astype(np.float16).astype(np.float32)
        residual=(features-muq[:,None,:]).reshape(16,-1)
        empirical=residual@residual.T/residual.shape[1]
        captured=[]
        def whitening(zero,cov):
            self.assertFalse(np.any(zero));captured.append(cov)
            return np.eye(16,dtype=np.float32),{}
        with patch.object(math,'whitening_from_moments',whitening):
            deployed,w,_=math.precision_corrected_moments(mean,second)
        self.assertTrue(np.array_equal(deployed,muq))
        self.assertTrue(np.allclose(captured[0],empirical,atol=2e-12,rtol=1e-10))
        wrong=second-muq.astype(float)@muq.astype(float).T/256
        self.assertGreater(float(np.max(np.abs(wrong-empirical))),1e-5)

    def test_whitening_half_rounding_keeps_bound(self):
        mean=np.linspace(-.1,.1,256)[None].repeat(16,0)
        second=mean@mean.T/256+np.diag(np.linspace(.001,.03,16))
        mu,w,report=math.precision_corrected_moments(mean,second)
        self.assertTrue(np.array_equal(mu,mu.astype(np.float16).astype(mu.dtype)))
        self.assertTrue(np.array_equal(w,w.astype(np.float16).astype(w.dtype)))
        self.assertLessEqual(float(np.abs(w).sum(1).max()),8.)
        self.assertTrue(np.isfinite(w).all())

    def test_independent_qspan_survives_rounded_quantile_collision(self):
        q50,q95,span=math.quantized_quantiles(.33334,.33335)
        self.assertEqual(q50,q95)
        self.assertGreater(span,0)
        self.assertNotEqual(span,q95-q50)
        self.assertEqual(span,float(np.float16(.001)))
        with self.assertRaises(ValueError):math.quantized_quantiles(1.,0.)
        with self.assertRaises(ValueError):math.quantized_quantiles(0.,float('nan'))


if __name__=='__main__':unittest.main()
