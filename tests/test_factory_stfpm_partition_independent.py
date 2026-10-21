"""Independent tiny tensor arithmetic only; no model, weights, graph or NN."""
import os,sys,unittest
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import torch
from torch.nn import functional as F
from factory_stfpm_graph import unit_feature as original_unit,disagreement as original_disagreement
from factory_stfpm_common import NORM_SCALE_FLOOR,NORM_SQUARED_FLOOR,MAP_FLOOR
import factory_stfpm_partition_math as candidate
torch.set_num_threads(1)

class IndependentPartitionArithmetic(unittest.TestCase):
    def tensor(self,value):return torch.from_numpy(np.ascontiguousarray(value,dtype=np.float32))

    def compare(self,actual,expected):
        self.assertEqual(actual.dtype,np.float32)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=2e-6)

    def test_unit_zero_near_floor_subnormal_and_high_finite_vs_original(self):
        levels=[0.,2**-24,1e-20,NORM_SCALE_FLOOR*(1-1e-6),NORM_SCALE_FLOOR,
            NORM_SCALE_FLOOR*(1+1e-6),1.,1e20,np.finfo(np.float32).max*.75]
        for channels in [1,7,40]:
            for scale in levels:
                data=np.linspace(-1,1,channels*3*5,dtype=np.float32).reshape(1,channels,3,5)*np.float32(scale)
                with torch.no_grad():expected=original_unit(self.tensor(data)).numpy()
                self.compare(candidate.unit_feature(data),expected)

    def test_squared_norm_floor_boundary_vs_original(self):
        # One nonzero channel remains below the absmax floor; target scaled
        # magnitude straddles sqrt(squared-denominator-floor).
        base=np.float32(NORM_SCALE_FLOOR*np.sqrt(NORM_SQUARED_FLOOR))
        for multiplier in [.5,1-1e-5,1.,1+1e-5,2.]:
            data=np.zeros((1,5,2,3),np.float32);data[:,0]=base*np.float32(multiplier)
            with torch.no_grad():expected=original_unit(self.tensor(data)).numpy()
            self.compare(candidate.unit_feature(data),expected)

    def test_disagreement_multiscale_random_identical_and_opposite_vs_original(self):
        rng=np.random.default_rng(20261030)
        for scale in [1e-20,1e-6,.0039,1.,1e20,1e37]:
            teacher=rng.normal(size=(1,16,5,3)).astype(np.float32)*np.float32(scale)
            for student in [teacher.copy(),-teacher,rng.normal(size=teacher.shape).astype(np.float32)*np.float32(scale)]:
                with torch.no_grad():expected=original_disagreement(self.tensor(teacher),self.tensor(student)).numpy()
                self.compare(candidate.disagreement(teacher,student),expected)

    def test_half_pixel_resize_odd_up_down_and_degenerate_edges_vs_torch(self):
        cases=[((2,3),(7,11)),((5,7),(3,4)),((1,3),(9,1)),((4,1),(1,8)),((1,1),(6,7)),((3,4),(3,4))]
        for source,target in cases:
            data=np.arange(2*source[0]*source[1],dtype=np.float32).reshape(1,2,*source)/13
            with torch.no_grad():expected=F.interpolate(self.tensor(data),size=target,mode='bilinear',align_corners=False).numpy()
            self.compare(candidate.bilinear_half_pixel(data,*target),expected)

    def test_complete_three_level_head_vs_original_tensor_functions(self):
        rng=np.random.default_rng(20261101)
        shapes=[(1,16,13,9),(1,24,4,3),(1,40,2,1)]
        for scale in [0.,1e-15,1e-5,1.,1e30]:
            features=[rng.normal(size=shape).astype(np.float32)*np.float32(scale) for shape in shapes for _ in range(2)]
            snapshots=[value.tobytes() for value in features]
            with torch.no_grad():
                maps=[original_disagreement(self.tensor(features[i]),self.tensor(features[i+1])) for i in [0,2,4]]
                maps=[maps[0],*[F.interpolate(v,size=maps[0].shape[-2:],mode='bilinear',align_corners=False) for v in maps[1:]]]
                expected=torch.exp(sum(torch.log(v.clamp(min=MAP_FLOOR,max=1.)) for v in maps)/3.).clamp(MAP_FLOOR,1.).numpy()
            actual=candidate.score_from_features(features)
            self.compare(actual,expected);self.assertTrue(actual.flags.owndata)
            self.assertEqual([value.tobytes() for value in features],snapshots)
            self.assertGreaterEqual(float(actual.min()),MAP_FLOOR);self.assertLessEqual(float(actual.max()),1.)

if __name__=='__main__':unittest.main()
