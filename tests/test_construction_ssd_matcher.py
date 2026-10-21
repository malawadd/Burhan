"""Training assignment regression tests; optional Torch dependency."""
from pathlib import Path
import sys
import pytest

try:
    import torch
except ImportError:
    torch=None
pytestmark=pytest.mark.skipif(torch is None,reason='Training-only matcher requires optional Torch environment')
if torch is not None:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from torchvision.models.detection._utils import SSDMatcher
    from construction_ssd_matcher import UniqueForcedSSDMatcher

def test_shared_best_anchor_does_not_remove_a_ground_truth():
    quality=torch.tensor([[.40,.39,.10],[.38,.10,.09]])
    stock=SSDMatcher(.5)(quality)
    assert int((stock==0).sum())==0
    matcher=UniqueForcedSSDMatcher(.5);actual=matcher(quality)
    assert actual.tolist()==[1,0,-1]
    assert all(int((actual==i).sum())>=1 for i in range(2))
    assert matcher.assignment_diagnostics[-1]['stock_positive_anchor_counts']==[0,1]

def test_noncollision_threshold_positives_remain_unchanged():
    quality=torch.tensor([[.9,.7,.1,.1],[.1,.1,.8,.6]])
    assert torch.equal(UniqueForcedSSDMatcher(.5)(quality),SSDMatcher(.5)(quality))

def test_empty_targets_leave_every_anchor_background():
    assert UniqueForcedSSDMatcher(.5)(torch.empty((0,4))).tolist()==[-1]*4

@pytest.mark.parametrize('case',['nan','inf','too_few','no_anchors','wrong_rank'])
def test_invalid_or_insufficient_anchor_matrices_fail(case):
    quality={'nan':lambda:torch.tensor([[float('nan'),.2]]),
             'inf':lambda:torch.tensor([[float('inf'),.2]]),
             'too_few':lambda:torch.ones((2,1)),
             'no_anchors':lambda:torch.empty((1,0)),
             'wrong_rank':lambda:torch.ones((1,2,3))}[case]()
    with pytest.raises(ValueError):UniqueForcedSSDMatcher(.5)(quality)
