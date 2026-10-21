"""Training-only unique forced SSD anchors; inference/priors remain unchanged.

Uses torchvision's BSD-3-Clause Matcher thresholding and SciPy's BSD-3-Clause
rectangular linear assignment. No site-packages are patched. Each ground-truth
box reserves a distinct anchor before forcing positives, preventing SSD's shared
argmax anchor overwrite. No localization metric or model score is used here.
"""
from __future__ import annotations
import numpy as np
import torch
try:
    from scipy.optimize import linear_sum_assignment
except ModuleNotFoundError as error:
    if error.name!='scipy':raise
    # Reuse the project's already installed compatible SciPy in this process;
    # do not install packages or mutate either virtual environment.
    import sys
    from pathlib import Path
    existing=Path(__file__).resolve().parents[1]/'.venv/Lib/site-packages'
    if not (existing/'scipy').is_dir():raise
    sys.path.append(str(existing))
    from scipy.optimize import linear_sum_assignment
from torchvision.models.detection import _utils as detection_utils

class UniqueForcedSSDMatcher(detection_utils.Matcher):
    def __init__(self, threshold):
        super().__init__(threshold,threshold,allow_low_quality_matches=False)
        self.assignment_diagnostics=[]
        self.stock=detection_utils.SSDMatcher(threshold)

    def __call__(self, match_quality_matrix):
        if match_quality_matrix.ndim!=2 or not torch.isfinite(match_quality_matrix).all():
            raise ValueError('SSD assignment needs a finite GT-by-anchor IoU matrix')
        gt_count,anchor_count=match_quality_matrix.shape
        if gt_count==0:
            return torch.full((anchor_count,),self.BELOW_LOW_THRESHOLD,dtype=torch.int64,device=match_quality_matrix.device)
        if anchor_count<gt_count:
            raise ValueError('Unique forced SSD assignment needs at least one anchor per target')
        # Preserve every ordinary threshold-positive assignment except where an
        # anchor is explicitly reserved to guarantee coverage of another GT.
        matches=super().__call__(match_quality_matrix)
        stock=self.stock(match_quality_matrix)
        gt,anchors=linear_sum_assignment(-match_quality_matrix.detach().cpu().numpy().astype(np.float64))
        if len(gt)!=gt_count or len(set(anchors.tolist()))!=gt_count:
            raise ValueError('Rectangular assignment did not cover every target')
        gt_tensor=torch.as_tensor(gt,dtype=torch.int64,device=matches.device)
        anchor_tensor=torch.as_tensor(anchors,dtype=torch.int64,device=matches.device)
        matches[anchor_tensor]=gt_tensor
        counts=[int((matches==i).sum())for i in range(gt_count)]
        if not all(n>=1 for n in counts):raise ValueError('A GT lost all positive anchors')
        self.assignment_diagnostics.append({'gt_count':gt_count,'anchor_count':anchor_count,
            'stock_positive_anchor_counts':[int((stock==i).sum())for i in range(gt_count)],
            'unique_positive_anchor_counts':counts,'forced_gt_indices':gt.tolist(),'forced_anchor_indices':anchors.tolist(),
            'forced_anchor_ious':match_quality_matrix[gt_tensor,anchor_tensor].tolist()})
        return matches

def install(model):
    """Replace the matcher on this training instance; preserve the threshold."""
    model.proposal_matcher=UniqueForcedSSDMatcher(model.proposal_matcher.high_threshold)
    return model
