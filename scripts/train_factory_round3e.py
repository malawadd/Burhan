"""Refinement C: phase-supervised detail, explicitly half-grid max-logit serving."""
import train_factory_round3c as base
from train_factory_round3d import PackedPixelRefiner


if __name__=='__main__':
    base.NAME='spatial_round3e_phase_max_refine'
    base.ARCHITECTURE='Packed2x2rawgray/localcontrast detail plushalfgrid8channel learnedrefiner; fourphasefullresiduallogits trainedatfullresolution;servingmax2x2logitpooltooriginal256x640 localizationgrid;noGTinput'
    base.FullResolutionRefiner=PackedPixelRefiner
    base.SERVING_POOL=2
    base.main()
