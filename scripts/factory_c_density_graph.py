"""C200 exact paired semantic/fine forward plus F2-corrected density arithmetic."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from train_factory_round3d import PackedPixelRefiner
from factory_c_density_common import ROOT, SOURCE, verify_source
from factory_c_density_math import dummy_bank


class CNormalDensityGraph(nn.Module):
    def __init__(self, bank=None, alpha=1., beta=4.):
        super().__init__()
        verify_source()
        self.base = PackedPixelRefiner(ROOT/'models/factory/encoder_sources/mobilenet_v3_small-047dcff4.pth')
        self.base.load_state_dict(torch.load(SOURCE,map_location='cpu',weights_only=True),strict=True)
        bank = dummy_bank() if bank is None else bank
        for key in ('mean','whitening','q95','qspan'):
            value = np.asarray(bank[key])
            if not np.isfinite(value).all() or not np.array_equal(value,value.astype(np.float16).astype(value.dtype)):
                raise ValueError('Density constants must be finite and exactly half-representable')
        self.register_buffer('normal_mean',torch.tensor(np.asarray(bank['mean']),dtype=torch.float32)[None,:,None,:])
        self.register_buffer('whitening',torch.tensor(np.asarray(bank['whitening']),dtype=torch.float32)[:,:,None,None])
        self.register_buffer('q95',torch.tensor(float(bank['q95']),dtype=torch.float32))
        self.register_buffer('qspan',torch.tensor(float(bank['qspan']),dtype=torch.float32))
        self.alpha,self.beta = float(alpha),float(beta)
        if self.normal_mean.shape != (1,16,1,256) or self.whitening.shape != (16,16,1,1) or float(self.qspan)<=0 or float(torch.abs(self.whitening).sum(1).max())>8.001:
            raise ValueError('Invalid normal bank geometry/scale')
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def parts(self,bgr):
        # Exact PackedPixelRefiner.forward operation order, with the original
        # C serving reduction after the full-resolution residual addition.
        base = self.base
        raw=bgr.to(torch.float32)/255.
        x=(raw[:,[2,1,0]]-base.mean)/base.std
        fine=middle=None
        for index,block in enumerate(base.encoder):
            x=block(x)
            if index==0:fine=x
            elif index==3:middle=x
        coarse=x
        for block in base.context:x=block(x)
        coarse=coarse+F.interpolate(base.context_projection(x),size=coarse.shape[-2:],mode='bilinear',align_corners=False)
        x=F.interpolate(base.coarse(coarse),size=middle.shape[-2:],mode='bilinear',align_corners=False)
        x=base.middle_spatial(F.relu(x+base.middle(middle)))
        x=F.interpolate(x,size=fine.shape[-2:],mode='bilinear',align_corners=False)
        features=base.fine_spatial(F.relu(x+base.fine(fine)))
        logits=F.interpolate(base.classifier(features),size=bgr.shape[-2:],mode='bilinear',align_corners=False)
        gray=raw.mean(1,keepdim=True)
        detail=gray-F.avg_pool2d(gray,5,stride=1,padding=2,count_include_pad=False)
        phases=F.pixel_unshuffle(torch.cat((gray,detail),dim=1),2)
        residual=F.pixel_shuffle(base.refine(torch.cat((features,phases),dim=1)),2)
        return F.max_pool2d(logits+residual,2),fine

    def evidence(self,fine):
        centered=torch.clamp(fine-self.normal_mean,-32.,32.)
        white=torch.clamp(F.conv2d(centered,self.whitening),-32.,32.)
        distance=torch.mean(white*white,dim=1,keepdim=True)
        return torch.clamp((torch.log(torch.clamp(distance,min=1e-6))-self.q95)/self.qspan,-8.,8.)

    def forward(self,bgr):
        logits,fine=self.parts(bgr)
        return logits*self.alpha if self.beta==0. else self.alpha*logits+self.beta*self.evidence(fine)


class CDiagnosticDensityGraph(CNormalDensityGraph):
    def forward(self,bgr):
        logits,fine=self.parts(bgr)
        evidence=self.evidence(fine)
        return self.alpha*logits+self.beta*evidence,evidence,fine
