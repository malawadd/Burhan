"""Refinement B: keep learned spatial work coarse, preserve2x2 raw pixel phases."""
import torch
from torch import nn
from torch.nn import functional as F
import train_factory_round3c as base


class PackedPixelRefiner(base.ContextSpatialUNet):
    def __init__(self,pretrained):
        super().__init__(pretrained)
        self.refine=nn.Sequential(nn.Conv2d(20,8,1),nn.ReLU(),
            nn.Conv2d(8,8,3,padding=1,groups=8),nn.ReLU(),
            nn.Conv2d(8,8,1),nn.ReLU(),nn.Conv2d(8,4,1))
        nn.init.zeros_(self.refine[-1].weight);nn.init.zeros_(self.refine[-1].bias)

    def forward(self,bgr):
        raw=bgr.to(torch.float32)/255.
        x=(raw[:,[2,1,0]]-self.mean)/self.std
        fine=middle=None
        for index,block in enumerate(self.encoder):
            x=block(x)
            if index==0:fine=x
            elif index==3:middle=x
        coarse=x
        for block in self.context:x=block(x)
        coarse=coarse+F.interpolate(self.context_projection(x),size=coarse.shape[-2:],mode='bilinear',align_corners=False)
        x=F.interpolate(self.coarse(coarse),size=middle.shape[-2:],mode='bilinear',align_corners=False)
        x=self.middle_spatial(F.relu(x+self.middle(middle)))
        x=F.interpolate(x,size=fine.shape[-2:],mode='bilinear',align_corners=False)
        features=self.fine_spatial(F.relu(x+self.fine(fine)))
        logits=F.interpolate(self.classifier(features),size=bgr.shape[-2:],mode='bilinear',align_corners=False)
        gray=raw.mean(1,keepdim=True)
        detail=gray-F.avg_pool2d(gray,5,stride=1,padding=2,count_include_pad=False)
        phases=F.pixel_unshuffle(torch.cat((gray,detail),dim=1),2)
        residual=F.pixel_shuffle(self.refine(torch.cat((features,phases),dim=1)),2)
        return logits+residual


if __name__=='__main__':
    base.NAME='spatial_round3d_packed_refine'
    base.ARCHITECTURE='Compact context U-Net withhalfgrid20-to8 pointwise,depthwise3x3,pointwise8,4phasehead;12decoderfeatures+pixelunshuffled2x2rawmeanand5x5localcontrast;pixelshuffle2tofullresiduallogits;noGTinput'
    base.FullResolutionRefiner=PackedPixelRefiner
    base.main()
