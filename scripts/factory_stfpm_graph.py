"""L: independent normal-only MobileNet feature pyramids, no reference/GT input."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_small
from factory_stfpm_common import ROOT,SEED,NORM_SCALE_FLOOR,NORM_SQUARED_FLOOR,MAP_FLOOR,sha


def unit_feature(value):
    # maxabs scaling prevents FP16 sumsquare overflow, even at finite halfmax.
    scale=value.abs().amax(dim=1,keepdim=True).clamp(min=NORM_SCALE_FLOOR)
    scaled=value/scale
    denominator=(scaled*scaled).sum(dim=1,keepdim=True).clamp(min=NORM_SQUARED_FLOOR).sqrt()
    return scaled/denominator


def disagreement(teacher,student):
    return ((unit_feature(teacher)-unit_feature(student))**2).sum(dim=1,keepdim=True).mul(.25).clamp(0.,1.)


class PyramidTeacherStudent(nn.Module):
    def __init__(self,teacher_spec):
        super().__init__()
        path=ROOT/teacher_spec['file']
        if sha(path)!=teacher_spec['sha256']:raise ValueError('Teacher file changed')
        teacher=mobilenet_v3_small(weights=None)
        teacher.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED)
            student=mobilenet_v3_small(weights=None)
        self.teacher=nn.ModuleList(list(teacher.features.children())[:7])
        self.student=nn.ModuleList(list(student.features.children())[:7])
        for parameter in self.teacher.parameters():parameter.requires_grad_(False)
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
        self.eval()

    def train_normals(self):
        self.train()
        self.teacher.eval()  # Never let recursive train() alter teacher BN.

    def pyramids(self,bgr):
        raw=(bgr.to(torch.float32)[:,[2,1,0]]/255.-self.mean)/self.std
        t,s=raw,raw
        teachers=[];students=[]
        for i,(teacher,student) in enumerate(zip(self.teacher,self.student)):
            with torch.no_grad():t=teacher(t)
            s=student(s)
            if i in (0,3,6):teachers.append(t);students.append(s)
        return teachers,students

    def normal_loss(self,bgr):
        teachers,students=self.pyramids(bgr)
        return sum(((unit_feature(t)-unit_feature(s))**2).sum(1).mean()*.5
                   for t,s in zip(teachers,students))/3.

    def forward(self,bgr):
        teachers,students=self.pyramids(bgr)
        maps=[disagreement(t,s) for t,s in zip(teachers,students)]
        size=maps[0].shape[-2:]
        maps=[maps[0]]+[F.interpolate(m,size=size,mode='bilinear',align_corners=False) for m in maps[1:]]
        return torch.exp(sum(torch.log(m.clamp(min=MAP_FLOOR,max=1.)) for m in maps)/3.).clamp(MAP_FLOOR,1.)
