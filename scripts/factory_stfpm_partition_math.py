"""L9 fixed float32 CPU score head; pure NumPy, no NN/runtime imports."""
import numpy as np
from factory_stfpm_common import NORM_SCALE_FLOOR,NORM_SQUARED_FLOOR,MAP_FLOOR


def unit_feature(value):
    value=np.asarray(value,dtype=np.float32)
    scale=np.maximum(np.max(np.abs(value),axis=1,keepdims=True),np.float32(NORM_SCALE_FLOOR))
    scaled=value/scale
    denom=np.sqrt(np.maximum(np.sum(scaled*scaled,axis=1,keepdims=True,dtype=np.float32),np.float32(NORM_SQUARED_FLOOR)))
    return scaled/denom


def disagreement(teacher,student):
    delta=unit_feature(teacher)-unit_feature(student)
    return np.clip(np.sum(delta*delta,axis=1,keepdims=True,dtype=np.float32)*np.float32(.25),0,1)


def bilinear_half_pixel(value,out_h,out_w):
    """ONNX/PyTorch align_corners=False half-pixel coordinates, edge clamping."""
    value=np.asarray(value,dtype=np.float32);in_h,in_w=value.shape[-2:]
    if (in_h,in_w)==(out_h,out_w):return value.copy()
    y=np.maximum((np.arange(out_h,dtype=np.float32)+np.float32(.5))*np.float32(in_h/out_h)-np.float32(.5),0)
    x=np.maximum((np.arange(out_w,dtype=np.float32)+np.float32(.5))*np.float32(in_w/out_w)-np.float32(.5),0)
    y0=np.floor(y).astype(np.intp);x0=np.floor(x).astype(np.intp)
    y1=np.minimum(y0+1,in_h-1);x1=np.minimum(x0+1,in_w-1)
    wy=(y-y0).astype(np.float32)[:,None];wx=(x-x0).astype(np.float32)[None,:]
    top=value[...,y0[:,None],x0[None,:]]*(1-wx)+value[...,y0[:,None],x1[None,:]]*wx
    bottom=value[...,y1[:,None],x0[None,:]]*(1-wx)+value[...,y1[:,None],x1[None,:]]*wx
    return top*(1-wy)+bottom*wy


def score_from_features(features):
    if len(features)!=6:raise ValueError('L9 fixed six-feature interface required')
    arrays=[np.asarray(v,dtype=np.float32) for v in features]
    if any(not np.isfinite(v).all() for v in arrays):raise ValueError('L9 encoder feature is nonfinite')
    maps=[disagreement(arrays[i],arrays[i+1]) for i in (0,2,4)]
    size=maps[0].shape[-2:]
    maps=[maps[0],bilinear_half_pixel(maps[1],*size),bilinear_half_pixel(maps[2],*size)]
    logs=[np.log(np.clip(v,np.float32(MAP_FLOOR),np.float32(1))) for v in maps]
    # Same left-associated sum/3 as original graph; no fitted normalization.
    score=np.clip(np.exp(((logs[0]+logs[1])+logs[2])/np.float32(3)),np.float32(MAP_FLOOR),np.float32(1))
    if score.dtype!=np.float32 or not np.isfinite(score).all():raise ValueError('L9 CPU head produced invalid score')
    return np.array(score,dtype=np.float32,copy=True)
