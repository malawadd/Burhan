"""Pure NumPy reference for exact fixed L score arithmetic; no NN imports."""
import numpy as np
from factory_stfpm_common import NORM_SCALE_FLOOR,NORM_SQUARED_FLOOR,MAP_FLOOR


def unit_feature(value):
    value=np.asarray(value)
    scale=np.maximum(np.max(np.abs(value),axis=1,keepdims=True),NORM_SCALE_FLOOR)
    scaled=value/scale
    denominator=np.sqrt(np.maximum(np.sum(scaled*scaled,axis=1,keepdims=True),NORM_SQUARED_FLOOR))
    return scaled/denominator


def disagreement(teacher,student):
    return np.clip(np.sum((unit_feature(teacher)-unit_feature(student))**2,axis=1,keepdims=True)*.25,0,1)


def geometric_mean(maps):
    return np.clip(np.exp(sum(np.log(np.clip(m,MAP_FLOOR,1.)) for m in maps)/3.),MAP_FLOOR,1.)
