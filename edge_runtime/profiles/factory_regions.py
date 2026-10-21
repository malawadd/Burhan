"""Vectorized, real-pixel component descriptors for the local two-stage model."""
import cv2
import numpy as np

REGION_FEATURE_VERSION = 'surface_region_shape_v1'
REGION_COLUMNS = 19


def region_features(frame, scores, binary):
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return np.empty((0,REGION_COLUMNS),dtype=np.float32),labels,stats
    h,w=scores.shape
    gray=cv2.cvtColor(cv2.resize(frame,(w,h),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
    blur=cv2.GaussianBlur(gray,(21,21),0)
    residual=gray-blur
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)
    gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    flat=labels.ravel()
    area=stats[1:,cv2.CC_STAT_AREA].astype(np.float64)
    def mean(values):
        return np.bincount(flat,weights=values.ravel(),minlength=count)[1:]/area
    score_mean=mean(scores)
    gray_mean=mean(gray)
    gray_std=np.sqrt(np.maximum(mean(gray*gray)-gray_mean*gray_mean,0))
    x,y,rw,rh=[stats[1:,index].astype(np.float64) for index in range(4)]
    values=[np.log1p(area),np.log1p(rw),np.log1p(rh),rw/(rh+1),rh/(rw+1),
            area/np.maximum(rw*rh,1),x/w,y/h,(x+rw)/w,(y+rh)/h,
            centroids[1:,0]/w,centroids[1:,1]/h,score_mean,gray_mean,gray_std,
            mean(residual),mean(abs(residual)),mean(abs(gx)),mean(abs(gy))]
    return np.ascontiguousarray(np.stack(values,axis=1),dtype=np.float32),labels,stats
