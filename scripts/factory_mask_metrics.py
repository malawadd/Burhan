"""Original-resolution annotation metrics for thresholded factory score maps."""
import cv2
import numpy as np

def original_pixel_weights(annotation,grid_shape):
    """Exact nearest-projection pixel counts per analysis cell, computed once."""
    height,width=grid_shape
    if height*width>=2**24:
        raise ValueError('Analysis grid exceeds exact float32 index range')
    index=np.arange(height*width,dtype=np.float32).reshape(height,width)
    native=cv2.resize(index,(annotation.shape[1],annotation.shape[0]),interpolation=cv2.INTER_NEAREST).astype(np.int32)
    area=np.bincount(native.ravel(),minlength=height*width)
    true=np.bincount(native.ravel(),weights=(annotation>0).ravel().astype(np.float64),minlength=height*width)
    return area,true


def component_mask_statistics(scores,threshold,annotation,projection_weights=None):
    count,labels,stats,_=cv2.connectedComponentsWithStats((scores>=threshold).astype(np.uint8),connectivity=8)
    if projection_weights is not None:
        area,true=projection_weights
        area_native=np.bincount(labels.ravel(),weights=area,minlength=count)
        true_native=np.bincount(labels.ravel(),weights=true,minlength=count)
        return stats[:,cv2.CC_STAT_AREA],area_native,true_native,int(np.count_nonzero(annotation))
    # Nearest projection follows the analysis grid back to actual source pixels.
    native=cv2.resize(labels.astype(np.float32),(annotation.shape[1],annotation.shape[0]),interpolation=cv2.INTER_NEAREST).astype(np.int32)
    area_native=np.bincount(native.ravel(),minlength=count)
    true_native=np.bincount(native.ravel(),weights=(annotation>0).ravel().astype(np.float64),minlength=count)
    return stats[:,cv2.CC_STAT_AREA],area_native,true_native,int(np.count_nonzero(annotation))

def at_area(statistics,minimum_area):
    grid_area,native_area,native_true,target_count=statistics
    keep=grid_area>=minimum_area;keep[0]=False
    indices=np.flatnonzero(keep)
    if len(indices)>20:
        retained=indices[np.argsort(-grid_area[indices],kind='stable')[:20]]
        keep[:]=False;keep[retained]=True  # Match runtime's largest-20 candidate cap.
    predicted=int(native_area[keep].sum());intersection=int(native_true[keep].sum())
    union=target_count+predicted-intersection
    return {'flagged':bool(keep.any()),'localized':intersection>0,'foreground_tp':intersection,
            'foreground_predicted':predicted,'foreground_target':target_count,
            'foreground_pixel_recall':intersection/target_count if target_count else None,
            'foreground_pixel_precision':intersection/predicted if predicted else 0.,
            'pixel_iou':intersection/union if union else None}

def aggregate(rows):
    positives=[row for row in rows if row['defective']]
    normal=[row for row in rows if not row['defective']]
    normal_scored=[row for row in normal if not row.get('abstained',False)]
    tp=sum(row['localized'] for row in positives)
    fp=sum(row['flagged'] for row in normal)
    pixel_tp=sum(row['foreground_tp'] for row in rows)
    pixel_pred=sum(row['foreground_predicted'] for row in rows)
    pixel_gt=sum(row['foreground_target'] for row in rows)
    return {'localized_tp':tp,'localized_fn':len(positives)-tp,'normal_fp':fp,'normal_tn':len(normal_scored)-fp,
            'localized_view_recall':tp/max(1,len(positives)),
            'flagged_image_recall':sum(row['flagged'] for row in positives)/max(1,len(positives)),
            'normal_view_false_positive_rate':fp/max(1,len(normal_scored)),
            'abstained_defective':sum(row.get('abstained',False) for row in positives),
            'abstained_normal':sum(row.get('abstained',False) for row in normal),
            'foreground_pixel_recall':pixel_tp/max(1,pixel_gt),'foreground_pixel_precision':pixel_tp/max(1,pixel_pred),
            'micro_pixel_iou':pixel_tp/max(1,pixel_gt+pixel_pred-pixel_tp),
            'foreground_tp':pixel_tp,'foreground_predicted':pixel_pred,'foreground_target':pixel_gt,
            'defective_views_flagged_without_localization':sum(row['flagged'] and not row['localized'] for row in positives)}
