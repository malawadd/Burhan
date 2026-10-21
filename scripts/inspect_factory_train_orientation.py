"""TRAIN-only mask orientation/extent audit, never an inference benchmark."""
from pathlib import Path
import json,math
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]


def main():
    cv2.setNumThreads(1)
    original=json.loads((ROOT/'models/factory/spatial_unet_v10h_sdd2-step300.onnx.json').read_text(encoding='utf-8'))
    records=[];views=positives=0
    for item in original['split_items']['train']:
        for image in sorted((ROOT/'datasets/factory/KolektorSDD'/item).glob('*.jpg')):
            truth=cv2.imread(str(image.with_name(image.stem+'_label.bmp')),0)
            if truth is None:raise ValueError('Missing real train annotation')
            views+=1
            if not np.any(truth):continue
            positives+=1;count,labels,stats,_=cv2.connectedComponentsWithStats((truth>0).astype(np.uint8),connectivity=8)
            for index in range(1,count):
                ys,xs=np.nonzero(labels==index);xy=np.stack((xs,ys),axis=1).astype(float)
                centered=xy-xy.mean(0);values,vectors=np.linalg.eigh(centered.T@centered/max(1,len(xy)))
                major=vectors[:,-1];minor=vectors[:,0]
                angle=math.degrees(math.atan2(major[1],major[0]))%180
                folded=min(angle,180-angle)
                extent=float(np.ptp(centered@major))+1
                projected=cv2.resize((labels==index).astype(np.float32),(512,1280),interpolation=cv2.INTER_AREA)>0
                _,projected_x=np.nonzero(projected)
                records.append({'image':f'{item}/{image.name}','component':index,'source_area':len(xs),
                    'source_bbox_width':int(stats[index,cv2.CC_STAT_WIDTH]),'source_bbox_height':int(stats[index,cv2.CC_STAT_HEIGHT]),
                    'train_input_bbox_width':int(np.ptp(projected_x))+1,
                    'angle_to_horizontal_degrees':folded,'orientation_bin':'horizontal' if folded<22.5 else 'vertical' if folded>=67.5 else 'diagonal',
                    'pca_elongation':float(math.sqrt(max(values[-1],0)/max(values[0],1e-9))),
                    'major_axis_extent_pixels':extent,'mean_area_per_major_extent_pixels':len(xs)/extent})
    value={'scope':'Original thirty TRAIN physical items only; no comparison pixels/filenames or calibration; annotation geometry, not NN accuracy',
        'views':views,'positive_views':positives,'components':len(records),'orientation_counts':{key:sum(r['orientation_bin']==key for r in records) for key in ['horizontal','diagonal','vertical']},
        'thin_elongated_components_pca_ratio_ge3':sum(r['pca_elongation']>=3 for r in records),
        'components_bbox_wider_than_prior256_patch':sum(r['train_input_bbox_width']>256 for r in records),
        'components_bbox_wider_than_max512_square':sum(r['train_input_bbox_width']>512 for r in records),
        'source_bbox_width_median':float(np.median([r['source_bbox_width'] for r in records])),
        'train_input_bbox_width_median':float(np.median([r['train_input_bbox_width'] for r in records])),
        'angle_definition':'PCAmajoraxis anglefolded0..90 relativehorizontal; flipsreflectangle butcannotexchangehorizontal/vertical; curvedcomponents maynotbe wellrepresentedbyoneaxis',
        'crop_history':'Prior640x256rectangles and matchingflips; no quarterturn or spatialscaleaugmentation',
        'records':records}
    path=ROOT/'output/optimization/factory-round3/attempt4-train-orientation.json';path.write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in value.items() if k!='records'},indent=2))


if __name__=='__main__':main()
