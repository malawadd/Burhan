"""Real TRAIN-mask component geometry only; no NN or comparison data."""
from pathlib import Path
import argparse,hashlib,json
import cv2,numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--full-resolution',action='store_true');args=parser.parse_args()
    cv2.setNumThreads(1)
    metadata=json.loads((ROOT/'models/factory/spatial_unet_v10h_sdd2-step300.onnx.json').read_text(encoding='utf-8'))
    source=[];grid=[];positive=0;views=0
    for item in metadata['split_items']['train']:
        for image in sorted((ROOT/'datasets/factory/KolektorSDD'/item).glob('*.jpg')):
            truth=cv2.imread(str(image.with_name(image.stem+'_label.bmp')),0)
            if truth is None:raise ValueError('Missing original training annotation')
            views+=1
            if not np.any(truth):continue
            positive+=1
            source.extend(cv2.connectedComponentsWithStats((truth>0).astype(np.uint8),connectivity=8)[2][1:,cv2.CC_STAT_AREA].tolist())
            grid_shape=(512,1280) if args.full_resolution else (256,640)
            occupied=(cv2.resize((truth>0).astype(np.float32),grid_shape,interpolation=cv2.INTER_AREA)>0).astype(np.uint8)
            grid.extend(cv2.connectedComponentsWithStats(occupied,connectivity=8)[2][1:,cv2.CC_STAT_AREA].tolist())
    def summary(values):return {'regions':len(values),'min':min(values),'p10':float(np.percentile(values,10)),'median':float(np.median(values)),
                                'p90':float(np.percentile(values,90)),'max':max(values),'below_area':{str(k):sum(v<k for v in values) for k in ([2,4,8,16,32,64,128] if args.full_resolution else [2,4,8,16,32])}}
    value={'scope':'Original SDD1 thirty TRAIN physical items only; true mask components, not inferred detections or chosen comparison failures',
           'training_items':metadata['split_items']['train'],'views':views,'positive_views':positive,
           'source_pixel_components':summary(source),('output_512x1280_any_occupied_cell_components' if args.full_resolution else 'output_256x640_any_occupied_cell_components'):summary(grid),
           'qualification':'Training label geometry only; occupancy can merge source components. Does not measure NN recall or imply every source component is a separate physical defect.'}
    output=ROOT/'output/optimization/factory-round3'/('attempt3-train-mask-geometry.json' if args.full_resolution else 'attempt2-train-mask-geometry.json');output.write_text(json.dumps(value,indent=2),encoding='utf-8');print(json.dumps(value,indent=2))


if __name__=='__main__':main()
