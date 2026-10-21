"""Fixed DACL appearance transfer on real wall photos; no training or videos.

Leakage->wetspot and abscission->spalling are explicitly broad appearance-proxy
comparisons, not material/cause verification or moisture/paint measurements.
"""
from pathlib import Path
import argparse,hashlib,json,sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from edge_runtime.profiles.construction_dacl import DaclConstructionModel,CLASSES,CAPABILITIES
from evaluate_construction_defect_video import match
DATA=ROOT/'datasets/construction/finishes-expansion/mbdd2025-wall-photos'
OUT=ROOT/'output/evaluation/construction-finishes-wall-dacl'
MODEL=ROOT/'models/construction/dacl19/dacl-segformer-b1-512.onnx'
MAP={'leakage':'wetspot','abscission':'spalling'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['freeze','evaluate'],required=True);args=parser.parse_args()
    cv2.setNumThreads(1);OUT.mkdir(parents=True,exist_ok=True)
    manifest_path=DATA/'asset-manifest.json';manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    protocol_path=OUT/'protocol-before-inference.json'
    if args.stage=='freeze':
        protocol={'source_manifest_sha256':sha(manifest_path),'model_sha256':sha(MODEL),'script_sha256':sha(Path(__file__)),
                  'adapter_sha256':sha(ROOT/'edge_runtime/profiles/construction_dacl.py'),'photos':[r['image'] for r in manifest['records']],
                  'point':{'probability_strictly_greater_than':.5,'minimum_grid_component':4,'input_size':512,'classes':['wetspot','spalling']},
                  'proxy_label_crosswalk':MAP,'proxy_scope':'Publisher leakage visible marks compared to wetspot; publisher broad abscission compared to materialloss. No activewater/paint-only/concrete-only groundtruth.',
                  'evaluation':'Classaware one-to-one nativeboxIoU>=.50, sourceXYXY convertedXYWH; publisherboxes donot support pixelmask accuracy. Full18 images/eachclass, class-absent flags separately; no thresholdsearch/ROI.',
                  'independence':'Correlated exterior photographs;9487/9501 visibly samewall. DACLcheckpoint-source overlap unknown. This is new-image cross-source diagnostic, not blindproject-independent qualification.',
                  'license':'Sourcephotos CCBY4. DACLtrainingdataCCBYNC4; checkpoint separatelicenseundeclared. No video release inMBDDZIP.'}
        if protocol_path.exists() and json.loads(protocol_path.read_text(encoding='utf-8'))!=protocol:raise RuntimeError('Frozen wall diagnostic changed')
        write(protocol_path,protocol);print('FROZEN',sha(protocol_path),flush=True);return
    if (OUT/'report.json').exists():raise RuntimeError('Preserve completed wall diagnostic')
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    for path,expected in [(manifest_path,protocol['source_manifest_sha256']),(MODEL,protocol['model_sha256']),
                          (Path(__file__),protocol['script_sha256']),(ROOT/'edge_runtime/profiles/construction_dacl.py',protocol['adapter_sha256'])]:
        if sha(path)!=expected:raise RuntimeError('Protected wall diagnostic input differs')
    spec={'file':MODEL.relative_to(ROOT/'models/construction').as_posix(),'sha256':sha(MODEL),'input_size':512,'classes':CLASSES,'capabilities':CAPABILITIES,
          'source':'https://huggingface.co/spaces/phiyodr/dacl-challenge/tree/f5bee34782e7e02393ba4f206a95ba5ed67eb8a5',
          'license':protocol['license']}
    model=DaclConstructionModel({'damage_variant':'dacl_segformer_b1','engine':'openvino','openvino_device':'GPU','cpu_threads':1,
                                'damage_classes':['wetspot','spalling'],'confidence':.5,'minimum_component_pixels':4,'mask_rendering':'masks'},spec)
    records=[];first=set()
    for row in manifest['records']:
        path=DATA/row['image']
        if sha(path)!=row['image_sha256']:raise RuntimeError('Originalwallphoto changed')
        frame=cv2.imread(str(path));h,w=frame.shape[:2]
        expected=[]
        for index,b in enumerate(row['boxes']):
            if b['label'] not in MAP:continue
            x1,y1,x2,y2=b['source_xyxy']
            expected.append({'id':row['filename']+':'+str(index),'label':MAP[b['label']],'source_label':b['label'],'bbox':[x1,y1,x2-x1,y2-y1]})
        actual=model.infer(frame)
        predictions=[{'id':i+1,'label':p['label'],'bbox':list(p['bbox']),'score':p['score']} for i,p in enumerate(actual)]
        matches,misses=match(predictions,expected)
        counts={name:{'tp':sum(bool(p.get('matched_annotation_id')) for p in predictions if p['label']==name),
                      'fn':sum(e['label']==name for e in misses),
                      'fp':sum(not p.get('matched_annotation_id') for p in predictions if p['label']==name),
                      'annotations':sum(e['label']==name for e in expected)} for name in MAP.values()}
        records.append({'image':row['image'],'image_sha256':row['image_sha256'],'frame_size_wh':[w,h],'expected':expected,'predictions':predictions,'by_class':counts})
        for name in MAP.values():
            if counts[name]['annotations'] and name not in first:
                panels=[frame.copy(),frame.copy(),frame.copy()]
                for e in expected:
                    if e['label']!=name:continue
                    x,y,bw,bh=e['bbox'];cv2.rectangle(panels[1],(round(x),round(y)),(round(x+bw),round(y+bh)),(0,170,255),2)
                for p in actual:
                    if p['label']!=name:continue
                    panels[2][p['mask']]=(.45*panels[2][p['mask']]+.55*np.array([0,170,255])).astype(np.uint8)
                canvas=np.full((310,1440,3),22,np.uint8)
                for i,panel in enumerate(panels):
                    canvas[28:298,i*480:(i+1)*480]=cv2.resize(panel,(480,270),interpolation=cv2.INTER_AREA)
                    cv2.putText(canvas,['RAW PHOTO','PUBLISHER BOXES (APPEARANCE PROXY)','ACTUAL LOCAL MODEL MASK'][i],(i*480+5,18),cv2.FONT_HERSHEY_SIMPLEX,.42,(235,235,235),1)
                cv2.imwrite(str(OUT/(name+'-first-raw-gt-model.jpg')),canvas);first.add(name)
        print(row['filename'],counts,flush=True)
    summaries={}
    for name in MAP.values():
        counts={k:sum(r['by_class'][name][k] for r in records) for k in ['tp','fn','fp','annotations']}
        negatives=[r for r in records if not r['by_class'][name]['annotations']]
        counts.update(precision=counts['tp']/(counts['tp']+counts['fp']) if counts['tp']+counts['fp'] else None,
                      recall=counts['tp']/counts['annotations'] if counts['annotations'] else None,
                      class_absent_photos=len(negatives),class_absent_flagged=sum(any(p['label']==name for p in r['predictions']) for r in negatives))
        summaries[name]=counts
    write(OUT/'report.json',{'protocol_sha256':sha(protocol_path),'model':model.describe(),'scope':protocol['proxy_scope'],'by_class':summaries,'records':records})
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__':main()
