"""Protected SDD1 TRAIN annotation boundaries only; no NN/CAL/test input."""
from pathlib import Path
import hashlib,json
import cv2,numpy as np
from audit_factory_fit_geometry_headers import header_size

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/optimization/factory-round3'
PROTOCOL=OUT/'spatial_round3f_orientation_multiscale-protocol-before-fit.json'
PROTOCOL_SHA='c47b8e7a46f69d6018a39ebae9b25d934038c44af5250ae34ab915af52d346c3'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    cv2.setNumThreads(1)
    if sha(PROTOCOL)!=PROTOCOL_SHA:raise ValueError('Protected D fitting protocol changed')
    protocol=json.loads(PROTOCOL.read_text(encoding='utf-8'));splits=protocol['split_items']
    fitting=set(splits['train']);reserved=set(splits['calibration'])|set(splits['test'])
    if fitting&reserved or len(fitting)!=30:raise ValueError('Protected training item membership changed')
    base=ROOT/'datasets/factory/KolektorSDD'
    files=[path for item in splits['train'] for path in sorted((base/item).glob('*.jpg'))]
    if len(files)!=239:raise ValueError('Expected exactly239 protected SDD1 TRAIN views')
    views=[];components=[];source_count=0
    for path in files:
        width,height,_=header_size(path);mask_path=path.with_name(path.stem+'_label.bmp')
        truth=cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE)
        if truth is None or truth.shape!=(height,width):raise ValueError('TRAIN image/header and mask geometry differ')
        native=(truth>0).astype(np.uint8)
        source_count+=cv2.connectedComponentsWithStats(native,connectivity=8)[0]-1
        target=(cv2.resize(native.astype(np.float32),(512,1280),interpolation=cv2.INTER_AREA)>0).astype(np.uint8)
        count,_,stats,_=cv2.connectedComponentsWithStats(target,connectivity=8)
        name=path.relative_to(base).as_posix()
        views.append({'image':name,'image_sha256':sha(path),'mask_sha256':sha(mask_path),
                      'native_width':width,'native_height':height,'native_mask_shape_hw':list(truth.shape),
                      'positive':bool(target.any()),'training_grid_components':count-1})
        for index in range(1,count):
            x,y,w,h,area=map(int,stats[index]);edges={'left':x,'right':512-(x+w),'top':y,'bottom':1280-(y+h)}
            components.append({'image':name,'component':index,'bbox_xywh':[x,y,w,h],'target_area':area,
                               'bbox_distance_to_source_image_edge_pixels':edges,'min_source_image_edge_distance':min(edges.values()),
                               'full_bbox_fits_old_640x256_patch':w<=256 and h<=640,
                               'full_bbox_fits_D_square':{str(side):w<=side and h<=side for side in [256,384,512]},
                               'full_real_context_available':{str(margin):all(value>=margin for value in edges.values()) for margin in [4,8,16,32]}})
    thresholds=[0,4,8,16,32]
    by_edge={edge:{str(t):sum(r['bbox_distance_to_source_image_edge_pixels'][edge]<=t for r in components) for t in thresholds} for edge in ['left','right','top','bottom']}
    any_edge={str(t):sum(r['min_source_image_edge_distance']<=t for r in components) for t in thresholds}
    context={str(t):sum(r['min_source_image_edge_distance']>=t for r in components) for t in [4,8,16,32]}
    patches={'old_640x256':{'height':640,'width':256,'components_not_wholly_containable':sum(not r['full_bbox_fits_old_640x256_patch'] for r in components)}}
    for side in [256,384,512]:
        patches[f'D_square_{side}']={'native_side':side,'resize_to':320,'linear_scale_to_training_tensor':320/side,
            'components_not_wholly_containable':sum(not r['full_bbox_fits_D_square'][str(side)] for r in components),
            'components_without32px_context_in_patch_by_bbox_size':sum(r['bbox_xywh'][2]+64>side or r['bbox_xywh'][3]+64>side for r in components)}
    orientation_path=OUT/'attempt4-train-orientation.json';orientation=json.loads(orientation_path.read_text(encoding='utf-8'))
    wide=sum(r['bbox_xywh'][2]>256 for r in components)
    if len(components)!=orientation['components'] or wide!=orientation['components_bbox_wider_than_prior256_patch']:raise ValueError('TRAIN component/span results disagree with preserved orientation audit')
    result={'scope':'Protected239 original SDD1 TRAIN views only; actual masks and JPG headers/file hashes. No calibration/comparison pixels, no SDD2 archive/test, no NN.',
            'protocol_sha256':PROTOCOL_SHA,'training_items':splits['train'],'input_grid_wh':[512,1280],
            'target_projection':'Exact active trainer: original binary GT float32 INTER_AREA resize to512x1280, threshold>0,8-connected components.',
            'edge_distance_definition':'Integer empty-cell gap from component bounding box to full source IMAGE boundary: left=x,right=512-(x+w),top=y,bottom=1280-(y+h).0 means GT touches that image edge, not a verified mechanical part edge.',
            'summary':{'views':len(views),'positive_views':sum(r['positive'] for r in views),'source_mask_components':source_count,
                'training_grid_components':len(components),'within_distance_any_edge':any_edge,'within_distance_by_edge':by_edge,
                'full_real_context_available_all_edges':context,'bbox_width_gt256':wide,
                'min_edge_distance_min_median_max':[min(r['min_source_image_edge_distance'] for r in components),float(np.median([r['min_source_image_edge_distance'] for r in components])),max(r['min_source_image_edge_distance'] for r in components)]},
            'patch_geometry_limits':patches,'orientation_reference':{'path':str(orientation_path),'sha256':sha(orientation_path),'orientation_counts':orientation['orientation_counts']},
            'sampling_interpretation':['Containability is a necessary geometric possibility, not observed sample frequency or NN accuracy. Positive-centred jitter can still crop part of a containable component.',
                'D chooses256/384/512 real source squares uniformly and resizes isotropically320. A512square retains both horizontal source-image bounds; no square spans all1280vertical pixels.',
                'Quarter turns/flip rotate patch boundaries and labels together. Most patch borders are internal crop boundaries, not full source-image bounds; zero-padded convolutions also see these internal cut edges.',
                'The previous640x256rectangle cannot contain any component widerthan256 regardless of positive centring. Larger D squares remove that impossibility for some/all components, but may downsample thin detail.',
                'Right-source-image-edge GT is abundant:30/33 touch it, whereas no GT component is within32px of the other source edges before augmentation. This is directional boundary concentration, not generic lack of boundary GT.',
                'This only supports a future TRAIN-derived boundary/context hypothesis if a frozen model underperforms. It selects no padding, threshold or new model and makes no claim about original comparison misses.'],
            'views':views,'components':components}
    output=OUT/'attempt4-train-boundary-context-audit.json';output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    lines=['# TRAIN-only factory boundary/context audit','',result['scope'],'',
        f"The protected30 fitting items contain239 views, {sum(r['positive'] for r in views)} positive views and{len(components)} training-grid GT components. Original masks contain{source_count} components. The exact512×1280 occupancy projection matches the active trainer. These components are not counts of independent physical damage sites.",'',
        'Distances are empty pixels between each component bounding box and a FULL SOURCE IMAGE edge, not a verified mechanical part edge.0 means touching. A component contributes once to “any edge” and can contribute to multiple directional columns.','',
        '| Distance at most(px) | Any source-image edge | Left | Right | Top | Bottom |','|---:|---:|---:|---:|---:|---:|']
    for t in thresholds:lines.append(f"| {t} | {any_edge[str(t)]} | "+' | '.join(str(by_edge[e][str(t)]) for e in ['left','right','top','bottom'])+' |')
    lines.extend(['',f"Minimum-edge distance min/median/max: {result['summary']['min_edge_distance_min_median_max']}. Components with at least32 real source pixels of context on every side: {context['32']}/{len(components)}.",'',
        f"**{wide}/{len(components)} component boxes are wider than the previous256-pixel crop.** This agrees with the preserved TRAIN orientation audit ({orientation['orientation_counts']}). Flips alone do not exchange horizontal/vertical orientation.",'',
        '| Patch | Cannot wholly contain GT bbox | Cannot fit bbox plus32px each side by size |','|---|---:|---:|',
        f"| Previous640×256 | {patches['old_640x256']['components_not_wholly_containable']} | Not measured |"])
    for side in [256,384,512]:
        s=patches[f'D_square_{side}'];lines.append(f"| D {side}×{side} →320×320 | {s['components_not_wholly_containable']} | {s['components_without32px_context_in_patch_by_bbox_size']} |")
    lines.extend(['',*['- '+text for text in result['sampling_interpretation']],'',
        'No causal failure diagnosis or architecture recommendation follows from these counts. The measurable limitations are directional source-boundary concentration and patch-context truncation: right-edge examples dominate, while left/top/bottom source-image-boundary examples are absent within32px before augmentation. Quarter turns supply transformed patch views; they do not establish independently captured conditions at those other image edges. Any future hypothesis needs a separately frozen experiment after the current model is evaluated. The current model, padding, operating point and serving admission policy remain unchanged.','',
        f"[Source identities and all component distances]({output.as_posix()})"])
    doc=ROOT/'docs/factory-train-boundary-context-audit.md';doc.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'summary':result['summary'],'patch_geometry_limits':patches,'json':str(output),'markdown':str(doc)},indent=2))

if __name__=='__main__':main()
