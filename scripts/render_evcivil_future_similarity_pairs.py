"""Render every conservative similarity flag, retaining all pairs and their IDs."""
from pathlib import Path
import json
import cv2,h5py,numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'datasets/construction/evcivil-spall-future-r4'

def main():
    cv2.setNumThreads(1)
    audit=json.loads((BASE/'qa/duplicate-audit.json').read_text())
    pairs=[]
    for row in audit['findings']:
        for flag in row['non_neural_similarity_flags']:
            pairs.append({'type':'known_reference','left':row['image'],'right':flag['reference'],
                          'dhash_hamming':flag['dhash_hamming'],'normalized_gray_correlation':flag['normalized_gray_correlation']})
    for row in audit['cross_proposed_sequence_flags']:
        pairs.append({'type':'cross_proposed','left':row['left'],'right':{'path':row['right'],'kind':'proposed'},
                      'dhash_hamming':row['dhash_hamming'],'normalized_gray_correlation':row['normalized_gray_correlation']})
    cache={};rows=[]
    for index,pair in enumerate(pairs):
        ref=pair['right'];key=json.dumps(ref,sort_keys=True)
        if key not in cache:
            if ref['kind']=='exposed_original_camera':
                with h5py.File(ROOT/ref['path'],'r') as data:frame=data['frames'][ref['frame_index']]
            else:frame=cv2.imread(str((BASE if ref['kind']=='proposed' else ROOT)/ref['path']))
            cache[key]=frame
        left=cv2.imread(str(BASE/pair['left']));right=cache[key]
        panels=[]
        for title,frame in [('NEW '+Path(pair['left']).stem,left),('MATCH '+Path(ref['path']).stem+(':'+str(ref['frame_index']) if 'frame_index' in ref else ''),right)]:
            if frame.ndim==2:frame=cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
            image=cv2.resize(frame,(320,240));header=np.zeros((32,320,3),np.uint8)
            cv2.putText(header,f'PAIR {index:03d} '+title[:44],(2,12),cv2.FONT_HERSHEY_SIMPLEX,.32,(255,255,255),1)
            cv2.putText(header,f"dHash {pair['dhash_hamming']}, corr {pair['normalized_gray_correlation']:.3f} - not model output",(2,27),cv2.FONT_HERSHEY_SIMPLEX,.32,(255,255,255),1)
            panels.append(np.vstack([header,image]))
        pair['pair_id']=index;pair['page']=index//8+1;pair['row_on_page']=index%8
        rows.append(np.hstack(panels))
    for page in range((len(rows)+7)//8):cv2.imwrite(str(BASE/f'qa/all-similarity-pairs-{page+1:02d}.jpg'),np.vstack(rows[page*8:page*8+8]))
    (BASE/'qa/all-similarity-pairs-index.json').write_text(json.dumps(pairs,indent=2)+'\n',encoding='utf-8')
    print('All pairs',len(pairs),'pages',(len(pairs)+7)//8)

if __name__=='__main__':main()
