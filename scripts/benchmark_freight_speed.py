"""Identical public-pixel freight OCR CPU speed/accuracy comparison; no async timing shortcuts."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import time
import cv2
import numpy as np
from edge_runtime.profiles.freight_ocr import LocalOCR, parse_document

ROOT=Path(__file__).resolve().parents[1]
CANDIDATES={
    "baseline": ("baseline", {}),
    "arena": ("optimized", {"_disable_spinning":False,"EngineConfig.onnxruntime.intra_op_num_threads":2,"Rec.rec_batch_num":6}),
    "max1024": ("optimized", {"_detector_max_side":1024,"_disable_spinning":False,"EngineConfig.onnxruntime.intra_op_num_threads":2,"Rec.rec_batch_num":6}),
    "max736": ("optimized", {"_detector_max_side":736,"_disable_spinning":False,"EngineConfig.onnxruntime.intra_op_num_threads":2,"Rec.rec_batch_num":6}),
    "batch12": ("optimized", {"_disable_spinning":False,"EngineConfig.onnxruntime.intra_op_num_threads":2,"Rec.rec_batch_num":12,"Rec.rec_img_shape":[3,48,160]}),
    "threads1": ("optimized", {"_disable_spinning":False,"Rec.rec_batch_num":6,"EngineConfig.onnxruntime.intra_op_num_threads":1}),
    "bounded1": ("optimized", {"_disable_spinning":False,"_detector_max_side":1024,"EngineConfig.onnxruntime.intra_op_num_threads":1,"Rec.rec_batch_num":12,"Rec.rec_img_shape":[3,48,160]}),
    "compact": ("optimized", {"_disable_spinning":False,"EngineConfig.onnxruntime.intra_op_num_threads":2,"Rec.rec_batch_num":1,"Rec.rec_img_shape":[3,48,48]}),
    "single": ("optimized", {"_disable_spinning":False,"EngineConfig.onnxruntime.intra_op_num_threads":2,"Rec.rec_batch_num":1}),
    "nospin": ("optimized", {"EngineConfig.onnxruntime.intra_op_num_threads":2,"Rec.rec_batch_num":1,"_disable_spinning":True}),
    "nospin4": ("optimized", {"Rec.rec_batch_num":1,"_disable_spinning":True,"EngineConfig.onnxruntime.intra_op_num_threads":4}),
    "optimized": ("optimized", {}),
}
def norm(text): return re.sub(r"[^A-Z0-9]", "", text.upper())
def samples():
    return sorted((ROOT/"datasets/freight").rglob("*.jpg"))+sorted((ROOT/"datasets/freight/invoice-pack").rglob("invoice.png"))
def assess(path, result):
    if path.suffix==".jpg":
        ground=json.loads(path.with_suffix('.json').read_text())
        words=[word['text'] for line in ground['valid_line'] for word in line['words'] if norm(word['text'])]
        return {"annotated_words":len(words),"word_substring_hits":sum(norm(w) in norm(result['text']) for w in words),
                "out_of_scope_rejection":result['document_type'] is None}
    expected=json.loads(path.with_name('expected.json').read_text())
    value=lambda k:result['fields'].get(k,{}).get('value','')
    return {"invoice_heading":result['document_type']=='invoice',"field_exact_checks":{
        'reference':value('reference')==expected['invoiceNumber'],
        'date':value('date')==expected['invoiceDate'],
        'currency':value('currency')==expected['currency'],
        'total_text':norm(value('total'))==norm(expected['currency']+' '+format(expected['total'],'.2f'))}}
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--candidates',nargs='+',choices=list(CANDIDATES),default=['baseline','optimized'])
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--limit',type=int,default=6)
    p.add_argument('--output',default='output/optimization/freight-speed.json')
    args=p.parse_args()
    cv2.setNumThreads(2)
    inputs=[(path,cv2.imread(str(path))) for path in samples()[:args.limit]]
    report={'created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'host':platform.platform(),'processor':platform.processor(),
            'opencv_threads':cv2.getNumThreads(),'network_inference':False,
            'protocol':'One reusable fresh engine per candidate. First sample first pass cold includes session init. Subsequent calls warm; repeated complete sets. Decoding, hashes and file I/O outside wall timing. Identical fixed public pixels, parser and score thresholds. No page skips or asynchronous OCR.',
            'accuracy_caveats':'Only 3 genuine CORD receipts and3 external synthetic invoice fixtures; substring hits not WER/CER. No genuine freight holdout. Runtime guidance and camera acquisition excluded; OCR page latency explicitly includes all detection, orientation and recognition.',
            'runs':{}}
    for name in args.candidates:
        preset,overrides=CANDIDATES[name]
        reader=LocalOCR(preset,overrides)
        items=[]
        for repeat in range(args.repeats):
            for index,(path,image) in enumerate(inputs):
                started=time.perf_counter()
                result=parse_document(reader.read(image))
                wall=(time.perf_counter()-started)*1000
                item={'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                      'shape':list(image.shape),'pass':repeat,'cold':repeat==0 and index==0,
                      'wall_ms':round(wall,3),'ocr_ms':result['ocr_ms'],'session_load_ms':result.get('session_load_ms'),
                      'stages_ms':result.get('ocr_stage_ms'),'result':result,'checks':assess(path,result)}
                items.append(item)
                print(f'{name} pass{repeat} {path.parent.name}/{path.name}: {wall:.1f}ms {item["checks"]}',flush=True)
        warm=[x['wall_ms'] for x in items if not x['cold']]
        last=[x for x in items if x['pass']==args.repeats-1]
        checks=[v for x in last for v in x['checks'].get('field_exact_checks',{}).values()]
        report['runs'][name]={'preset':preset,'overrides':overrides,'model':reader.info,'cold_wall_ms':items[0]['wall_ms'],
                              'warm_p50_ms':float(np.median(warm)) if warm else None,
                              'warm_p95_ms':float(np.percentile(warm,95)) if warm else None,
                              'full_last_pass_ms':sum(x['wall_ms'] for x in last),
                              'exact_fields_pass':sum(checks),'exact_fields_total':len(checks),
                              'cord_word_hits':sum(x['checks'].get('word_substring_hits',0) for x in last),
                              'cord_annotated_words':sum(x['checks'].get('annotated_words',0) for x in last),
                              'samples':items}
        reader=None
        out=ROOT/args.output
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({name:{k:v for k,v in run.items() if k not in ('samples','model')} for name,run in report['runs'].items()},indent=2))
if __name__=='__main__': main()
