"""Evaluate offline OCR on public samples without treating receipts as freight documents."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import re
import statistics
import time
import cv2
from edge_runtime.contracts import FrameContext
from edge_runtime.profiles.freight import FreightProfile
from edge_runtime.profiles.freight_ocr import LocalOCR, parse_document, model_inventory

ROOT=Path(__file__).resolve().parents[1]

def norm(text):
    return re.sub(r"[^A-Z0-9]", "", text.upper())

def main():
    if "--reparse" in sys.argv:
        out=ROOT/'output/evaluation/freight-public.json'
        report=json.loads(out.read_text(encoding='utf-8'))
        for item in report['samples']:
            item['result']=parse_document(item['result'])
            if 'fixture_expected' in item:
                expected=item['fixture_expected']
                fields=item['result']['fields']
                value=lambda key:fields.get(key,{}).get('value','')
                item['field_exact_checks']={
                    'reference':value('reference')==expected['invoiceNumber'],
                    'date':value('date')==expected['invoiceDate'],
                    'currency':value('currency')==expected['currency'],
                    'total_text':norm(value('total'))==norm(expected['currency']+' '+format(expected['total'],'.2f')),
                }
        report['extraction_reparsed_from_saved_actual_ocr']=True
        out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
        print(json.dumps([{'path':x['path'],'fields':x.get('field_exact_checks')} for x in report['samples']],indent=2))
        return
    cv2.setNumThreads(2)
    reader=LocalOCR()
    report={"model":model_inventory(), "note":"First three CORD test rows plus three externally authored synthetic invoice fixtures. Smoke subset only; not freight benchmark accuracy.","samples":[]}
    for path in sorted((ROOT/"datasets/freight").rglob("*.jpg"))+sorted((ROOT/"datasets/freight/invoice-pack").rglob("invoice.png")):
        image=cv2.imread(str(path))
        start=time.perf_counter()
        result=parse_document(reader.read(image))
        item={"path":str(path.relative_to(ROOT)),"shape":list(image.shape),"wall_ms":round((time.perf_counter()-start)*1000,2),"result":result}
        if path.suffix==".jpg":
            ground=json.loads(path.with_suffix('.json').read_text())
            words=[word['text'] for line in ground['valid_line'] for word in line['words'] if norm(word['text'])]
            recognized=norm(result['text'])
            # Includes substring matches and isn't a CER/WER; preserve the exact definition.
            item['annotated_word_substring_recall']=round(sum(norm(word) in recognized for word in words)/max(len(words),1),4)
            item['annotated_words']=len(words)
            item['out_of_scope_not_accepted_as_freight']=result['document_type'] is None
        else:
            expected=json.loads(path.with_name('expected.json').read_text())
            item['fixture_expected']=expected
            item['invoice_heading_recognized']=result['document_type']=='invoice'
            fields=result['fields']
            value=lambda key:fields.get(key,{}).get('value','')
            item['field_exact_checks']={
                'reference':value('reference')==expected['invoiceNumber'],
                'date':value('date')==expected['invoiceDate'],
                'currency':value('currency')==expected['currency'],
                'total_text':norm(value('total'))==norm(expected['currency']+' '+format(expected['total'],'.2f')),
            }
        report['samples'].append(item)
        print(str(path.relative_to(ROOT))+': '+str(item['wall_ms'])+' ms',flush=True)
    # Process the public digital invoice through the actual profile contract, not reader only.
    path=ROOT/'datasets/freight/invoice-pack/01-simple-english/invoice.png'
    profile=FreightProfile({'stable_frames':2,'required_pages':['invoice'],'ocr_async':False})
    for i in range(2):
        result=profile.process(cv2.imread(str(path)),FrameContext(i,time.monotonic(),source_id='image:'+str(path),session_id='public-validation'))
    duplicate=profile.process(cv2.imread(str(path)),FrameContext(2,time.monotonic(),source_id='image:'+str(path),session_id='public-validation'))
    report['runtime_profile']={'accepted':result.status=='accepted','checklist_complete':result.domain_state['complete'],'duplicate_suppressed':duplicate.status=='duplicate' and not duplicate.emit_event,'fields':result.domain_state['ocr']['fields'] if result.domain_state['ocr'] else {},'error':result.domain_state['ocr_error']}
    out=ROOT/'output/evaluation/freight-public.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'report':str(out),'samples':[{k:v for k,v in x.items() if k not in ('result','fixture_expected')} for x in report['samples']],'runtime_profile':report['runtime_profile']},indent=2))

if __name__=='__main__':main()
