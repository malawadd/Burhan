import cv2
import numpy as np
import pytest
from pathlib import Path

from edge_runtime.contracts import FrameContext
from edge_runtime.profiles.freight import FreightProfile, demo_frame
from edge_runtime.profiles.freight_ocr import LocalOCR, parse_document, model_inventory


def fake_ocr(text, score=.95):
    lines=[{'text':line,'score':score,'box':[[0,i*30],[500,i*30],[500,i*30+20],[0,i*30+20]]} for i,line in enumerate(text.splitlines())]
    return {'text':text,'lines':lines,'mean_score':score,'ocr_ms':0.,'provenance':{'test':True}}


def test_english_heading_rules_extract_evidence_and_reject_ambiguous():
    result=parse_document(fake_ocr('PACKING LIST\nReference: SHP-1042\nDate: 2026-10-20\nGross Weight: 245 KG\nCurrency USD\nItem\nItem\nItem\nItem\nItem'))
    assert result['document_type']=='packing_list'
    assert result['fields']['reference']['value']=='SHP-1042'
    assert result['fields']['gross_weight']['value']=='245 KG'
    assert result['fields']['date']['verified'] is False
    assert parse_document(fake_ocr('INVOICE AND PACKING LIST\nItem\nItem\nItem'))['document_type'] is None


def test_real_mode_ignores_demo_color_marker(monkeypatch):
    profile=FreightProfile({'stable_frames':1,'ocr_async':False})
    monkeypatch.setattr(profile._ocr,'read',lambda page:fake_ocr('UNKNOWN BUSINESS PAGE\nA\nB\nC'))
    result=profile.process(demo_frame(40),FrameContext(0,0,source_id='camera:0'))
    assert result.domain_state['detected_template'] is None
    assert result.status=='needs_review'
    assert not result.domain_state['complete']


def test_digital_scan_ocr_cache_and_reset(monkeypatch):
    profile=FreightProfile({'stable_frames':1,'required_pages':['invoice'],'ocr_async':False})
    count=[]
    monkeypatch.setattr(profile._ocr,'read',lambda page:(count.append(1) or fake_ocr('INVOICE\nInvoice Number: INV-1042\nDate: 2026-10-20\nItem\nItem\nItem')))
    page=demo_frame(40)
    first=profile.process(page,FrameContext(0,0,source_id='image:local.png'))
    assert first.status=='accepted'
    assert first.domain_state['complete']
    second=profile.process(page,FrameContext(1,0,source_id='image:local.png'))
    assert second.status=='duplicate'
    assert second.metrics['ocr_cached']
    assert len(count)==1
    profile.reset()
    assert profile._ocr_result is None
    assert profile.accepted=={}


def test_low_score_never_completes_checklist(monkeypatch):
    profile=FreightProfile({'stable_frames':1,'required_pages':['invoice'],'ocr_async':False})
    monkeypatch.setattr(profile._ocr,'read',lambda page:fake_ocr('INVOICE\nItem\nItem\nItem',.4))
    result=profile.process(demo_frame(40),FrameContext(0,0,source_id='image:local.png'))
    assert result.status=='needs_review'
    assert not result.domain_state['complete']


def test_ocr_failure_is_visible_not_accepted(monkeypatch):
    profile=FreightProfile({'stable_frames':1,'ocr_async':False})
    def fail(page):raise RuntimeError('missing local model')
    monkeypatch.setattr(profile._ocr,'read',fail)
    result=profile.process(demo_frame(40),FrameContext(0,0,source_id='image:local.png'))
    assert 'missing local model' in result.domain_state['ocr_error']
    assert not result.emit_event
    assert not result.domain_state['complete']


def test_actual_public_invoice_ocr_with_network_disabled(monkeypatch):
    path=Path('datasets/freight/invoice-pack/01-simple-english/invoice.png')
    if not path.exists() or not model_inventory()['available']:
        pytest.skip('Run public asset/model setup first')
    import socket
    def blocked(*args,**kwargs):raise AssertionError('Offline inference attempted networking')
    monkeypatch.setattr(socket.socket,'connect',blocked)
    ocr=LocalOCR().read(cv2.imread(str(path)))
    parsed=parse_document(ocr)
    assert parsed['document_type']=='invoice'
    assert parsed['fields']['currency']['value']=='AED'
    assert parsed['fields']['date']['value']=='2026-08-01'
    assert ocr['lines']
    assert ocr['mean_score']>.65

def test_async_worker_nonblocking_reset_and_bounded_close(monkeypatch):
    import threading
    import time
    profile=FreightProfile({'stable_frames':1,'required_pages':['invoice']})
    started,release=threading.Event(),threading.Event()
    def slow(page):
        started.set()
        release.wait(3)
        return fake_ocr('INVOICE\nItem\nItem\nItem')
    monkeypatch.setattr(profile._ocr,'read',slow)
    page=demo_frame(40)
    begin=time.perf_counter()
    first=profile.process(page,FrameContext(0,0,source_id='image:local.png',session_id='a'))
    assert time.perf_counter()-begin<1
    assert started.wait(1)
    assert first.domain_state['ocr_pending']
    assert first.title=='Reading page locally'
    profile.reset()
    assert profile.close(timeout=.01) is False
    release.set()
    assert profile.close(timeout=2) is True
    assert profile.accepted=={}
    with pytest.raises(RuntimeError,match='closed'):
        profile.process(page,FrameContext(1,0,source_id='image:local.png',session_id='a'))


def test_async_old_session_result_is_discarded(monkeypatch):
    import threading
    profile=FreightProfile({'stable_frames':1,'required_pages':['invoice']})
    started,release=threading.Event(),threading.Event()
    def slow(page):
        started.set()
        release.wait(3)
        return fake_ocr('INVOICE\nItem\nItem\nItem')
    monkeypatch.setattr(profile._ocr,'read',slow)
    page=demo_frame(40)
    profile.process(page,FrameContext(0,0,source_id='image:a.png',session_id='a'))
    assert started.wait(1)
    result=profile.process(page,FrameContext(1,0,source_id='image:b.png',session_id='b'))
    release.set()
    profile._job['thread'].join(2)
    # Collecting the prior session job cannot accept it into this packet.
    profile.config['ocr_enabled']=False
    old_ocr=profile._ocr
    profile._ocr=None
    result=profile.process(page,FrameContext(2,0,source_id='image:b.png',session_id='b'))
    assert result.status!='accepted'
    assert profile._ocr_result is None
    assert not profile.accepted
    profile._ocr=old_ocr
    assert profile.close()

def test_finish_source_consumes_original_frame_once(monkeypatch):
    import threading
    profile=FreightProfile({'stable_frames':1,'required_pages':['invoice']})
    started,release=threading.Event(),threading.Event()
    calls=[]
    def slow(page):
        calls.append(1)
        started.set()
        release.wait(3)
        return fake_ocr('INVOICE\nItem\nItem\nItem')
    monkeypatch.setattr(profile._ocr,'read',slow)
    ctx=FrameContext(42,123.456,source_id='image:last.png',session_id='eof')
    profile.process(demo_frame(40),ctx)
    assert started.wait(1)
    assert profile.finish_source(timeout=.01) is None
    assert profile.source_completion_state()['pending_ocr']
    release.set()
    result=profile.finish_source(timeout=2)
    assert result.status=='accepted'
    assert result.metrics['source_finished']
    assert result.metrics['final_source_sequence']==42
    assert result.metrics['final_captured_monotonic']==123.456
    assert len(calls)==1
    assert profile.finish_source(timeout=0).status=='duplicate'
    assert len(calls)==1
    assert profile.close()


def test_total_prefers_currency_evidence_over_quantity_under_column_heading():
    result=parse_document(fake_ocr('INVOICE\nItem\nPrice\nTotal\n24\nEA\nSubtotal\nAED 60.00\nTotal\nAED 63.00'))
    assert result['fields']['total']['value']=='AED 63.00'

def test_parsed_document_is_json_serializable():
    import json
    result=parse_document(fake_ocr('INVOICE\nInvoice: INV-1234\nDate: 2026-10-20\nTotal\nAED 63.00'))
    assert json.loads(json.dumps(result))['heading_matches']==['invoice']
