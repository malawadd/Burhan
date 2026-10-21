"""Verify actual local OCR completion after a short public one-page replay ends."""
from pathlib import Path
import json
import shutil
import sys
import tempfile
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from edge_runtime.runtime import EdgeRuntime,RuntimeConfig


def main():
    sample=ROOT/'datasets/freight/invoice-pack/01-simple-english/invoice.png'
    def deny(*args,**kwargs):
        raise AssertionError('Outbound socket attempted')
    with tempfile.TemporaryDirectory(prefix='inspectflow-eof-') as folder:
        folder=Path(folder)
        pages=folder/'pages';pages.mkdir()
        shutil.copyfile(sample,pages/'invoice.png')
        runtime=EdgeRuntime(folder/'events')
        try:
            with patch('socket.socket.connect',deny),patch('socket.create_connection',deny):
                runtime.start(RuntimeConfig(profile='freight',source='images:'+str(pages),
                    input_fps=2,source_hold_seconds=1,deadline_ms=2000,
                    profile_options={'required_pages':['invoice'],'stable_frames':2}))
                deadline=time.monotonic()+65
                while time.monotonic()<deadline:
                    state=runtime.snapshot()
                    if state['state']=='eof':break
                    time.sleep(.05)
                else:raise RuntimeError('Final document did not complete within 65 seconds')
                assert not state['error'],state
                assert state['metrics']['captured']==state['metrics']['processed']==2,state
                assert state['metrics']['async_completions']==1,state
                result=state['last_result']
                assert result['async_source_completion'] is True
                assert result['sequence']==2
                assert result['host_receipt_to_decision_ms'] is None
                assert result['domain_state']['complete'] is True
                assert result['domain_state']['ocr']['fields']['reference']['value']=='INV-2026-001'
                report={'public_input':str(sample.relative_to(ROOT)),
                        'note':'Public synthetic invoice fixture, copied into a finite one-page folder. Actual offline OCR, not synthetic marker detection.',
                        'network_connections_denied':True,'saved_media':False,
                        'state':state}
        finally:runtime.close()
    target=ROOT/'output/evaluation/document-eof-smoke.json'
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'state':state['state'],'captured':state['metrics']['captured'],
        'processed':state['metrics']['processed'],'async_completions':state['metrics']['async_completions'],
        'guidance_p95_ms':state['metrics']['p95_ms'],
        'ocr_job_ms':result['domain_state']['ocr']['ocr_job_ms']},indent=2))


if __name__=='__main__':main()
