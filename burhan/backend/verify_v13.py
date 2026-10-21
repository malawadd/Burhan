"""Two real Codex turns in an isolated database; never connects to the live service.

Run: python -m burhan.backend.verify_v13 --output <trace.json> [--image <photo>]
This uses the signed-in CLI account and spends two online image-planning turns.
"""
import argparse
import base64
import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
from fastapi.testclient import TestClient

from .app import create_app
from .config import Settings


def run():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--image', type=Path, default=root/'datasets/construction/dacl10k/images/negative_calibration/dacl10k_v2_train_5381.jpg')
    args = parser.parse_args()
    photo = cv2.imread(str(args.image))
    if photo is None:
        raise RuntimeError('Evaluation photograph is unavailable')
    height, width = photo.shape[:2]
    views = [photo, photo[height//4:3*height//4, width//4:3*width//4]]

    def image(frame):
        h, w = frame.shape[:2]
        scale = min(1, 768/max(h,w))
        frame = cv2.resize(frame, (round(w*scale), round(h*scale)))
        return 'data:image/jpeg;base64,' + base64.b64encode(cv2.imencode('.jpg',frame)[1]).decode()

    with tempfile.TemporaryDirectory(prefix='burhan-v13-codex-') as directory:
        config = Settings(data_dir=Path(directory), provider='codex', min_interval=0, timeout=150)
        with TestClient(create_app(config)) as client:
            service = client.app.state.service
            paired = client.post('/api/v1/pair', json={'code':service.pair_code,'deviceName':'isolated-v13-evaluation'})
            paired.raise_for_status()
            headers = {'Authorization':'Bearer '+paired.json()['token']}
            capabilities = {'expandedInspectionModels':True,
                'installedModelIds':['research-phele','research-fasteners','research-stain']}
            session = client.post('/api/v1/sessions', headers=headers,
                json={'inspectionId':'v13-real-codex-evaluation','language':'en', **capabilities})
            session.raise_for_status()
            sid = session.json()['id']
            selected = next(t for t in service.tasks if t.get('modelId') == 'research-phele')
            contexts = [
                {'mode':'manual','selectionMode':'confirm', **capabilities},
                {'mode':'research','selectionMode':'confirm','selectionConfirmed':True,
                 'activeTaskId':selected['id'],'correction':'The inspector selected this appearance check. The closer view is new evidence; no local detector observations are supplied. Do not claim the detector ran.', **capabilities},
            ]
            turns = []
            for index, (context, frame) in enumerate(zip(contexts, views)):
                if index:
                    event = client.post(f'/api/v1/sessions/{sid}/events', headers=headers,
                        json={'eventId':'v13-selected-1','type':'selected','taskId':selected['id'],'sceneVersion':2})
                    event.raise_for_status()
                submitted = client.post(f'/api/v1/sessions/{sid}/turns', headers=headers,
                    json={'image':image(frame),'sceneVersion':index+1,'context':context})
                submitted.raise_for_status()
                tid = submitted.json()['id']
                deadline = time.monotonic()+170
                while True:
                    response = client.get(f'/api/v1/sessions/{sid}/turns/{tid}', headers=headers)
                    response.raise_for_status()
                    result = response.json()
                    if result['status'] not in ('queued','running'):
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError('Isolated evaluation exceeded its deadline')
                    time.sleep(.5)
                turns.append(result)
                if result['status'] != 'completed':
                    break
            trace = client.get(f'/api/v1/sessions/{sid}/trace', headers=headers).json()
            client.delete(f'/api/v1/sessions/{sid}', headers=headers)
            complete = len(turns) == 2 and all(t['status']=='completed' for t in turns)
            record = {'version':'1.3.0','at':datetime.now(timezone.utc).isoformat(),
                'provider':'codex','actualOnlineInference':True,'transportMocked':False,
                'source':str(args.image.name),'physicalPhoneTest':False,'localDetectorExecuted':False,
                'success':complete,'turns':turns,'trace':trace,
                'limitations':['Backend image interpretation only; does not qualify detector accuracy, camera performance or AWS access.']}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps({'success':complete,'turns':len(turns),'output':str(args.output)}))
            if not complete:
                raise SystemExit(1)


if __name__ == '__main__':
    run()
