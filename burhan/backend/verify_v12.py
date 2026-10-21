"""Run a real, isolated Codex inspection and save a credential-free evaluation trace.

From the repository root: python -m burhan.backend.verify_v12
This spends two signed-in Codex planning turns; it does not touch the live server.
"""
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
    source = root / 'datasets/construction/dacl10k/images/negative_calibration/dacl10k_v2_train_5381.jpg'
    image = cv2.imread(str(source))
    if image is None:
        raise RuntimeError('Evaluation photograph is unavailable')
    h,w = image.shape[:2]
    frames = [image, image[h//4:3*h//4,w//4:3*w//4]]
    def encoded(frame):
        height,width = frame.shape[:2]
        scale = min(1,768/max(height,width))
        frame = cv2.resize(frame,(round(width*scale),round(height*scale)))
        return 'data:image/jpeg;base64,' + base64.b64encode(cv2.imencode('.jpg',frame)[1]).decode()
    with tempfile.TemporaryDirectory(prefix='burhan-v12-codex-') as folder:
        settings = Settings(data_dir=Path(folder),provider='codex',min_interval=0,timeout=150)
        with TestClient(create_app(settings)) as client:
            response = client.post('/api/v1/pair',json={'code':client.app.state.service.pair_code,'deviceName':'Burhan 1.2 isolated evaluation'})
            response.raise_for_status()
            headers = {'Authorization':'Bearer '+response.json()['token']}
            def request(method,path,body=None):
                response = client.request(method,'/api/v1'+path,headers=headers,json=body)
                response.raise_for_status()
                return response.json()
            session = request('POST','/sessions',{'inspectionId':'v12-real-codex-evaluation','language':'en','mode':'manual','location':'External paved area'})
            sid = session['id']
            def turn(version,context,frame):
                created = request('POST',f'/sessions/{sid}/turns',{'sceneVersion':version,'image':encoded(frame),'context':context,'requestId':f'v12-evaluation-{version}'})
                deadline = time.monotonic()+180
                while time.monotonic()<deadline:
                    result = request('GET',f'/sessions/{sid}/turns/{created["id"]}')
                    if result['status'] not in ('queued','running'):
                        if result['status'] != 'completed':
                            raise RuntimeError(json.dumps(result))
                        return result
                    time.sleep(1)
                raise TimeoutError('Evaluation planning timed out')
            first = turn(1,{'mode':'manual','selectionMode':'confirm','selectionConfirmed':False},frames[0])
            suggestions = first['result']['suggestions']
            if not suggestions:
                raise RuntimeError('Real provider returned no selectable proposal')
            for index,task in enumerate(suggestions):
                request('POST',f'/sessions/{sid}/events',{'eventId':f'proposed-{index}','type':'proposed','taskId':task['taskId'],'sceneVersion':1,'sourceTurnId':first['id']})
            task = suggestions[0]
            request('POST',f'/sessions/{sid}/events',{'eventId':'human-selected-1','type':'selected','taskId':task['taskId'],'sceneVersion':1,'sourceTurnId':first['id']})
            second = turn(2,{'mode':task['mode'],'selectionMode':'confirm','selectionConfirmed':True,'activeTaskId':task['taskId'],
                'correction':'I selected this check myself. This is the closer view of the same area. A visible straight line might be a construction joint; guide me without assuming a confirmed crack.','evidenceCount':1},frames[1])
            if second['result'].get('guidanceTaskId') != task['taskId']:
                raise AssertionError('Guidance was not associated with the selected check')
            request('POST',f'/sessions/{sid}/events',{'eventId':'human-completed-1','type':'completed','taskId':task['taskId'],'sceneVersion':2,'sourceTurnId':second['id']})
            request('DELETE',f'/sessions/{sid}')
            trace = request('GET',f'/sessions/{sid}/trace')
            record = {'version':'1.2.0','recordedAt':datetime.now(timezone.utc).isoformat(),'provider':'codex','execution':'actual signed-in Codex CLI; online inference',
                'sourceImage':str(source.relative_to(root)),'workflow':'Explore → proposals → simulated human selection → closer evidence with user correction → completion',
                'phoneCameraExecuted':False,'bedrockExecuted':False,'first':first,'second':second,'trace':trace}
            destination = root/'output/smart-smoke/v12-codex-selection.json'
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps({'trace':str(destination),'selectedTask':task['taskId'],'firstGuidance':first['result']['guidance'],'secondGuidance':second['result']['guidance'],'changedGuidance':first['result']['guidance']!=second['result']['guidance']}))


if __name__ == '__main__':
    run()
