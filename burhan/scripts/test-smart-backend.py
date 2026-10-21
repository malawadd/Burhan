"""Exercise the REAL configured provider through the public tunnel; never records credentials.

Run from project root with .venv/Scripts/python.exe burhan/scripts/test-smart-backend.py
--image path/to/permitted-photo.jpg. Uses local pairing.json; no credentials in CLI args.
"""
import argparse
import base64
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2

sys.stdout.reconfigure(encoding='utf-8')
parser = argparse.ArgumentParser()
parser.add_argument('--image', type=Path, required=True)
parser.add_argument('--url', default='https://merhaba-api.kiyan.finance')
parser.add_argument('--output', type=Path, default=Path('output/smart-smoke/backend-real.json'))
args = parser.parse_args()
app_root = Path(__file__).resolve().parents[1]
token = ''


def call(path, body=None, method=None):
    headers = {'Content-Type': 'application/json', 'User-Agent':'Burhan/1.1.0 (IntegrationVerification)'}
    if token:
        headers['Authorization'] = 'Bearer ' + token
    request = urllib.request.Request(args.url.rstrip('/') + '/api/v1' + path,
        data=None if body is None else json.dumps(body).encode(), headers=headers,
        method=method or ('GET' if body is None else 'POST'))
    with urllib.request.urlopen(request, timeout=25) as response:
        return json.loads(response.read())


def jpeg(image):
    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 82])
    assert ok
    return 'data:image/jpeg;base64,' + base64.b64encode(encoded).decode()


image = cv2.imread(str(args.image))
if image is None:
    raise SystemExit('Cannot read the test photo')
height, width = image.shape[:2]
if max(height, width) > 1024:
    image = cv2.resize(image, (round(width*1024/max(height,width)), round(height*1024/max(height,width))))
health = call('/health')
if health.get('provider') == 'off' or not health.get('ready'):
    raise SystemExit('A real provider must be enabled and ready for this integration test')
pair = json.loads((app_root/'backend/data/pairing.json').read_text())
token = call('/pair', {'code': pair['code'], 'deviceName':'Burhan integration verification'})['token']
session = call('/sessions', {'inspectionId':'integration-verification','language':'en','mode':'manual',
    'stage':'','taskId':'','location':'Permitted evaluation photograph'})
sid = session['id']
record = {'testedAt': time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), 'endpoint':args.url,
    'provider':health['provider'], 'source': {'file':args.image.name,
    'sha256':hashlib.sha256(args.image.read_bytes()).hexdigest()},
    'scope':'Two real provider turns; second image deliberately loses detail. Not a scene or defect accuracy benchmark.',
    'turns':[]}
try:
    blurred = cv2.resize(cv2.resize(image,(8,8),interpolation=cv2.INTER_AREA),
        (image.shape[1],image.shape[0]),interpolation=cv2.INTER_LINEAR)
    for version, snapshot in enumerate((image, blurred), start=1):
        if version > 1:
            time.sleep(10)
        job = call(f'/sessions/{sid}/turns', {'image':jpeg(snapshot),'sceneVersion':version,
            'requestId':f'real-verification-{version}', 'context':{'mode':'manual','locked':False,
            'activeTaskId':'','completedTaskIds':[],'correction':'','evidenceCount':0,'observations':[],
            'quality':{'source':'integration-test-original' if version==1 else 'integration-test-deliberately-defocused'}}})
        start = time.monotonic()
        while time.monotonic()-start < 160:
            status = call(f'/sessions/{sid}/turns/{job["id"]}')
            if status['status'] in ('completed','failed','cancelled','superseded'):
                break
            time.sleep(1)
        if status['status'] != 'completed':
            record['turns'].append(status)
            raise RuntimeError('Real provider turn did not complete: '+status.get('error',status['status']))
        result = status['result']
        assert result['needsHumanReview'] is True
        acknowledgements = [{'actionId':a['id'],'status':'accepted',
            'detail':'API verification acknowledged this suggestion; no physical inspection action was performed.'}
            for a in result['actions']]
        call(f'/sessions/{sid}/tool-results',{'turnId':job['id'],'sceneVersion':version,'results':acknowledgements})
        record['turns'].append({**status,'durationSeconds':round(time.monotonic()-start,2)})
        print(json.dumps({'turn':version,'scene':result['scene'],'guidance':result['guidance'],
            'actions':[a['type'] for a in result['actions']]}, ensure_ascii=False),flush=True)
    record['trace'] = call(f'/sessions/{sid}/trace')
    perceptions = [e['data']['context']['backendQuality'] for e in record['trace']['events'] if e['type']=='perception']
    assert len(perceptions)==2 and perceptions[1]['laplacianVariance'] < perceptions[0]['laplacianVariance']
    record['qualityRespondedTo'] = perceptions
    record['guidanceChanged'] = record['turns'][0]['result']['guidance'] != record['turns'][1]['result']['guidance']
    print(json.dumps({'passed':True,'provider':record['provider'],'guidanceChanged':record['guidanceChanged'],
        'opencvQuality':perceptions},ensure_ascii=False),flush=True)
finally:
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    call(f'/sessions/{sid}',method='DELETE')
