import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from burhan.backend.app import create_app
from burhan.backend.config import Settings
from burhan.backend.models import Plan
from burhan.backend.providers import BedrockAgent, CodexAgent, ProviderFailure, plan_schema


def plan(mode='crack'):
    task = {'crack':'visual-cracks','surface':'visual-surfaces','ppe':'visual-ppe'}[mode]
    return Plan.model_validate({'scene':{'environment':'indoor','description':'A visible wall','confidence':0.7},
        'guidance':'Move closer and hold still.','reason':'Check the possible surface line.',
        'suggestions':[{'taskId':task,'title':'Visible surface','mode':mode,'reason':'Visible wall','automatic':True}],
        'actions':[{'id':'view-1','type':'request_view','mode':mode,'taskId':task,'instruction':'Capture a close-up.'}],
        'needsHumanReview':True})


class Agent:
    def __init__(self, delay=0, error=None, result=None):
        self.delay, self.error, self.result = delay, error, result
        self.contexts = []

    def ready(self):
        return True

    async def plan(self, context, snapshot):
        import asyncio
        self.contexts.append(context)
        await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return self.result or plan(), {'engine':'test-fixture'}


def image():
    data = np.zeros((80,120,3), np.uint8)
    cv2.line(data,(20,5),(60,75),(255,255,255),2)
    return 'data:image/jpeg;base64,' + base64.b64encode(cv2.imencode('.jpg',data)[1]).decode()


def connect(client):
    code = client.app.state.service.pair_code
    result = client.post('/api/v1/pair',json={'code':code,'deviceName':'test-phone'})
    assert result.status_code == 200
    headers = {'Authorization':'Bearer '+result.json()['token']}
    session = client.post('/api/v1/sessions',headers=headers,json={'inspectionId':'inspection1','language':'en','mode':'crack'})
    assert session.status_code == 201
    return headers, session.json()['id']


def submit(client, headers, sid, version=1, **kwargs):
    return client.post(f'/api/v1/sessions/{sid}/turns',headers=headers,
        json={'image':image(),'sceneVersion':version,'context':{'mode':'crack'},**kwargs})


def wait(client, headers, sid, tid):
    for _ in range(200):
        row = client.get(f'/api/v1/sessions/{sid}/turns/{tid}',headers=headers).json()
        if row['status'] not in ('queued','running'):
            return row
        time.sleep(.01)
    raise AssertionError('Worker did not finish')


def settings(tmp_path, **kwargs):
    return Settings(data_dir=tmp_path,min_interval=0,**kwargs)


def test_pair_auth_and_ownership(tmp_path):
    with TestClient(create_app(settings(tmp_path),Agent())) as client:
        assert client.get('/api/v1/health').json()['provider'] == 'codex'
        assert client.post('/api/v1/sessions',json={'inspectionId':'x'}).status_code == 401
        assert client.post('/api/v1/pair',json={'code':'BADBAD','deviceName':'x'}).status_code == 401
        headers, sid = connect(client)
        other, _ = connect(client)
        assert submit(client,other,sid).status_code == 404
        assert client.delete('/api/v1/sessions/'+sid,headers=headers).status_code == 200
        assert submit(client,headers,sid).status_code == 410
        assert client.get(f'/api/v1/sessions/{sid}/trace',headers=headers).status_code == 200
        assert client.get(f'/api/v1/sessions/{sid}/trace',headers=other).status_code == 404
        db = client.app.state.service.db
        assert db.execute('SELECT token_hash FROM devices').fetchone()[0] != headers['Authorization'][7:]


def test_real_quality_plan_tools_and_memory(tmp_path):
    agent = Agent()
    with TestClient(create_app(settings(tmp_path),agent)) as client:
        headers,sid = connect(client)
        first = submit(client,headers,sid,requestId='retry-one')
        assert first.status_code == 202
        result = wait(client,headers,sid,first.json()['id'])
        assert result['status'] == 'completed'
        assert result['result']['needsHumanReview'] is True
        assert agent.contexts[0]['context']['backendQuality']['opencvVersion'].startswith('5.')
        retry = submit(client,headers,sid,requestId='retry-one')
        assert retry.json()['id'] == first.json()['id']
        tool = {'turnId':first.json()['id'],'sceneVersion':1,'results':[{'actionId':'view-1','status':'completed','detail':'Close-up captured'}]}
        assert client.post(f'/api/v1/sessions/{sid}/tool-results',headers=headers,json=tool).json()['accepted']
        next_turn = submit(client,headers,sid,image=None,context={'mode':'crack','correction':'It is a construction joint.'})
        assert wait(client,headers,sid,next_turn.json()['id'])['status'] == 'completed'
        assert agent.contexts[-1]['recentPlans']
        assert agent.contexts[-1]['toolResults'][0]['results'][0]['detail'] == 'Close-up captured'
        trace = client.get(f'/api/v1/sessions/{sid}/trace',headers=headers).json()['events']
        assert {'perception','decision','tool_results'} <= {x['type'] for x in trace}
        tool['results'][0]['actionId'] = 'shell-command'
        assert client.post(f'/api/v1/sessions/{sid}/tool-results',headers=headers,json=tool).status_code == 422


def test_scene_supersedes_active_and_queued(tmp_path):
    with TestClient(create_app(settings(tmp_path),Agent(delay=.15))) as client:
        headers,sid = connect(client)
        first = submit(client,headers,sid,1).json()['id']
        time.sleep(.03)
        second = submit(client,headers,sid,2).json()['id']
        third = submit(client,headers,sid,3).json()['id']
        assert wait(client,headers,sid,third)['status'] == 'completed'
        assert wait(client,headers,sid,first)['status'] == 'superseded'
        assert wait(client,headers,sid,second)['status'] == 'superseded'
        assert submit(client,headers,sid,2).status_code == 409


def test_off_and_provider_failures(tmp_path):
    with TestClient(create_app(settings(tmp_path/'off',provider='off'))) as client:
        headers,sid = connect(client)
        assert submit(client,headers,sid).status_code == 503
    with TestClient(create_app(settings(tmp_path/'failure'),Agent(error=ProviderFailure('Account rate limit reached')))) as client:
        headers,sid = connect(client)
        tid = submit(client,headers,sid).json()['id']
        assert wait(client,headers,sid,tid)['error'] == 'Account rate limit reached'


def test_limits_and_invalid_image(tmp_path):
    with TestClient(create_app(settings(tmp_path,session_limit=1),Agent())) as client:
        headers,sid = connect(client)
        assert submit(client,headers,sid,image='data:image/jpeg;base64,invalid').status_code == 422
        tid = submit(client,headers,sid).json()['id']
        wait(client,headers,sid,tid)
        assert submit(client,headers,sid).status_code == 429
    with TestClient(create_app(settings(tmp_path/'daily',daily_limit=1),Agent())) as client:
        headers,sid = connect(client)
        submit(client,headers,sid)
        other,sid2 = connect(client)
        assert submit(client,other,sid2).status_code == 429


def test_registry_and_locked_detector_reject(tmp_path):
    bad = plan()
    bad.actions[0].type = 'select_detector'
    with TestClient(create_app(settings(tmp_path),Agent(result=bad))) as client:
        headers,sid = connect(client)
        tid = submit(client,headers,sid,context={'locked':True,'mode':'surface','activeTaskId':'visual-surfaces'}).json()['id']
        result = wait(client,headers,sid,tid)
        assert result['status'] == 'failed'
        assert 'locked' in result['error']


def test_restart_cancels_pending_and_preserves_tokens(tmp_path):
    config = settings(tmp_path)
    with TestClient(create_app(config,Agent(delay=2))) as client:
        headers,sid = connect(client)
        tid = submit(client,headers,sid).json()['id']
    with TestClient(create_app(config,Agent())) as client:
        result = client.get(f'/api/v1/sessions/{sid}/turns/{tid}',headers=headers)
        assert result.status_code == 200
        assert result.json()['status'] == 'cancelled'


def test_bedrock_converse_contract(tmp_path):
    class Client:
        def converse(self, **kwargs):
            self.request = kwargs
            return {'output':{'message':{'content':[{'toolUse':{'name':'submit_inspection_plan','input':plan().model_dump()}}]}},'usage':{'inputTokens':100,'outputTokens':40}}
    import asyncio
    client = Client()
    path = tmp_path/'image.jpg'
    path.write_bytes(base64.b64decode(image().split(',')[1]))
    provider = BedrockAgent(settings(tmp_path,provider='aws'),client)
    result,metrics = asyncio.run(provider.plan({'language':'en'},path))
    assert result.scene.environment == 'indoor'
    assert metrics['engine'] == 'bedrock-converse'
    assert client.request['modelId'] == 'us.amazon.nova-2-lite-v1:0'
    assert client.request['messages'][0]['content'][0]['image']['source']['bytes'] == path.read_bytes()
    assert client.request['toolConfig']['toolChoice'] == {'any':{}}


def test_pair_throttle_and_snapshot_retention(tmp_path):
    import os
    with TestClient(create_app(settings(tmp_path),Agent())) as client:
        for _ in range(5):
            assert client.post('/api/v1/pair',json={'code':'WRONG000','deviceName':'x'}).status_code == 401
        assert client.post('/api/v1/pair',json={'code':'WRONG000','deviceName':'x'}).status_code == 429
        old = tmp_path/'snapshots'/'old.jpg'
        old.write_bytes(b'expired')
        os.utime(old,(time.time()-86401,time.time()-86401))
        client.app.state.service.cleanup()
        assert not old.exists()


def test_image_header_bomb_rejected_before_decode(tmp_path,monkeypatch):
    import struct
    bad = b'\x89PNG\r\n\x1a\n' + b'\0\0\0\rIHDR' + struct.pack('>II',100000,100000)
    with TestClient(create_app(settings(tmp_path),Agent())) as client:
        headers,sid = connect(client)
        monkeypatch.setattr(cv2,'imdecode',lambda *args: pytest.fail('Oversized header reached decoder'))
        assert submit(client,headers,sid,image='data:image/png;base64,'+base64.b64encode(bad).decode()).status_code == 422


def test_codex_subprocess_isolation_and_schema(tmp_path,monkeypatch):
    import asyncio
    captured = {}
    work = tmp_path/'work'
    work.mkdir()
    source = tmp_path/'image.jpg'
    source.write_bytes(base64.b64decode(image().split(',')[1]))
    class Process:
        returncode = 0
        async def communicate(self,data):
            captured['stdin'] = data
            Path(captured['args'][captured['args'].index('--output-last-message')+1]).write_text(plan().model_dump_json())
            return b'{"type":"turn.completed","usage":{"input_tokens":10}}\n',b''
    async def spawn(*args,**kwargs):
        captured.update(args=args,kwargs=kwargs)
        return Process()
    monkeypatch.setattr(asyncio,'create_subprocess_exec',spawn)
    monkeypatch.setattr('shutil.which',lambda _: 'codex.exe')
    monkeypatch.setenv('AWS_SECRET_ACCESS_KEY','do-not-forward')
    result,metrics = asyncio.run(CodexAgent(settings(tmp_path)).plan({'inspection':{'language':'en'}},source))
    assert result.needsHumanReview
    assert metrics['usage']['input_tokens'] == 10
    assert '--ignore-user-config' in captured['args'] and '--ignore-rules' in captured['args']
    assert '--ephemeral' in captured['args'] and '--no-daemon' in captured['args']
    assert 'AWS_SECRET_ACCESS_KEY' not in captured['kwargs']['env']
    assert 'shell' not in captured['kwargs']
    assert not list(work.iterdir())


def test_portable_backend_imports_from_arbitrarily_named_kit(tmp_path):
    import shutil
    import subprocess
    import sys
    app_folder = tmp_path/'my-renamed-field-kit'
    backend_source = Path(__file__).resolve().parents[1]
    shutil.copytree(backend_source,app_folder/'backend',ignore=shutil.ignore_patterns('data','.venv','__pycache__','.pytest_cache','deploy','tests'))
    (app_folder/'assets').mkdir()
    shutil.copyfile(backend_source.parent/'assets'/'catalogue.json',app_folder/'assets'/'catalogue.json')
    result = subprocess.run([sys.executable,'-c','from backend.app import create_app; from backend.config import Settings; s=Settings(); print(s.catalogue.exists()); print(create_app(s).title)'],
                cwd=app_folder,capture_output=True,text=True,check=True)
    assert 'True' in result.stdout
    assert 'Burhan Smart Inspection' in result.stdout


def test_non_detector_mode_canonicalized_without_changing_task(tmp_path):
    observed = plan()
    observed.actions[0].taskId = 'permit-design'
    # A camera-view instruction for a manual catalogue task does not run crack inference.
    with TestClient(create_app(settings(tmp_path),Agent(result=observed))) as client:
        headers,sid = connect(client)
        tid = submit(client,headers,sid).json()['id']
        result = wait(client,headers,sid,tid)
        assert result['status'] == 'completed'
        assert result['result']['actions'][0]['mode'] is None
        assert result['result']['actions'][0]['taskId'] == 'permit-design'
        events = client.get(f'/api/v1/sessions/{sid}/trace',headers=headers).json()['events']
        assert any(x['type']=='plan_normalized' for x in events)


@pytest.mark.parametrize('unknown,expected',[(True,'unknown_action_task'),(False,'detector_task_mode_mismatch')])
def test_rejected_plan_diagnostics_preserve_strict_capabilities(tmp_path,unknown,expected):
    observed = plan()
    observed.actions[0].taskId = 'invented-paving-detector' if unknown else 'permit-design'
    observed.actions[0].type = 'select_detector'
    with TestClient(create_app(settings(tmp_path),Agent(result=observed))) as client:
        headers,sid = connect(client)
        tid = submit(client,headers,sid).json()['id']
        assert wait(client,headers,sid,tid)['status'] == 'failed'
        events = client.get(f'/api/v1/sessions/{sid}/trace',headers=headers).json()['events']
        rejection = next(x['data'] for x in events if x['type']=='rejected_plan')
        assert rejection['code'] == expected
        assert rejection['plan']['actions'][0]['taskId'] == observed.actions[0].taskId


def test_generation_schema_uses_exact_catalogue_ids():
    schema = plan_schema({'capabilityCatalogue':[{'id':'visual-cracks'},{'id':'permit-design'}]})
    assert schema['$defs']['Suggestion']['properties']['taskId']['enum'] == ['visual-cracks','permit-design']
    branches = schema['$defs']['Action']['properties']['taskId']['anyOf']
    assert next(x for x in branches if x['type']=='string')['enum'] == ['visual-cracks','permit-design']
    guidance = schema['properties']['guidanceTaskId']['anyOf']
    assert next(x for x in guidance if x['type']=='string')['enum'] == ['visual-cracks','permit-design']
    assert set(schema['required']) == set(schema['properties'])


def decision(kind='selected', task='visual-cracks', event_id='phone-1', **kwargs):
    return {'eventId':event_id,'type':kind,'taskId':task,'sceneVersion':1,**kwargs}


def test_check_events_authenticated_idempotent_and_late(tmp_path):
    agent = Agent()
    with TestClient(create_app(settings(tmp_path),agent)) as client:
        headers,sid = connect(client)
        other,other_sid = connect(client)
        url = f'/api/v1/sessions/{sid}/events'
        assert client.post(url,json=decision()).status_code == 401
        assert client.post(url,headers=other,json=decision()).status_code == 404
        assert client.post(url,headers=headers,json=decision(task='invented-task')).status_code == 422
        assert client.post(url,headers=headers,json=decision(event_id='')).status_code == 422
        assert client.post(url,headers=headers,json=decision(kind='executed')).status_code == 422
        tid = submit(client,headers,sid,version=3).json()['id']
        wait(client,headers,sid,tid)
        body = decision(sourceTurnId=tid)
        assert client.post(url,headers=headers,json=body).json()['duplicate'] is False
        assert client.post(url,headers=headers,json=body).json()['duplicate'] is True
        assert client.post(url,headers=headers,json={**body,'type':'stopped'}).status_code == 409
        assert client.post(url,headers=headers,json=decision(event_id='foreign-turn',sourceTurnId='unknown')).status_code == 404
        another = submit(client,headers,sid,version=4).json()['id']
        assert wait(client,headers,sid,another)['status'] == 'completed'
        assert agent.contexts[-1]['checkEvents'] == [body]
        client.delete(f'/api/v1/sessions/{sid}',headers=headers)
        assert client.post(url,headers=headers,json=decision('stopped',event_id='phone-2')).status_code == 200
        rows = client.get(f'/api/v1/sessions/{sid}/trace',headers=headers).json()['events']
        assert len([e for e in rows if e['type']=='check_event']) == 2
    # Deduplication survives a backend restart without re-pairing.
    with TestClient(create_app(settings(tmp_path),Agent())) as client:
        assert client.post(url,headers=headers,json=body).json()['duplicate'] is True


def test_confirm_selection_context_and_scene_scoped_suppression(tmp_path):
    with TestClient(create_app(settings(tmp_path),Agent())) as client:
        headers,sid = connect(client)
        assert submit(client,headers,sid,context={'selectionMode':'confirm','mode':'crack'}).status_code == 422
        assert submit(client,headers,sid,context={'selectionMode':'confirm','mode':'crack','activeTaskId':'visual-cracks'}).status_code == 422
        assert submit(client,headers,sid,context={'selectionMode':'confirm','mode':'manual','dismissedTaskIds':['invented']}).status_code == 422
        ctx = {'selectionMode':'confirm','selectionConfirmed':True,'mode':'surface','activeTaskId':'visual-surfaces','dismissedTaskIds':['visual-cracks']}
        tid = submit(client,headers,sid,context=ctx).json()['id']
        row = wait(client,headers,sid,tid)
        assert row['status'] == 'completed'
        assert row['result']['guidanceTaskId'] == 'visual-surfaces'
        assert row['result']['suggestions'] == []
        # A new stable scene clears the client-scoped list, so the proposal can return.
        ctx['dismissedTaskIds'] = []
        tid = submit(client,headers,sid,version=2,context=ctx).json()['id']
        assert wait(client,headers,sid,tid)['result']['suggestions'][0]['taskId'] == 'visual-cracks'


def test_mismatched_guidance_cannot_overwrite_selected_check(tmp_path):
    result = plan()
    result.guidanceTaskId = 'visual-cracks'
    with TestClient(create_app(settings(tmp_path),Agent(result=result))) as client:
        headers,sid = connect(client)
        tid = submit(client,headers,sid,context={'selectionMode':'confirm','selectionConfirmed':True,'mode':'surface','activeTaskId':'visual-surfaces'}).json()['id']
        assert wait(client,headers,sid,tid)['status'] == 'failed'
        rows = client.get(f'/api/v1/sessions/{sid}/trace',headers=headers).json()['events']
        assert next(e for e in rows if e['type']=='rejected_plan')['data']['code'] == 'guidance_task_mismatch'


def test_proposal_does_not_report_detector_execution_before_selection(tmp_path):
    result = plan()
    result.actions[0].type = 'select_detector'
    with TestClient(create_app(settings(tmp_path),Agent(result=result))) as client:
        headers,sid = connect(client)
        tid = submit(client,headers,sid,context={'selectionMode':'confirm','mode':'manual'}).json()['id']
        assert wait(client,headers,sid,tid)['status'] == 'completed'
        tool = {'turnId':tid,'sceneVersion':1,'results':[{'actionId':'view-1','status':'completed'}]}
        url = f'/api/v1/sessions/{sid}/tool-results'
        assert client.post(url,headers=headers,json=tool).status_code == 409
        tool['results'][0]['status'] = 'accepted'
        assert client.post(url,headers=headers,json=tool).status_code == 200
        event_url = f'/api/v1/sessions/{sid}/events'
        client.post(event_url,headers=headers,json=decision())
        tool['results'][0]['status'] = 'completed'
        assert client.post(url,headers=headers,json=tool).status_code == 200
        client.post(event_url,headers=headers,json=decision('stopped',event_id='phone-2'))
        assert client.post(url,headers=headers,json=tool).status_code == 409
        result.actions[0].taskId = None
        tid = submit(client,headers,sid,version=2,context={'selectionMode':'confirm','mode':'manual'}).json()['id']
        assert wait(client,headers,sid,tid)['status'] == 'failed'
        events = client.get(f'/api/v1/sessions/{sid}/trace',headers=headers).json()['events']
        assert next(e for e in reversed(events) if e['type']=='rejected_plan')['data']['code'] == 'missing_detector_task'
