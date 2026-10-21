"""Capability gating and provider parity, without live credentials or model claims."""
import asyncio
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from burhan.backend.app import PlanRejection, Service, create_app
from burhan.backend.models import Context, NewSession, Plan
from burhan.backend.providers import BedrockAgent, CodexAgent, SYSTEM, plan_schema
from burhan.backend.tests.test_backend import Agent, connect, decision, plan, settings, submit, wait

PAINT = 'research:walls:peeling'
STAIN = 'research:walls:stain'
FASTENER = 'research:brackets:fastener'
TILE = 'research:tile:coverage'
MODELS = ['research-phele', 'research-fasteners', 'research-stain']


def expanded_settings(tmp_path, **kwargs):
    tmp_path.mkdir(parents=True, exist_ok=True)
    tasks = []
    for task_id, model in [(PAINT, MODELS[0]), (FASTENER, MODELS[1]), (STAIN, MODELS[2]), (TILE, None)]:
        tasks.append({'id':task_id, 'title':'Candidate appearance', 'titleAr':'ملاحظة تجريبية',
            'stage':'finishes', 'mode':'research' if model else 'manual', 'automatic':bool(model),
            'researchOnly':True, 'family':task_id.split(':')[1], 'categoryId':task_id.split(':')[2],
            'modelId':model, 'scope':'Visible appearance only; human review required.',
            'requiredEvidence':['Photograph', 'Human confirmation']})
    catalogue = tmp_path/'research.json'
    catalogue.write_text(json.dumps({'tasks':tasks}), encoding='utf-8')
    return settings(tmp_path, research_catalogue=catalogue, **kwargs)


def capabilities(enabled=True, models=MODELS):
    return {'expandedInspectionModels':enabled, 'installedModelIds':models}


def research_plan(task=PAINT):
    data = plan().model_dump()
    data['suggestions'][0].update(taskId=task, mode='research', title='Peeling candidate', automatic=True)
    data['actions'][0].update(taskId=task, mode='research', type='select_detector')
    return Plan.model_validate(data)


def test_capability_defaults_are_backward_compatible():
    for value in (Context(), NewSession(inspectionId='legacy')):
        assert value.expandedInspectionModels is False
        assert value.installedModelIds == []
    for ids in (['../../arbitrary.onnx'], ['https://example.com/model'], ['']):
        with pytest.raises(ValueError):
            Context(installedModelIds=ids)


@pytest.mark.parametrize('context,expected', [
    ({}, set()),
    (capabilities(False), set()),
    (capabilities(True, []), {TILE}),
    (capabilities(True, ['made-up-detector']), {TILE}),
    (capabilities(True, ['research-phele']), {PAINT, TILE}),
    (capabilities(), {PAINT, STAIN, FASTENER, TILE}),
])
def test_both_provider_catalogues_only_expose_installed_enabled_tasks(tmp_path, context, expected):
    agent = Agent()
    with TestClient(create_app(expanded_settings(tmp_path), agent)) as client:
        headers, sid = connect(client)
        tid = submit(client, headers, sid, context={'mode':'manual', **context}).json()['id']
        assert wait(client, headers, sid, tid)['status'] == 'completed'
        catalogue = agent.contexts[-1]['capabilityCatalogue']
        assert {t['id'] for t in catalogue if t['researchOnly']} == expected
        assert {'visual-cracks', 'visual-surfaces', 'visual-ppe'} <= {t['id'] for t in catalogue}
        for task in catalogue:
            if task['mode'] == 'research':
                assert task['modelId'] in context['installedModelIds']
                assert task['scope'] and task['requiredEvidence']


@pytest.mark.parametrize('context', [{}, capabilities(False), capabilities(True, []), capabilities(True, ['unknown'])])
def test_disabled_research_proposals_rejected_not_silently_repaired(tmp_path, context):
    with TestClient(create_app(expanded_settings(tmp_path), Agent(result=research_plan()))) as client:
        headers, sid = connect(client)
        tid = submit(client, headers, sid, context={'mode':'manual', **context}).json()['id']
        result = wait(client, headers, sid, tid)
        assert result['status'] == 'failed'
        trace = client.get(f'/api/v1/sessions/{sid}/trace', headers=headers).json()['events']
        assert next(e for e in trace if e['type']=='rejected_plan')['data']['code'] == 'suggestion_capability_mismatch'


def test_research_requires_registered_task_and_user_selection(tmp_path):
    with TestClient(create_app(expanded_settings(tmp_path), Agent(result=research_plan()))) as client:
        headers, sid = connect(client)
        for ctx in (
            {'mode':'research', **capabilities()},
            {'mode':'research', 'activeTaskId':PAINT, **capabilities(False)},
            {'mode':'research', 'activeTaskId':PAINT, **capabilities(True, [])},
            {'mode':'research', 'activeTaskId':'research:unknown', **capabilities()},
            {'mode':'research', 'activeTaskId':TILE, **capabilities()},
            {'mode':'research', 'activeTaskId':PAINT, 'selectionMode':'confirm', **capabilities()},
        ):
            assert submit(client, headers, sid, context=ctx).status_code == 422
        tid = submit(client, headers, sid, context={'mode':'manual', 'selectionMode':'confirm', **capabilities()}).json()['id']
        result = wait(client, headers, sid, tid)
        assert result['status'] == 'completed'
        assert result['result']['guidanceTaskId'] is None
        tool = {'turnId':tid, 'sceneVersion':1, 'results':[{'actionId':'view-1','status':'completed'}]}
        url = f'/api/v1/sessions/{sid}/tool-results'
        assert client.post(url, headers=headers, json=tool).status_code == 409
        event = decision(task=PAINT)
        assert client.post(f'/api/v1/sessions/{sid}/events', headers=headers, json=event).status_code == 200
        assert client.post(url, headers=headers, json=tool).status_code == 200
        # The research mode cannot bypass its registered model binding.
        selected = {'mode':'research', 'activeTaskId':PAINT, 'selectionMode':'confirm',
                    'selectionConfirmed':True, **capabilities()}
        tid = submit(client, headers, sid, version=2, context=selected).json()['id']
        result = wait(client, headers, sid, tid)
        assert result['status'] == 'completed' and result['result']['guidanceTaskId'] == PAINT


def test_switch_off_supersedes_pending_same_scene_and_rejects_new_events(tmp_path):
    agent = Agent(delay=.3)
    with TestClient(create_app(expanded_settings(tmp_path), agent)) as client:
        headers, sid = connect(client)
        first = submit(client, headers, sid, context={'mode':'manual', **capabilities()}).json()['id']
        time.sleep(.03)
        second = submit(client, headers, sid, context={'mode':'manual', **capabilities(False)}).json()['id']
        assert wait(client, headers, sid, first)['status'] == 'superseded'
        assert wait(client, headers, sid, second)['status'] == 'completed'
        for kind in ('selected', 'proposed', 'dismissed', 'completed'):
            assert client.post(f'/api/v1/sessions/{sid}/events', headers=headers,
                json=decision(kind, task=PAINT, event_id=kind)).status_code == 422
        assert all(not task['researchOnly'] for task in agent.contexts[-1]['capabilityCatalogue'])


def test_old_tool_results_cannot_execute_after_switch_off(tmp_path):
    agent = Agent(result=research_plan())
    with TestClient(create_app(expanded_settings(tmp_path), agent)) as client:
        headers, sid = connect(client)
        tid = submit(client, headers, sid, context={'mode':'manual', **capabilities()}).json()['id']
        assert wait(client, headers, sid, tid)['status'] == 'completed'
        agent.result = plan()
        off = submit(client, headers, sid, context={'mode':'manual'}).json()['id']
        assert wait(client, headers, sid, off)['status'] == 'completed'
        response = client.post(f'/api/v1/sessions/{sid}/tool-results', headers=headers,
            json={'turnId':tid, 'sceneVersion':1, 'results':[{'actionId':'view-1', 'status':'completed'}]})
        assert response.status_code == 409


def test_session_capabilities_allow_initial_events_and_omitted_turn_disables(tmp_path):
    agent = Agent()
    with TestClient(create_app(expanded_settings(tmp_path), agent)) as client:
        headers, _ = connect(client)
        base = {'inspectionId':'expanded', 'mode':'research', 'taskId':PAINT}
        assert client.post('/api/v1/sessions', headers=headers, json=base).status_code == 422
        response = client.post('/api/v1/sessions', headers=headers, json={**base, **capabilities()})
        assert response.status_code == 201
        sid = response.json()['id']
        url = f'/api/v1/sessions/{sid}/events'
        assert client.post(url, headers=headers, json=decision(task=PAINT)).status_code == 200
        tid = submit(client, headers, sid, context={'mode':'manual'}).json()['id']
        assert wait(client, headers, sid, tid)['status'] == 'completed'
        assert client.post(url, headers=headers, json=decision(task=PAINT, event_id='new')).status_code == 422
        assert client.post(url, headers=headers, json=decision(task=PAINT)).json()['duplicate'] is True


def test_registry_rejects_unknown_model_binding(tmp_path):
    config = expanded_settings(tmp_path)
    data = json.loads(config.research_catalogue.read_text())
    data['tasks'][0]['modelId'] = 'custom-model'
    config.research_catalogue.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='registered bundled model'):
        Service(config, Agent())


def test_shipped_catalogue_has_thirty_checks_with_five_registered_bindings(tmp_path):
    service = Service(settings(tmp_path), Agent())
    expanded = [t for t in service.tasks if t.get('researchOnly')]
    assert len(expanded) == 30
    assert sum(t['automatic'] for t in expanded) == 5
    assert {t['modelId'] for t in expanded if t['automatic']} == set(MODELS)
    assert {t['family'] for t in expanded} == {'walls_finishes', 'brackets_railings', 'tile_installation'}
    service.db.close()


def test_codex_and_aws_share_research_schema_and_normalized_contract(tmp_path, monkeypatch):
    config = expanded_settings(tmp_path)
    service = Service(config, Agent())
    context = {'context':capabilities(), 'capabilityCatalogue':list(service.allowed_registry(capabilities()).values())}
    expected = research_plan()
    schema = plan_schema(context)
    assert PAINT in schema['$defs']['Suggestion']['properties']['taskId']['enum']
    assert 'research' in schema['$defs']['Suggestion']['properties']['mode']['enum']
    assert 'registered bundled experimental detector' in SYSTEM
    source = tmp_path/'image.jpg'
    source.write_bytes(b'fixture-image')
    captured = {}
    class Client:
        def converse(self, **kwargs):
            captured['aws'] = kwargs
            return {'output':{'message':{'content':[{'toolUse':{'name':'submit_inspection_plan','input':expected.model_dump()}}]}}}
    class Process:
        returncode = 0
        async def communicate(self, data):
            captured['prompt'] = data.decode()
            args = captured['args']
            Path(args[args.index('--output-last-message')+1]).write_text(expected.model_dump_json())
            return b'', b''
    async def spawn(*args, **kwargs):
        captured['args'] = args
        captured['codexSchema'] = json.loads(Path(args[args.index('--output-schema')+1]).read_text())
        return Process()
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', spawn)
    monkeypatch.setattr('shutil.which', lambda _: 'codex.exe')
    aws, _ = asyncio.run(BedrockAgent(config, Client()).plan(context, source))
    codex, _ = asyncio.run(CodexAgent(config).plan(context, source))
    assert aws == codex == expected
    assert captured['codexSchema'] == captured['aws']['toolConfig']['tools'][0]['toolSpec']['inputSchema']['json']
    assert captured['aws']['system'][0]['text'] in captured['prompt']
    assert service.validate_plan(aws, capabilities())[0] == service.validate_plan(codex, capabilities())[0]
    with pytest.raises(PlanRejection):
        service.validate_plan(aws, capabilities(False))
    service.db.close()
