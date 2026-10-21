"""Verify environment-driven switching in isolated API instances, preserving the live tunnel."""
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from burhan.backend.app import create_app
from burhan.backend.config import load_settings

records=[]
token=None
with tempfile.TemporaryDirectory(prefix='burhan-provider-verification-') as data:
    for provider in ('codex','aws','off','codex'):
        with patch.dict(os.environ, {'SMART_INSPECTION_PROVIDER':provider,'BURHAN_DATA_DIR':data,
                'AWS_EC2_METADATA_DISABLED':'true'}):
            with TestClient(create_app(load_settings()),base_url='https://merhaba-api.kiyan.finance') as client:
                health=client.get('/api/v1/health').json()
                assert health['provider']==provider
                if token is None:
                    code=client.app.state.service.pair_code
                    token=client.post('/api/v1/pair',json={'code':code,'deviceName':'Provider verification'}).json()['token']
                headers={'Authorization':'Bearer '+token}
                session=client.post('/api/v1/sessions',headers=headers,json={'inspectionId':'provider-verification'})
                assert session.status_code==201 and session.json()['provider']==provider
                sid=session.json()['id']
                if provider=='off':
                    result=client.post(f'/api/v1/sessions/{sid}/turns',headers=headers,json={'sceneVersion':1,'context':{}})
                    assert result.status_code==503
                assert client.delete(f'/api/v1/sessions/{sid}',headers=headers).status_code==200
                records.append({'provider':provider,'health':health,'persistedPairingAccepted':True})
output=Path('output/smart-smoke/provider-switching.json');output.parent.mkdir(parents=True,exist_ok=True)
record={'testedAt':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'endpoint':'https://merhaba-api.kiyan.finance',
    'scope':'Isolated restarted API instances with real provider selection/readiness; live public tunnel remains on codex.',
    'appRebuiltForSwitch':False,'switches':records,
    'awsLiveInference':'Not run: no AWS credentials/model access available. SDK request contract tested separately.'}
output.write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
