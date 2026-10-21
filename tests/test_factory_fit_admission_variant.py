"""Temporary fake NN bytes only: prove additive identity and gate-only policy."""
from copy import deepcopy
import hashlib,json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from create_factory_fit_admission_variant import create_variant,GATE

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value),encoding='utf-8')

@pytest.fixture
def setup(tmp_path):
    model=tmp_path/'frozen-D.onnx';model.write_bytes(b'TEMPORARY NONEXECUTABLE FIXTURE NN BYTES')
    meta=model.with_suffix('.onnx.json')
    policy={'high':.995,'area':16,'growth':.7}
    metadata={'model_sha256':sha(model),'supported_height_width_ratio':[2.3,2.7],
              'pixel_threshold':policy['high'],'minimum_component_pixels':policy['area'],'hysteresis_growth_threshold':policy['growth'],
              'output_logits':True,'manual_review_only':True,'automatic_acceptance':False,
              'input_shape':[1,3,1280,512],'output_shape':[1,1,640,256],
              'training_protocol':{'name':'spatial_round3f_orientation_multiscale','fine_steps':900,'save_at':[300,600,900],
                                   'sdd2_inner_split_sha256':'f'*64,'split_items':{'train':['kos01'],'calibration':['kos02'],'test':['kos03']}}}
    write(meta,metadata)
    freeze=tmp_path/'parent-freeze.json'
    write(freeze,{'attempt':4,'variant':'orientation-multiscaleD','winner':{'step':300,'model_path':str(model),'model_sha256':sha(model),'metadata_sha256':sha(meta),'point':policy}})
    images=[]
    for n in range(239):images.append({'dataset':'SDD1_TRAIN','image':f'datasets/factory/KolektorSDD/kos01/{n}.jpg','sha256':'1'*64,'width':500,'height':1235,'height_width_ratio':2.47})
    for n in range(1865):images.append({'dataset':'SDD2_inner_FIT','image':f'datasets/factory/KolektorSDD2/train/{n}.png','sha256':'2'*64,
                'frozen_fit_image_sha256':'2'*64,'source_hash_matches':True,'width':184,'height':627,'height_width_ratio':627/184})
    proposal=tmp_path/'fit-proposal.json'
    write(proposal,{'passed_source_identity':True,'errors':[],'preserved_height_width_ratio_gate':[2.3,2.7],
                    'sdd2_inner_split_sha256':'f'*64,'images':images,'proposal_only':{'candidate_ratio_envelope_with_2pct_tolerance':GATE}})
    return {'freeze':freeze,'proposal':proposal,'model':model,'meta':meta,'out':tmp_path/'additive.onnx','outfreeze':tmp_path/'additive-freeze.json'}

def run(s,**changes):
    values={'freeze_path':s['freeze'],'expected_freeze_sha':sha(s['freeze']),'proposal_path':s['proposal'],
            'expected_proposal_sha':sha(s['proposal']),'output_model':s['out'],'output_freeze':s['outfreeze']}
    values.update(changes);return create_variant(**values)

def test_identical_nn_bytes_and_only_gate_plus_provenance_metadata_change(setup):
    s=setup;originals={p:p.read_bytes() for p in [s['freeze'],s['proposal'],s['model'],s['meta']]}
    result=run(s);child=json.loads(s['out'].with_suffix('.onnx.json').read_text());parent=json.loads(s['meta'].read_text())
    assert s['out'].read_bytes()==s['model'].read_bytes()
    assert result['model_sha256']==sha(s['model']) and child['supported_height_width_ratio']==GATE
    for key in parent:
        if key!='supported_height_width_ratio':assert child[key]==parent[key]
    assert set(child)-set(parent)=={'admission_variant'}
    frozen=json.loads(s['outfreeze'].read_text());old=json.loads(s['freeze'].read_text())
    assert frozen['winner']['point']==old['winner']['point']
    assert frozen['winner']['metadata_sha256']==sha(s['out'].with_suffix('.onnx.json'))
    assert frozen['parent_freeze_sha256']==sha(s['freeze'])
    for path,raw in originals.items():assert path.read_bytes()==raw

def test_existing_outputs_cannot_be_overwritten(setup):
    run(setup);before=setup['outfreeze'].read_bytes()
    with pytest.raises(FileExistsError):run(setup)
    assert setup['outfreeze'].read_bytes()==before

def test_changed_parent_or_proposal_identity_fails_before_any_output(setup):
    with pytest.raises(ValueError,match='Parent freeze differs'):run(setup,expected_freeze_sha='0'*64)
    with pytest.raises(ValueError,match='FIT proposal differs'):run(setup,expected_proposal_sha='0'*64)
    assert not setup['out'].exists()

def test_changed_nn_bytes_or_parent_metadata_rejected(setup):
    setup['model'].write_bytes(b'changed')
    with pytest.raises(ValueError,match='Frozen parent NN differs'):run(setup)
    assert not setup['out'].exists()

def test_metadata_hash_drift_rejected(setup):
    setup['meta'].write_text('{}')
    with pytest.raises(ValueError,match='Frozen parent metadata differs'):run(setup)

def test_even_rehashed_parent_sidecar_cannot_change_selected_seed_area_growth(setup):
    meta=json.loads(setup['meta'].read_text());meta['minimum_component_pixels']=99;write(setup['meta'],meta)
    freeze=json.loads(setup['freeze'].read_text());freeze['winner']['metadata_sha256']=sha(setup['meta']);write(setup['freeze'],freeze)
    with pytest.raises(ValueError,match='seed/area/growth differs'):run(setup)

def test_unfrozen_or_non_D_checkpoint_cannot_be_used(setup):
    freeze=json.loads(setup['freeze'].read_text());freeze['attempt']=3;write(setup['freeze'],freeze)
    with pytest.raises(ValueError,match='Only a frozen attempt4'):run(setup)

def test_actual_future_freeze_must_exist(setup):
    with pytest.raises(FileNotFoundError):run(setup,freeze_path=setup['freeze'].with_name('does-not-exist.json'))

def test_fit_proposal_wrong_source_or_gate_fails_even_with_new_proposal_hash(setup):
    proposal=json.loads(setup['proposal'].read_text());proposal['images'][-1]['image']='datasets/factory/KolektorSDD2/test/999.png';write(setup['proposal'],proposal)
    with pytest.raises(ValueError,match='outside official TRAIN'):run(setup)

def test_proposal_cannot_change_the_predeclared_gate(setup):
    proposal=json.loads(setup['proposal'].read_text());proposal['proposal_only']['candidate_ratio_envelope_with_2pct_tolerance']=[1.,5.];write(setup['proposal'],proposal)
    with pytest.raises(ValueError,match='fixed min/max'):run(setup)

def test_destinations_cannot_alias_original_models_or_sidecars(setup):
    with pytest.raises(ValueError,match='cannot overwrite source'):run(setup,output_model=setup['model'])
    assert not setup['outfreeze'].exists()

def test_outputs_are_deterministic_for_the_same_pinned_inputs(setup):
    run(setup);paths=[setup['out'],setup['out'].with_suffix('.onnx.json'),setup['outfreeze']]
    first={path:path.read_bytes() for path in paths}
    for path in paths:path.unlink()
    run(setup)
    for path in paths:assert path.read_bytes()==first[path]

def test_interrupted_output_write_removes_only_its_partial_new_artifacts(setup,monkeypatch):
    original_open=Path.open
    def fail_last(path,*args,**kwargs):
        if path==setup['outfreeze'] and args and args[0]=='xb':raise OSError('fixture simulates final-output failure')
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',fail_last)
    with pytest.raises(OSError,match='fixture simulates'):run(setup)
    assert not setup['out'].exists() and not setup['out'].with_suffix('.onnx.json').exists()
    assert not setup['outfreeze'].exists() and setup['model'].is_file() and setup['meta'].is_file()
