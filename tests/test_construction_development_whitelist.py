"""The development exporter must refuse non-whitelisted sources before pixels."""
import importlib.util
import json
from pathlib import Path
import sys
import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('development_video_comparison',SCRIPTS/'compare_construction_development_videos.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def setup_protocol(tmp_path,monkeypatch,heldout=()):
    base=tmp_path/'base';base.mkdir()
    selected=[{'group':g,'sequence':n}for g,n in module.SELECTION if g!='exposed']
    (base/'protocol.json').write_text(json.dumps({'sequences':{'calibration':selected,'heldout':list(heldout)}}))
    monkeypatch.setattr(module,'ROOT',tmp_path);monkeypatch.setattr(module,'BASE',base)
    monkeypatch.setattr(module,'sha',lambda path:'metadata-fixture-sha')


def test_fixed_development_selection_is_four_explicit_sources(tmp_path,monkeypatch):
    setup_protocol(tmp_path,monkeypatch)
    rows=module.selected_sources()
    assert [(r['group'],r['sequence'])for r in rows]==module.SELECTION
    assert [r['role']for r in rows]==['reused_exposed_development']+['calibration_development']*3
    assert not any('heldout' in p for r in rows for p in r['files'])


def test_heldout_collision_fails_before_camera_load(tmp_path,monkeypatch):
    setup_protocol(tmp_path,monkeypatch,[{'group':'dset_6','sequence':'crack_129'}])
    with pytest.raises(RuntimeError,match='heldout'):
        module.selected_sources()


def test_unlisted_train_source_is_rejected(tmp_path,monkeypatch):
    setup_protocol(tmp_path,monkeypatch)
    monkeypatch.setattr(module,'SELECTION',[('dset_3','spalling_54')])
    with pytest.raises(RuntimeError,match='calibration whitelist'):
        module.selected_sources()
