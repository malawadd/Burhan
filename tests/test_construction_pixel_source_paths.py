"""Regression for actual failed Windows metadata lookup; no NN or images."""
from pathlib import Path
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from construction_pixel_source_paths import source_pin


@pytest.mark.parametrize('stored', ['models/construction/source.pt', r'models\construction\source.pt'])
@pytest.mark.parametrize('requested', ['models/construction/source.pt', r'models\construction\source.pt'])
def test_source_metadata_lookup_survives_both_path_separators(stored, requested):
    assert source_pin({stored: 'a' * 64}, requested) == 'a' * 64


def test_actual_immutable_proposal_tut_pin_resolves_without_loading_checkpoint():
    protocol = json.loads((ROOT / 'output/optimization/construction-pixel-supervision-v1/protocol-before-probe-and-fit.json').read_text())
    assert source_pin(protocol['source_pins'], 'models/construction/thin-contingency/litecrackseg-TUT.pth') == 'c12fb1ce991e479a32e5a26d7648ce9d20a440b9af4fd0411d321aa6a2eed497'


def test_conflicting_aliases_or_missing_sources_fail_instead_of_selecting_one():
    with pytest.raises(ValueError, match='conflicting'):
        source_pin({'models/a.pt': 'a' * 64, r'models\a.pt': 'b' * 64}, 'models/a.pt')
    with pytest.raises(ValueError, match='Missing'):
        source_pin({}, 'models/a.pt')
