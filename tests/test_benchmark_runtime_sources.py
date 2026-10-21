"""Hardware-free replay source contracts; no image decoder or model is loaded."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/benchmark_runtime.py'
spec = importlib.util.spec_from_file_location('benchmark_runtime_sources', SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


@pytest.fixture
def replay_tree(tmp_path):
    test = ['kos02', 'kos04', 'kos07', 'kos12', 'kos26', 'kos32', 'kos37', 'kos38', 'kos40', 'kos45']
    calibration = ['kos03', 'kos05', 'kos08', 'kos09', 'kos13', 'kos16', 'kos23', 'kos27', 'kos29', 'kos48']
    dataset = tmp_path / 'datasets/factory/KolektorSDD'
    for item in test + calibration:
        directory = dataset / item
        directory.mkdir(parents=True)
        for index in range(8):
            (directory / f'Part{index}.jpg').touch()  # Not valid image bytes.
    historical = tmp_path / 'output/evaluation/factory-speed-comparison.json'
    historical.parent.mkdir(parents=True)
    historical.write_text(json.dumps({'split_items': test}))
    model = tmp_path / 'model.onnx'  # Model deliberately does not exist.
    model.with_suffix('.onnx.json').write_text(json.dumps({'split_items': {
        'train': ['kos01'], 'calibration': calibration, 'test': test}}))
    return tmp_path, model, dataset, test, calibration


def test_calibration_selects_only_its_80_views_without_reading_image_payloads(replay_tree, monkeypatch):
    root, model, dataset, test, calibration = replay_tree
    original = Path.read_bytes

    def forbid_images(path):
        assert path.suffix != '.jpg', 'Source selection must not decode/read any image'
        return original(path)

    monkeypatch.setattr(Path, 'read_bytes', forbid_images)
    # CAL selection must not depend on even opening the old test selection report.
    (root / 'output/evaluation/factory-speed-comparison.json').unlink()
    selected = benchmark.select_replay_inputs(root, model, varied=True, partition='calibration')
    assert len(selected['paths']) == 80
    assert {path.parent.name for path in selected['paths']} == set(calibration)
    assert not {path.parent.name for path in selected['paths']} & set(test)
    assert dataset / 'kos04/Part3.jpg' not in selected['paths']
    assert selected['partition'] == 'calibration'
    assert 'not independent validation' in selected['description']
    assert selected['metadata_sha256']


def test_default_test_preserves_exact_historical_sorted_order(replay_tree):
    root, model, dataset, test, _ = replay_tree
    model.with_suffix('.onnx.json').unlink()  # Default must not require selected model metadata.
    old_order = sorted(path for item in test for path in (dataset / item).glob('Part*.jpg'))
    selected = benchmark.select_replay_inputs(root, model, varied=True)
    assert selected['paths'] == old_order
    assert len(selected['paths']) == 80
    assert 'Exposed development comparison' in selected['description']


@pytest.mark.parametrize('decode_input', [False, True])
def test_single_and_decode_input_retain_original_sample(replay_tree, decode_input):
    root, model, dataset, _, _ = replay_tree
    assert benchmark.select_replay_inputs(root, model, decode_input=decode_input)['paths'] == [dataset / 'kos04/Part3.jpg']


def test_nondefault_partition_rejected_before_runtime_import(replay_tree):
    root, model, _, _, _ = replay_tree
    with pytest.raises(ValueError, match='requires --varied'):
        benchmark.select_replay_inputs(root, model, partition='calibration')
    with pytest.raises(SystemExit) as error:
        benchmark.main(['--output', str(root / 'unused.json'), '--partition', 'calibration'])
    assert error.value.code == 2
    with pytest.raises(ValueError, match='separate experiments'):
        benchmark.select_replay_inputs(root, model, varied=True, decode_input=True)


def test_calibration_identity_overlap_and_missing_views_fail_closed(replay_tree):
    root, model, dataset, _, calibration = replay_tree
    sidecar = model.with_suffix('.onnx.json')
    value = json.loads(sidecar.read_text())
    value['split_items']['train'] = [calibration[0]]
    sidecar.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='overlap'):
        benchmark.select_replay_inputs(root, model, varied=True, partition='calibration')
    value['split_items']['train'] = ['kos01']
    sidecar.write_text(json.dumps(value))
    (dataset / calibration[0] / 'Part0.jpg').unlink()
    with pytest.raises(ValueError, match='Expected 80'):
        benchmark.select_replay_inputs(root, model, varied=True, partition='calibration')
