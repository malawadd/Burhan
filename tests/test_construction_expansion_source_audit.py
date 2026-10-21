"""Source safety checks use artificial archive fixtures, never inspection evidence."""
import importlib.util
import json
from pathlib import Path
import zipfile
import pytest


@pytest.fixture
def audit(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'scripts/audit_construction_expansion_sources.py'
    spec = importlib.util.spec_from_file_location('construction_source_audit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = tmp_path / 'audit'
    return module


def test_archive_inventory_rejects_parent_traversal_without_extraction(audit, tmp_path):
    path = tmp_path / 'source.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('../outside.jpg', b'not an image')
    with pytest.raises(ValueError, match='Unsafe archive'):
        audit.archive('phele', path)
    assert not (tmp_path / 'outside.jpg').exists()


def test_sdnet_known_checksum_is_required(audit, tmp_path):
    path = tmp_path / 'source.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('labels.json', '{}')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        audit.archive('sdnet2025', path)


def test_inventory_is_not_training_authorization(audit, tmp_path):
    path = tmp_path / 'source.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('source/labels.json', '{}')
    audit.archive('phele', path)
    report = json.loads((audit.OUT / 'phele/archive-inventory.json').read_text())
    assert report['training_authorized'] is False
    assert report['publisher_checksum_verified'] is False
    assert report['suffix_counts'] == {'.json': 1}
    assert not (tmp_path / 'source/labels.json').exists()
