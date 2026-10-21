import pytest
from scripts.factory_training_keys import hardnegative_keys,require_matched_hardnegatives


def test_windows_mining_keys_match_canonical_training_paths():
    records=[{'image':r'kos01\Part0.jpg','locations':[{'x':5,'y':9}]}]
    mapping=hardnegative_keys(records)
    assert mapping['kos01/Part0.jpg']==records[0]['locations']
    require_matched_hardnegatives(['kos01/Part0.jpg'],mapping)


def test_unmatched_or_empty_mining_pool_fails_before_training():
    with pytest.raises(ValueError,match='No training hard-negative paths matched'):
        require_matched_hardnegatives(['kos01/Part0.jpg'],{'other/Part0.jpg':[{}]})
    with pytest.raises(ValueError,match='No training hard-negative paths matched'):
        require_matched_hardnegatives(['kos01/Part0.jpg'],{'kos01/Part0.jpg':[]})
