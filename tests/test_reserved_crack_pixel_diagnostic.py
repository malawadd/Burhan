"""Tiny fixed diagnostic contracts; no source images, models, network or GPU."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_construction_reserved_pixels as diagnostic


def test_ignore_pixels_do_not_create_false_positives_or_true_negatives():
    prediction = np.array([[0, 1], [1, 0]], bool)
    truth = np.array([[0, 1], [0, 1]], bool)
    valid = np.array([[1, 1], [0, 1]], bool)
    metric = diagnostic.pixel_counts(prediction, truth, valid)
    assert (metric['tp'], metric['fp'], metric['fn'], metric['ignored_pixels']) == (1, 0, 1, 1)
    assert metric['precision'] == 1 and metric['recall'] == metric['iou'] == .5
    assert metric['dice'] == pytest.approx(2 / 3)


def test_empty_predictions_are_retained_as_misses_not_dropped_for_macro_precision():
    truth = np.ones((2, 2), bool)
    valid = np.ones_like(truth)
    empty = diagnostic.pixel_counts(np.zeros_like(truth), truth, valid)
    correct = diagnostic.pixel_counts(truth, truth, valid)
    result = diagnostic.aggregate_pixels([{'model': empty}, {'model': correct}], 'model')
    assert result['images'] == 2 and result['empty_predictions_on_positive_gt'] == 1
    assert result['micro']['recall'] == .5 and result['micro']['iou'] == .5
    assert result['macro_all_views']['precision'] is None and result['undefined_precision_views'] == 1


def test_fixed_near_duplicate_rule_is_not_influenced_by_model_scores():
    first = {'dhash': 0, 'normalized': np.array([1.], np.float32)}
    second = {'dhash': 15, 'normalized': np.array([0.], np.float32)}
    assert diagnostic.near_signature(first, second)[0]  # distance4 regardless of correlation
    second = {'dhash': 255, 'normalized': np.array([.94], np.float32)}
    assert not diagnostic.near_signature(first, second)[0]
    second['normalized'][0] = .96
    assert diagnostic.near_signature(first, second)[0]


def test_acquisition_cannot_reach_network_before_postfit_binding(monkeypatch):
    def missing_binding():
        raise ValueError('Required final300 and all3 CAL binding missing')
    def network_forbidden(*args, **kwargs):
        raise AssertionError('Network must remain unreachable without completed postfit guards')
    monkeypatch.setattr(diagnostic, 'checked_binding', missing_binding)
    monkeypatch.setattr(diagnostic.urllib.request, 'urlopen', network_forbidden)
    with pytest.raises(ValueError, match='binding missing'):
        diagnostic.acquire()


def test_later_ambiguous_pair_also_removes_earlier_view_from_clean_group():
    prior = {'clean': True, 'quarantine_reasons': [], 'leakage_flags': []}
    diagnostic.quarantine_prior_ambiguous_view(prior, 'later-source', 3, .9)
    assert not prior['clean']
    assert prior['quarantine_reasons'] == ['leakage_or_ambiguous_view_quarantine_before_NN']
    assert prior['leakage_flags'][0]['reference'] == 'later-source'
    diagnostic.quarantine_prior_ambiguous_view(prior, 'third-source', 4, .7)
    assert len(prior['quarantine_reasons']) == 1 and len(prior['leakage_flags']) == 2
