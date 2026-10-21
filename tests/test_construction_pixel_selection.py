"""Pure CAL/freezer contracts; no model/data loads or benchmark."""
from pathlib import Path
import hashlib
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_construction_pixel_supervision import select_threshold, validate_report_set


def point(threshold, recall, precision, fpr):
    return {'threshold': threshold, 'recall': recall, 'precision': precision, 'negative_image_fpr': fpr,
            'f1': 2 * recall * precision / (recall + precision) if recall + precision else 0}


def test_recall_cannot_override_fixed_precision_and_negative_guards():
    selected, _ = select_threshold([point(.1, 1., .79, 0), point(.15, 1., .9, .2), point(.25, .5, .8, .1)])
    assert selected['threshold'] == .25


def test_ineligible_diagnostic_never_becomes_a_winner():
    selected, diagnostic = select_threshold([point(.1, .9, .2, .5), point(.2, .2, .7, .2)])
    assert selected is None and diagnostic is not None


def test_fixed_threshold_tie_break_prefers_higher_cutoff():
    selected, _ = select_threshold([point(.2, .5, .9, .1), point(.25, .5, .9, .1)])
    assert selected['threshold'] == .25


def reports():
    sources = [{'source': 'fixture', 'image': f'fixture/frame{i}.png', 'sha256': str(i)} for i in range(122)]
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    return [{'step': step, 'ordered_calibration_sources': [dict(r) for r in sources],
             'calibration_source_identity_sha256': digest, 'serving_parity_all122_masks_boxes_scores': True,
             'eligible_selected': point(.25, .5, .8, .1)} for step in [100, 200, 300]]


def test_all_three_sources_and_actual_serving_parity_are_mandatory():
    rows = reports()
    validate_report_set(rows)
    with pytest.raises(ValueError, match='All three'):
        validate_report_set(rows[:2])
    rows[-1]['serving_parity_all122_masks_boxes_scores'] = False
    with pytest.raises(ValueError, match='serving parity'):
        validate_report_set(rows)


def test_changed_ordered_source_payload_rejected_even_with_copied_digest():
    rows = reports()
    rows[-1]['ordered_calibration_sources'][0]['image'] = 'unexpected.png'
    with pytest.raises(ValueError, match='digest differ'):
        validate_report_set(rows)


def test_invalid_eligible_point_is_rejected_by_freezer_guard():
    rows = reports()
    rows[-1]['eligible_selected']['precision'] = .799
    with pytest.raises(ValueError, match='strict guards'):
        validate_report_set(rows)
