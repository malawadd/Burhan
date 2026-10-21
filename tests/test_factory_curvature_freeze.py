"""Hardware-free protection of all-checkpoint/source-identity global selection."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('curvature_freeze', Path(__file__).resolve().parents[1] / 'scripts/freeze_factory_round3j.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture():
    sources = [{'image': f'cal/{i}.jpg', 'defective': i < 10, 'image_sha256': f'file{i}', 'mask_sha256': f'mask{i}'} for i in range(80)]
    identity = {'ordered_sources': sources, 'sha256': hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()}
    return {'save_at': [200, 400, 600]}, [{'step': step, 'calibration_source_identity': json.loads(json.dumps(identity))} for step in [200, 400, 600]]


class CurvatureFreeze(unittest.TestCase):
    def test_all_three_valid(self):
        protocol, reports = fixture()
        self.assertEqual(MODULE.validate_report_set(protocol, reports), reports[0]['calibration_source_identity']['sha256'])

    def test_early_winner_denied(self):
        protocol, reports = fixture()
        with self.assertRaises(ValueError):
            MODULE.validate_report_set(protocol, reports[:1])

    def test_duplicate_checkpoint_denied(self):
        protocol, reports = fixture()
        reports[2]['step'] = 400
        with self.assertRaises(ValueError):
            MODULE.validate_report_set(protocol, reports)

    def test_changed_source_even_with_valid_new_hash_denied(self):
        protocol, reports = fixture()
        identity = reports[2]['calibration_source_identity']
        identity['ordered_sources'][0]['image_sha256'] = 'changed'
        identity['sha256'] = hashlib.sha256(json.dumps(identity['ordered_sources'], sort_keys=True).encode()).hexdigest()
        with self.assertRaises(ValueError):
            MODULE.validate_report_set(protocol, reports)

    def test_duplicate_cal_image_denied(self):
        protocol, reports = fixture()
        for report in reports:
            identity = report['calibration_source_identity']
            identity['ordered_sources'][1]['image'] = identity['ordered_sources'][0]['image']
            identity['sha256'] = hashlib.sha256(json.dumps(identity['ordered_sources'], sort_keys=True).encode()).hexdigest()
        with self.assertRaises(ValueError):
            MODULE.validate_report_set(protocol, reports)

    def test_tampered_source_digest_denied(self):
        protocol, reports = fixture()
        reports[0]['calibration_source_identity']['sha256'] = 'stale'
        with self.assertRaises(ValueError):
            MODULE.validate_report_set(protocol, reports)

    def test_exact_preexisting_calibration_pin_accepted(self):
        protocol, reports = fixture()
        expected = reports[0]['calibration_source_identity']['sha256']
        self.assertEqual(MODULE.validate_report_set(protocol, reports, expected), expected)

    def test_identical_but_unapproved_partition_denied(self):
        protocol, reports = fixture()
        with self.assertRaises(ValueError):
            MODULE.validate_report_set(protocol, reports, 'original-frozen-calibration-identity')

    def test_actual_contention_can_differ_without_policy_change(self):
        self.assertTrue(MODULE.same_search_policy({'grid': [1, 2], 'execution_scope': 'fit-running'}, {'grid': [1, 2], 'execution_scope': 'fit-finished'}))

    def test_changed_grid_denied_despite_matching_contention(self):
        self.assertFalse(MODULE.same_search_policy({'grid': [1, 2], 'execution_scope': 'same'}, {'grid': [1, 3], 'execution_scope': 'same'}))


if __name__ == '__main__':
    unittest.main()
