"""Pure candidate-order/eligibility fixtures: no model or image access."""
import importlib.util
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/freeze_factory_round4_lower_alarm.py'
spec = importlib.util.spec_from_file_location('lower_alarm', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(step, recall=1., fp=1, precision=.7, worst=.3):
    return {'step': step, 'point': {'meaningful_iou_0_1_recall': recall, 'normal_fp': fp, 'foreground_pixel_precision': precision,
            'worst_positive_iou': worst, 'coverage_50pct_recall': .9, 'worst_positive_coverage': .4,
            'micro_pixel_iou': .6, 'foreground_pixel_recall': .8}}


class LowerAlarmTests(unittest.TestCase):
    def test_lower_alarm_wins_only_after_equal_meaningful_count(self):
        global_point = row(600, fp=1, worst=.4)
        candidate, ordered = module.choose([row(300, fp=0, worst=.2), global_point, row(900, recall=.9, fp=0)], global_point)
        self.assertEqual(candidate['step'], 300)
        self.assertEqual([value['step'] for value in ordered], [300, 600, 900])

    def test_no_strict_alarm_gain_or_precision_or_recall_means_none(self):
        global_point = row(600)
        self.assertIsNone(module.choose([global_point, row(300, fp=0, precision=.64)], global_point)[0])
        self.assertIsNone(module.choose([row(300, fp=0, recall=.9)], global_point)[0])
        self.assertIsNone(module.choose([row(300, fp=0)], row(600, fp=0))[0])

    def test_existing_rank_then_earlier_step_are_fixed_tiebreaks(self):
        global_point = row(900, fp=1)
        candidate, _ = module.choose([row(600, fp=0), row(300, fp=0)], global_point)
        self.assertEqual(candidate['step'], 300)
        candidate, _ = module.choose([row(300, fp=0, worst=.2), row(600, fp=0, worst=.3)], global_point)
        self.assertEqual(candidate['step'], 600)


if __name__ == '__main__':
    unittest.main()
