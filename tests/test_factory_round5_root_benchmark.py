"""Pure new benchmark binding/schedule fixtures; no inputs, model or NN."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/benchmark_factory_round5_root.py'
SPEC = importlib.util.spec_from_file_location('round5_root_benchmark', SOURCE)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def candidate(step, worst=.3):
    return {'step': step, 'point': {'meaningful_iou_0_1_recall': 1., 'normal_fp': 0, 'foreground_pixel_precision': .7,
        'worst_positive_iou': worst, 'coverage_50pct_recall': 1., 'worst_positive_coverage': .6,
        'micro_pixel_iou': .6, 'foreground_pixel_recall': .8}}


class Round5RootPreparationTests(unittest.TestCase):
    def test_strict_all10_zero_alarm_precision_guard(self):
        point = candidate(200)['point']
        self.assertTrue(module.qualifies(point))
        self.assertFalse(module.qualifies(None))
        for key, value in [('meaningful_iou_0_1_recall', .9), ('normal_fp', 1), ('foreground_pixel_precision', .649)]:
            self.assertFalse(module.qualifies(point | {key: value}))

    def test_global_rank_has_earlier_step_tie_and_no_fallback(self):
        self.assertEqual(module.global_winner([]), (None, []))
        chosen, ordered = module.global_winner([candidate(600), candidate(400), candidate(200)])
        self.assertEqual(chosen['step'], 200)
        self.assertEqual([r['step'] for r in ordered], [200, 400, 600])
        self.assertEqual(module.global_winner([candidate(200), candidate(400, .4)])[0]['step'], 400)

    def test_reuse_only_rebinds_standalone_common_globals_and_new_folder(self):
        old = {k: getattr(module.common, k) for k in ['OUT', 'PLAN', 'FIXTURE', 'binding', 'PINS']}
        with patch.multiple(module.common, **old):
            result = module.configure_common()
            self.assertIs(result.binding, module.binding)
            self.assertEqual(result.OUT, module.OUT)
            self.assertNotEqual(result.OUT, old['OUT'])
            self.assertIs(result.PINS, module.PINS)
        self.assertEqual(module.common.OUT, old['OUT'])
        self.assertIs(module.common.binding, old['binding'])

    def test_runtime_requires_fixed_old_then_C200_then_winner_order(self):
        self.assertEqual(module.previous_runtime_models('v10h'), [])
        self.assertEqual(module.previous_runtime_models('C200'), ['v10h'])
        self.assertEqual(module.previous_runtime_models('winner'), ['v10h', 'C200'])
        with self.assertRaises(ValueError):
            module.previous_runtime_models('losing_checkpoint')


if __name__ == '__main__':
    unittest.main()
