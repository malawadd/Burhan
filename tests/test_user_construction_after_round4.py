"""Tiny metadata/cache fixtures; no source photos, videos, models or inference."""
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/evaluate_user_construction_after_round4.py'
SPEC = importlib.util.spec_from_file_location('private_after_helper', SOURCE)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class PrivateAfterPreparationTests(unittest.TestCase):
    def choice(self):
        return {'checkpoints': [dict(step=step, all_minimum_checks_pass=step != 900, rank=[step])
                               for step in [300, 600, 900]],
                'winner': dict(step=600, all_minimum_checks_pass=True, rank=[600])}

    def test_global_rank_and_exact_failed_fallback(self):
        value = self.choice()
        self.assertEqual(m.selected_checkpoint(value)[0]['step'], 600)
        for row in value['checkpoints']:
            row['all_minimum_checks_pass'] = False
        value['winner'] = None
        self.assertEqual(m.selected_checkpoint(value), (value['checkpoints'][2], False))

    def test_duplicate_steps_and_nonwinning_selection_rejected(self):
        value = self.choice()
        value['winner'] = value['checkpoints'][0]
        with self.assertRaises(ValueError):
            m.selected_checkpoint(value)
        value = self.choice()
        value['checkpoints'][2]['step'] = 600
        with self.assertRaises(ValueError):
            m.selected_checkpoint(value)

    def test_actual_config_is_not_inferred_from_requested_options(self):
        good = {'EXECUTION_DEVICES': ['GPU.0'], 'INFERENCE_PRECISION_HINT': '<Type: float32>', 'NUM_STREAMS': 1}
        self.assertTrue(m.config_passes(good))
        for key, wrong in [('EXECUTION_DEVICES', ['CPU']), ('INFERENCE_PRECISION_HINT', 'f16'), ('NUM_STREAMS', 2)]:
            self.assertFalse(m.config_passes(good | {key: wrong}))

    def test_report_source_enables_exact_PTS_UI_contract_without_inventory_mutation(self):
        source = {'file': 'private/original.mp4', 'decoded_frames': 3,
                  'all_frame_times': [{'pts': pts, 'time_base': '1/1000'} for pts in [0, 71, 159]]}
        reported = m.report_source(source, 3)
        self.assertEqual(reported['original_camera_frames'], 3)
        self.assertEqual(reported['original_path'], source['file'])
        self.assertEqual(reported['all_frame_times'], source['all_frame_times'])
        self.assertNotIn('original_camera_frames', source)
        self.assertNotIn('original_path', source)
        with self.assertRaises(ValueError):
            m.report_source(source, 2)

    def test_cached_native_mask_identity_count_and_tail(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'mask.npz'
            packed = np.packbits(np.array([[True, False, True], [False, True, False]]))
            np.savez_compressed(path, author_masks_0=packed)
            ref = {'key': 'author_masks_0', 'shape_hw': [2, 3], 'positive_pixels': 3,
                   'packed_sha256': hashlib.sha256(packed.tobytes()).hexdigest()}
            frame = {'actual_masks': {'path': 'mask.npz', 'sha256': m.sha(path)},
                     'models': {'author_masks': {'predictions': [{'label': 'crack', 'actual_mask': ref}]}}}
            with patch.object(m, 'ROOT', Path(folder)):
                rows = m.cached_author(frame, (2, 3))
                self.assertEqual(int(rows[0]['mask'].sum()), 3)
                self.assertEqual(rows[0]['mask'].dtype, np.bool_)
                ref['positive_pixels'] = 4
                with self.assertRaises(ValueError):
                    m.cached_author(frame, (2, 3))

    def test_inert_import_and_explicit_full_video_scope(self):
        source = SOURCE.read_text()
        self.assertIn("'private_source': 878", source)
        self.assertIn("'initialization': 3", source)
        self.assertIn("'old_model': 0", source)
        self.assertIn("root-network-release", source)
        self.assertNotIn('torch.load', source)


if __name__ == '__main__':
    unittest.main()
