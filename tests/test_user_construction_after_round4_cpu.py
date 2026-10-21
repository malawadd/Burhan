"""CPU private-helper preparation fixtures; no images, model or inference."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/evaluate_user_construction_after_round4_cpu.py'
SPEC = importlib.util.spec_from_file_location('private_cpu_preparation', SOURCE)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class CpuPrivatePreparationTests(unittest.TestCase):
    def test_actual_CPU_recipe_rejects_GPU_provider_or_extra_threads(self):
        good = {'providers': ['CPUExecutionProvider'], 'intra_op_num_threads': 1, 'inter_op_num_threads': 1}
        self.assertTrue(module.cpu_config_passes(good))
        for key, value in [('providers', ['CUDAExecutionProvider']), ('intra_op_num_threads', 4), ('inter_op_num_threads', 2)]:
            with self.assertRaises(ValueError):
                module.cpu_config_passes(good | {key: value})

    def test_reuse_is_explicit_process_local_CPU_and_new_outputs(self):
        saved = module.saved
        keys = ['OUT', 'PLAN', 'metadata_plan', 'config_passes', 'SCOPE', 'INFERENCE_SCOPE']
        original = {name: getattr(saved, name) for name in keys}
        with patch.multiple(saved, **original):
            self.assertIs(module.configured_reuse(), saved)
            self.assertEqual(saved.OUT, module.OUT)
            self.assertNotEqual(saved.OUT, original['OUT'])
            self.assertIs(saved.metadata_plan, module.metadata_plan)
            self.assertIs(saved.config_passes, module.cpu_config_passes)
        self.assertEqual(saved.OUT, original['OUT'])
        self.assertIs(saved.metadata_plan, original['metadata_plan'])

    def test_private_comparison_discloses_backend_and_training_confound(self):
        self.assertIn('historical GPU', module.CPU_SCOPE)
        self.assertIn('cannot be attributed solely to training', module.CPU_SCOPE)
        self.assertIn('not ConstructionProfile.process', module.INFERENCE_SCOPE)
        self.assertEqual(module.OPTIONS['engine'], 'onnxruntime')
        self.assertEqual(module.OPTIONS['openvino_device'], 'CPU')
        self.assertEqual(module.OPTIONS['confidence'], .25)
        self.assertEqual(module.OPTIONS['nms_iou'], .45)

    def test_original_GPU_preparation_is_pinned_and_no_execution_on_import(self):
        source = SOURCE.read_text()
        self.assertEqual(module.ORIGINAL_HELPER_SHA, '9859a61412192fe2252fc3bad7b83aefe98d054fee207b2f22588eba249cbca4')
        self.assertIn("'private_source': 878", source)
        self.assertIn("'old_model': 0", source)
        self.assertIn('root-network-release', source)
        self.assertNotIn('torch.load', source)


if __name__ == '__main__':
    unittest.main()
