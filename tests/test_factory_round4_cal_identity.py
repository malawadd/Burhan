"""Pure serializer/schema contracts; no model, image decoding or neural calls."""
import ast
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CalIdentityRecoveryTests(unittest.TestCase):
    def test_exact_historical_serializer_is_distinct_from_compact(self):
        tree = ast.parse((ROOT / 'scripts/evaluate_factory_round4_cal_identity_v2.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'protected_cal_identity')
        namespace = {'hashlib': hashlib, 'json': json}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<pure serializer>', 'exec'), namespace)
        rows = [{'image': 'kos01/Part0.jpg', 'defective': True, 'native_mask_shape': [1280, 500]}]
        self.assertEqual(namespace['protected_cal_identity'](rows), hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest())
        self.assertNotEqual(namespace['protected_cal_identity'](rows), hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest())

    def test_wrapper_only_rebinds_identity_and_lineage_writer_and_disallows_numeric_replay(self):
        source = (ROOT / 'scripts/evaluate_factory_round4_cal_identity_v2.py').read_text()
        tree = ast.parse(source)
        rebindings = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == 'original' for t in n.targets)]
        self.assertEqual([n.targets[0].attr for n in rebindings], ['identity', 'write'])
        self.assertNotIn("'numeric-all3'", source)


if __name__ == '__main__':
    unittest.main()
