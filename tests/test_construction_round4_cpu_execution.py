"""Research asset routing contracts, with no model loading or inference."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from edge_runtime.profiles import construction_model as model
from edge_runtime.profiles.construction_round4_cpu_asset import SPEC, VARIANT


OPTIONS = {'mode': 'damage', 'damage_variant': VARIANT, 'engine': 'onnxruntime',
           'openvino_device': 'CPU', 'cpu_threads': 1}


def fake_delegate():
    settings = SimpleNamespace(intra_op_num_threads=1, inter_op_num_threads=1)
    session = Mock()
    session.get_session_options.return_value = settings
    session.get_providers.return_value = ['CPUExecutionProvider']
    session.get_inputs.return_value = [SimpleNamespace(type='tensor(float)')]
    return SimpleNamespace(engine='onnxruntime', device='CPU', threads=1,
        actual_execution_config={'providers': ['CPUExecutionProvider'],
            'intra_op_num_threads': 1, 'inter_op_num_threads': 1},
        session=session, mask_rendering='masks')


class CpuResearchExecutionTests(unittest.TestCase):
    def test_explicit_CPU_route_accepts_only_mocked_delegate(self):
        delegate = fake_delegate()
        with patch('edge_runtime.profiles.construction_cubit.CubitConstructionModel', return_value=delegate) as constructor:
            instance = model.ConstructionModel(OPTIONS)
        constructor.assert_called_once_with(OPTIONS, SPEC)
        self.assertIs(instance.delegate, delegate)
        self.assertIs(instance.spec, SPEC)
        self.assertEqual(instance.mask_rendering, 'masks')
        self.assertTrue(SPEC['diagnostic_only'])

    def test_missing_defaults_or_wrong_requests_fail_before_constructor(self):
        requests = [dict(OPTIONS, engine='openvino'), dict(OPTIONS, openvino_device='GPU'),
                    dict(OPTIONS, cpu_threads=2), dict(OPTIONS, cpu_threads=True),
                    dict(OPTIONS, cpu_threads=1.0)]
        for key in ['engine', 'openvino_device', 'cpu_threads']:
            request = dict(OPTIONS)
            del request[key]
            requests.append(request)
        with patch('edge_runtime.profiles.construction_cubit.CubitConstructionModel') as constructor:
            for request in requests:
                with self.subTest(request=request), self.assertRaises(ValueError):
                    model.ConstructionModel(request)
            constructor.assert_not_called()

    def test_reported_provider_threads_and_backend_are_enforced(self):
        changes = [('engine', 'openvino'), ('device', 'GPU'), ('threads', 2), ('threads', True)]
        for key, value in changes:
            delegate = fake_delegate()
            setattr(delegate, key, value)
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                model._required_execution_guard(SPEC, OPTIONS, delegate)
        for key, value in [('providers', ['CUDAExecutionProvider']),
                           ('providers', ['CPUExecutionProvider', 'CUDAExecutionProvider']),
                           ('intra_op_num_threads', 2), ('inter_op_num_threads', 2),
                           ('intra_op_num_threads', True)]:
            delegate = fake_delegate()
            delegate.actual_execution_config[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                model._required_execution_guard(SPEC, OPTIONS, delegate)

    def test_actual_session_cannot_hide_behind_valid_reported_config(self):
        for attribute, value in [('intra_op_num_threads', 2), ('inter_op_num_threads', 2),
                                 ('intra_op_num_threads', True)]:
            delegate = fake_delegate()
            setattr(delegate.session.get_session_options.return_value, attribute, value)
            with self.subTest(attribute=attribute, value=value), self.assertRaises(ValueError):
                model._required_execution_guard(SPEC, OPTIONS, delegate)
        delegate = fake_delegate()
        delegate.session.get_providers.return_value = ['CUDAExecutionProvider']
        with self.assertRaises(ValueError):
            model._required_execution_guard(SPEC, OPTIONS, delegate)
        delegate = fake_delegate()
        delegate.session.get_inputs.return_value = [SimpleNamespace(type='tensor(float16)')]
        with self.assertRaises(ValueError):
            model._required_execution_guard(SPEC, OPTIONS, delegate)

    def test_existing_unconstrained_variant_keeps_previous_route(self):
        # An empty request retains the historical delegate defaults for old assets.
        delegate = SimpleNamespace(mask_rendering='masks')
        request = {'mode': 'damage', 'damage_variant': 'cubit_author_v2_failed'}
        with patch('edge_runtime.profiles.construction_cubit.CubitConstructionModel', return_value=delegate) as constructor:
            instance = model.ConstructionModel(request)
        constructor.assert_called_once_with(request, model.DAMAGE_VARIANTS['cubit_author_v2_failed'])
        self.assertIs(instance.delegate, delegate)

    def test_invalid_constraint_metadata_cannot_enable_permissive_route(self):
        for requirement in [{}, {'engine': 'onnxruntime', 'openvino_device': 'CPU', 'cpu_threads': True},
                            {'engine': 'onnxruntime', 'openvino_device': 'CPU', 'cpu_threads': 2}]:
            spec = deepcopy(SPEC)
            spec['required_execution'] = requirement
            with self.subTest(requirement=requirement), self.assertRaises(ValueError):
                model._required_execution_guard(spec, OPTIONS)


if __name__ == '__main__':
    unittest.main()
