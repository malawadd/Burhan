"""Identical NN bytes can have different admission policies; events must say which."""
from types import SimpleNamespace

from edge_runtime.contracts import FrameContext, InspectionResult
from edge_runtime.profiles.factory import FactoryProfile
from edge_runtime.runtime import EdgeRuntime, RuntimeConfig


def test_profile_and_event_distinguish_gate_only_variant_without_loading_nn():
    policy = {'variant': 'fit-admission-v1', 'parent_metadata_sha256': 'a' * 64,
              'changed_runtime_fields': ['supported_height_width_ratio']}
    model = SimpleNamespace(metadata={'supported_height_width_ratio': [2.4206, 3.47576],
                                     'admission_variant': policy},
                            backend='spatial_neural', requested_backend='auto',
                            backend_fallback_reason=None, native=None, neural=None,
                            path='same-nn-new-policy.onnx', sha256='b' * 64,
                            metadata_sha256='c' * 64, pixel_threshold=.999,
                            area_threshold=4)
    profile = object.__new__(FactoryProfile)
    profile._defect_model, profile.config = model, {}
    description = profile.describe()
    edge = object.__new__(EdgeRuntime)
    edge.config = RuntimeConfig(profile='factory', source='image:real-input.jpg')
    result = edge._serialize_result(InspectionResult('inconclusive', 'View rejected'),
        FrameContext(1, 1.0, source_id='image:real-input.jpg'), description, 1, 0, False)
    saved = result['model']
    assert saved['model_sha256'] == 'b' * 64
    assert saved['model_metadata_sha256'] == 'c' * 64
    assert saved['supported_height_width_ratio'] == [2.4206, 3.47576]
    assert saved['admission_variant'] == policy
    model.metadata['supported_height_width_ratio'][0] = 99
    policy['changed_runtime_fields'].append('not-allowed')
    description['admission_variant']['changed_runtime_fields'].append('later-change')
    assert saved['supported_height_width_ratio'] == [2.4206, 3.47576]
    assert saved['admission_variant']['changed_runtime_fields'] == ['supported_height_width_ratio']
