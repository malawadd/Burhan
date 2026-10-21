"""Damage events must identify the selected local detector and its class cuts."""
from edge_runtime.contracts import FrameContext, InspectionResult
from edge_runtime.runtime import EdgeRuntime, RuntimeConfig


def test_damage_event_retains_exact_model_and_operating_point():
    edge = object.__new__(EdgeRuntime)
    edge.config = RuntimeConfig(profile='construction',source='image:real-input.jpg')
    metadata = {'mode':'damage','damage_variant':'frozen_local_detector',
                'sha256':'a'*64,'classes':['spalling','crack'],
                'class_confidence':{'spalling':.45,'crack':.65},
                'selected_damage_classes':['spalling'],
                'selected_stage':'structure','confirm_frames':2,'diagnostic_only':True,
                'exposure_policy':'low_light_gain'}
    result = edge._serialize_result(InspectionResult('review','Visible defect candidate'),
        FrameContext(7,1.0,source_id='image:real-input.jpg'),metadata,40,35,False)
    assert result['model']['model_sha256'] == 'a'*64
    assert result['model']['damage_variant'] == 'frozen_local_detector'
    assert result['model']['class_confidence'] == {'spalling':.45,'crack':.65}
    assert result['model']['selected_stage'] == 'structure'
    assert result['model']['confirm_frames'] == 2
    assert result['model']['selected_damage_classes'] == ['spalling']
    assert result['model']['diagnostic_only'] is True
    assert result['model']['exposure_policy'] == 'low_light_gain'


def test_guided_event_keeps_exact_research_model_provenance():
    edge = object.__new__(EdgeRuntime)
    edge.config = RuntimeConfig(profile='construction', source='file:real-input.mp4')
    provenance = {'model_sha256': 'b'*64, 'class_confidence': {'peeling_paint': .35},
                  'classes': ['peeling_paint'], 'validation_status': 'research_only'}
    metadata = {'mode': 'guided_construction', 'inspection_task': 'walls_finishes',
                'inspection_kind': 'model', 'model_provenance': provenance}
    result = edge._serialize_result(InspectionResult('unassessed', 'Research only'),
        FrameContext(7, 1., source_id='file:real-input.mp4', source_time_ms=233.3), metadata, 40., 35., False)
    assert result['model']['inspection_task'] == 'walls_finishes'
    assert result['model']['inspection_kind'] == 'model'
    assert result['model']['guided_model_provenance'] == provenance
    provenance['class_confidence']['peeling_paint'] = .01
    assert result['model']['guided_model_provenance']['class_confidence']['peeling_paint'] == .35
