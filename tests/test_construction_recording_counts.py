"""Independent saved-evidence audit rejects plausible but geometrically false claims."""
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pytest
from audit_construction_recording_counts import audit_document,protect_inputs


def example():
    return {'matching_iou_threshold':.5,'frame_size_wh':[100,100],
            'source':{'original_camera_frames':1,'timebase':'one frame per exposure'},
            'derived_video':{'encoded_frames':1},
            'summary':{'annotated_frames':1,'evaluated_frames':1,'localized_matches':1,'missed_defects':0,
                       'false_alerts':0,'unsupported_defects':0,'unsupported_annotations':0,'precision':1.,'recall':1.},
            'frames':[{'source_frame_index':0,'source_timestamp_us':123456,'video_time_s':0.,'evaluated':True,
                       'expected':[{'id':'gt1','label':'crack','bbox':[10,20,30,40]}],
                       'predictions':[{'id':1,'label':'crack','bbox':[10,20,30,40],'score':.9,
                                       'matched_annotation_id':'gt1','iou':1.,'evaluation_status':'matched'}],
                       'localized_matches':1,'missed_defects':0,'false_alerts':0,'unsupported_defects':0}]}


def test_valid_actual_geometry_counts_and_complete_timeline():
    result=audit_document(example())
    assert result['passed'] and result['recomputed']['localized_matches']==1


def test_corrupted_summary_is_not_trusted():
    data=example();data['summary']['localized_matches']=42
    result=audit_document(data)
    assert not result['passed']
    assert any('summary.localized_matches' in e for e in result['errors'])
    assert result['recomputed']['localized_matches']==1


def test_plausible_reported_iou_cannot_hide_wrong_geometry():
    data=example();data['frames'][0]['predictions'][0]['bbox']=[70,70,10,10]
    result=audit_document(data)
    assert not result['passed'] and any('actual class-aware greedy match' in e for e in result['errors'])


def test_matched_iou_must_equal_actual_geometry_not_just_exceed_cutoff():
    data=example();data['frames'][0]['predictions'][0]['iou']=.6
    result=audit_document(data)
    assert not result['passed'] and any('matched IoU does not equal actual geometry' in e for e in result['errors'])


def test_same_annotation_cannot_be_credited_twice():
    data=example();extra=deepcopy(data['frames'][0]['predictions'][0]);extra['id']=2
    data['frames'][0]['predictions'].append(extra)
    assert not audit_document(data)['passed']


def test_class_mismatch_cannot_be_credited_despite_exact_box():
    data=example();data['frames'][0]['predictions'][0]['label']='spalling'
    assert not audit_document(data)['passed']


def test_unsupported_only_is_separate_from_false_negative_or_true_positive():
    data=example();frame=data['frames'][0]
    frame['evaluated']=False;frame['expected'][0].update(label='spalling',evaluation_status='unsupported')
    frame['predictions']=[];frame.update(localized_matches=0,missed_defects=0,false_alerts=None,unsupported_defects=1)
    data['summary'].update(evaluated_frames=0,localized_matches=0,missed_defects=0,false_alerts=0,
                           unsupported_defects=1,unsupported_annotations=1,precision=None,recall=None)
    result=audit_document(data)
    assert result['passed'] and result['recomputed']['unsupported_annotations']==1
    assert result['recomputed']['missed_defects']==0


def test_unsupported_aliases_must_agree_if_both_declared():
    data=example();data['summary']['unsupported_defects']=3
    assert not audit_document(data)['passed']


def test_unscored_prediction_on_verified_negative_is_not_false_alarm():
    data=example();frame=data['frames'][0];frame['expected']=[]
    frame['predictions'][0].update(matched_annotation_id=None,iou=0.,evaluation_status='unscored')
    frame.update(localized_matches=0,missed_defects=0,false_alerts=0)
    data['summary'].update(annotated_frames=0,localized_matches=0,precision=None,recall=None)
    result=audit_document(data)
    assert result['passed'] and result['recomputed']['false_alerts']==0
    assert result['recomputed']['evaluated_frames']==1 and result['recomputed']['unscored_predictions']==1


def test_unevaluated_unlabelled_prediction_is_not_healthy_negative_or_false_alarm():
    data=example();frame=data['frames'][0];frame['expected']=[];frame['evaluated']=False
    frame['predictions'][0].update(matched_annotation_id=None,iou=0.,evaluation_status='unscored')
    frame.update(localized_matches=0,missed_defects=0,false_alerts=None)
    data['summary'].update(annotated_frames=0,evaluated_frames=0,localized_matches=0,precision=None,recall=None)
    result=audit_document(data)
    assert result['passed'] and result['recomputed']['evaluated_frames']==0
    assert result['recomputed']['false_alerts']==0


def test_wrong_derived_frame_count_and_skipped_complete_indices_fail():
    data=example();data['derived_video']['encoded_frames']=2
    data['frames'][0]['source_frame_index']=4
    result=audit_document(data)
    assert not result['passed']
    assert any('contiguous source indices' in e for e in result['errors'])
    assert any('derived_video.encoded_frames' in e for e in result['errors'])


def test_original_source_timestamp_and_video_pts_must_agree():
    data=example();data['frames'][0]['video_time_s']=.3
    data['source']['timebase']='Original timestamps shifted to zero; one frame per exposure'
    assert not audit_document(data)['passed']


def test_publisher_boundary_roundoff_is_not_rejected_as_negative_geometry():
    data=example()
    data['frames'][0]['expected'][0]['bbox'][0]=-1.5e-14
    data['frames'][0]['predictions'][0]['bbox'][0]=-1.5e-14
    assert audit_document(data)['passed']


def test_corrupted_per_class_count_cannot_hide_behind_correct_global_total():
    data=example();data['summary']['by_class']={'crack':{'tp':0,'fn':0,'fp':0,'unsupported_annotations':0}}
    result=audit_document(data)
    assert not result['passed'] and any('summary.by_class.crack.tp' in e for e in result['errors'])


def test_missing_summary_and_malformed_class_summary_fail_as_audit_results():
    data=example();data['summary']={}
    assert not audit_document(data)['passed']
    data=example();data['summary']['by_class']=[]
    result=audit_document(data)
    assert not result['passed'] and 'summary.by_class must be an object' in result['errors']


def test_genuine_interval_drift_fails_even_when_both_timestamps_increase():
    data=example();second=deepcopy(data['frames'][0]);second.update(source_frame_index=1,source_timestamp_us=223456,video_time_s=.3)
    data['frames'].append(second);data['source']['original_camera_frames']=2;data['derived_video']['encoded_frames']=2
    result=audit_document(data)
    assert not result['passed'] and any('intervals differ' in e for e in result['errors'])


def test_audit_output_cannot_overwrite_read_only_evidence_or_registry(tmp_path):
    evidence=tmp_path/'evidence.json';registry=tmp_path/'registry.json'
    with pytest.raises(ValueError,match='cannot overwrite'):
        protect_inputs([evidence,registry],[evidence,tmp_path/'audit.md'])
    with pytest.raises(ValueError,match='must be distinct'):
        protect_inputs([evidence,registry],[tmp_path/'same',tmp_path/'same'])
    protect_inputs([evidence,registry],[tmp_path/'audit.json',tmp_path/'audit.md'])
