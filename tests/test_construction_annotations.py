import numpy as np
from scripts.construction_annotations import publisher_labels_at_or_after


def test_past_nearest_is_rejected_in_favor_of_bounded_next_annotation():
    rows=np.array([[99,0,1,2,3,4],[106,1,5,6,7,8]],dtype=float)
    selected=publisher_labels_at_or_after(rows,100,20)
    assert selected[:,0].tolist()==[106]
    assert selected[0,1]==1


def test_distant_future_is_not_carried_into_this_camera_exposure():
    rows=np.array([[99,0,1,2,3,4],[1200,1,5,6,7,8]],dtype=float)
    assert len(publisher_labels_at_or_after(rows,100,20))==0


def test_exact_timestamp_preserves_all_simultaneous_boxes_and_original_time():
    rows=np.array([[100,0,1,2,3,4],[100,1,5,6,7,8],[106,0,1,2,3,4]],dtype=float)
    selected=publisher_labels_at_or_after(rows,100,20)
    np.testing.assert_array_equal(selected,rows[:2])


def test_gap_boundary_includes_publisher_half_period_plus_1000_microseconds():
    rows=np.array([[1110,0,1,2,3,4],[1111,1,5,6,7,8]],dtype=float)
    assert publisher_labels_at_or_after(rows,100,20)[:,0].tolist()==[1110]
    assert len(publisher_labels_at_or_after(rows[1:],100,20))==0
