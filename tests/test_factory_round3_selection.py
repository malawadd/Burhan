"""The new rule must protect a weak positive without bypassing alarm guards."""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from calibrate_factory_round3 import point_rank,eligible_final


def test_weakest_view_rule_changes_the_aggregate_only_counterexample():
    aggregate={'meaningful_iou_0_1_recall':1.,'worst_positive_iou':.14,'coverage_50pct_recall':.9,
               'worst_positive_coverage':.15,'micro_pixel_iou':.64,'normal_fp':0,
               'foreground_pixel_recall':.77,'foreground_pixel_precision':.79}
    consistent={**aggregate,'worst_positive_iou':.31,'worst_positive_coverage':.6,
                'coverage_50pct_recall':1.,'micro_pixel_iou':.60,'normal_fp':1,'foreground_pixel_recall':.82}
    assert eligible_final(aggregate,True) and eligible_final(consistent,True)
    assert point_rank(aggregate,False)>point_rank(consistent,False)
    assert point_rank(consistent,True)>point_rank(aggregate,True)


def test_ineligible_alarm_or_pixel_precision_cannot_enter_new_ranking():
    assert eligible_final({'normal_fp':1,'foreground_pixel_precision':.65},True)
    assert not eligible_final({'normal_fp':2,'foreground_pixel_precision':.99},True)
    assert not eligible_final({'normal_fp':0,'foreground_pixel_precision':.64999},True)
