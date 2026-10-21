"""Scoring protocol fixtures; never demonstration footage or model evidence."""
from pathlib import Path
import importlib


def evaluator(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return importlib.import_module('evaluate_construction_round3')


def test_unsupported_only_frame_does_not_claim_false_alert_or_zero_miss(monkeypatch):
    module=evaluator(monkeypatch)
    result=module.score_frame([{'label':'crack','bbox':[0,0,10,10],'score':.8}],
                             [{'id':'external-spall','label':'spalling','bbox':[0,0,10,10]}],{'crack'})
    assert result['evaluated'] is False
    assert result['false_alerts'] is None
    assert result['unsupported_defects']==1
    assert result['predictions'][0]['evaluation_status']=='unscored'
    assert result['by_class']['crack']['fp']==0


def test_common_class_scope_uses_same_annotated_frames_and_true_negatives(monkeypatch):
    module=evaluator(monkeypatch)
    prediction={'id':1,'label':'crack','bbox':[0,0,10,10],'score':.8}
    records=[{'expected':[{'id':'c','label':'crack','bbox':[0,0,10,10]}],'predictions':[prediction]},
             {'expected':[{'id':'s','label':'spalling','bbox':[0,0,10,10]}],'predictions':[prediction]},
             {'expected':[],'predictions':[prediction],'negative':True}]
    scoped=module.class_annotation_scope(records,'crack',{'crack'})
    assert len(scoped)==2  # No unsupported-only frame, no invented healthy exposure.
    summary=module.summarize(scoped)
    assert (summary['localized_matches'],summary['missed_defects'],summary['false_alerts'])==(1,0,1)
