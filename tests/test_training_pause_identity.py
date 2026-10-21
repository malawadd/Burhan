"""Never suspend real processes in tests; verify the identity guard independently."""
from types import SimpleNamespace

import pytest

from scripts.benchmark_factory_with_training_pause import ROOT, TRAINING_NAME, verified_training


class FakeTraining:
    created=123.0
    directory=str(ROOT)
    command=['python','scripts/train_construction_damage_round3.py','--name',TRAINING_NAME]
    rss=1024**3
    def __init__(self,pid):
        self.pid=pid
    def create_time(self):
        return self.created
    def cwd(self):
        return self.directory
    def cmdline(self):
        return self.command
    def memory_info(self):
        return SimpleNamespace(rss=self.rss)


def test_exact_owned_training_identity_is_accepted():
    assert verified_training(123,123.0,process_factory=FakeTraining,require_actual=True).pid==123


@pytest.mark.parametrize('field,value',[
    ('created',124.0),('directory',str(ROOT.parent)),
    ('command',['python',str(ROOT.parent/'train_construction_damage_round3.py'),'--name',TRAINING_NAME]),
    ('command',['python','scripts/train_construction_damage_round3.py','--name','another-run']),
    ('rss',4*1024**2),
])
def test_reused_pid_other_workspace_script_run_or_launcher_is_rejected(field,value):
    changed=type('ChangedTraining',(FakeTraining,),{field:value})
    with pytest.raises(RuntimeError):
        verified_training(123,123.0,process_factory=changed,require_actual=True)
