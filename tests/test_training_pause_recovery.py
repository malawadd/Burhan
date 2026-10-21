"""Fake-process safety tests only: NEVER suspend/resume a real process."""
from types import SimpleNamespace
from contextlib import nullcontext
import json
import os
import sys
import pytest
from scripts import benchmark_factory_with_training_pause as pause


class Training:
    def __init__(self,status='stopped',resume_error=None):
        self.state=status;self.resumes=0;self.resume_error=resume_error
    def status(self):return self.state
    def cpu_times(self):return SimpleNamespace(user=2.,system=1.)
    def resume(self):
        self.resumes+=1
        if self.resume_error:raise self.resume_error
        self.state='running'


class Guard:
    pid=44
    def __init__(self,result=None):self.result=result
    def poll(self):return self.result


def ready(tmp_path,**changes):
    value={'token':'token','pid':44,'in_windows_job':False,'deadline_monotonic':200.}
    value.update(changes)
    pause.write(tmp_path/'watchdog-ready.json',value)


def test_atomic_marker_and_malformed_or_stale_marker(tmp_path):
    path=tmp_path/'marker.json'
    pause.write(path,{'token':'new','data':3})
    assert pause.read_marker(path,'new')['data']==3
    assert pause.read_marker(path,'old') is None
    path.write_text('{',encoding='utf-8')
    assert pause.read_marker(path,'new') is None
    assert not list(tmp_path.glob('*.tmp'))


def test_ready_handshake_requires_alive_independent_guard(tmp_path):
    ready(tmp_path)
    assert pause.wait_ready(Guard(),tmp_path,'token',clock=lambda:100)['pid']==44
    with pytest.raises(RuntimeError,match='exited'):
        pause.wait_ready(Guard(1),tmp_path,'token',clock=lambda:100)
    ready(tmp_path,in_windows_job=True)
    with pytest.raises(RuntimeError,match='independence'):
        pause.wait_ready(Guard(),tmp_path,'token',clock=lambda:100)


def test_real_venv_watchdog_child_must_be_owned_descendant(tmp_path):
    ready(tmp_path,pid=45)
    assert pause.wait_ready(Guard(),tmp_path,'token',clock=lambda:100,descendant_pids=lambda:[45])['pid']==45
    with pytest.raises(RuntimeError,match='owned interpreter'):
        pause.wait_ready(Guard(),tmp_path,'token',clock=lambda:100,descendant_pids=lambda:[])


def test_missing_ready_times_out_before_control(tmp_path):
    values=iter([0,0,20])
    with pytest.raises(RuntimeError,match='readiness timeout'):
        pause.wait_ready(Guard(),tmp_path,'token',clock=lambda:next(values),sleep=lambda _:None)


def test_remaining_budget_reserves_recovery_margin():
    assert pause.remaining_timeout(120,200,clock=lambda:100)==85
    assert pause.remaining_timeout(10,200,clock=lambda:100)==10
    with pytest.raises(RuntimeError,match='recovery margin'):
        pause.remaining_timeout(120,115,clock=lambda:100)


@pytest.mark.parametrize('change',['running','cpu','watchdog_dead','watchdog_fired'])
def test_quiescence_failure_invalidates_timing(tmp_path,change):
    process=Training(status='running' if change=='running' else 'stopped')
    if change=='watchdog_fired':pause.write(tmp_path/'resume-watchdog.json',{'token':'token'})
    with pytest.raises(RuntimeError):
        pause.assert_quiescent(process,2 if change=='cpu' else 3,
             guard=Guard(1 if change=='watchdog_dead' else None),directory=tmp_path,token='token')


def test_quiescent_trainer_is_accepted(tmp_path):
    assert pause.assert_quiescent(Training(),3,guard=Guard(),directory=tmp_path,token='token')==0


def test_interrupted_suspend_attempt_resumes_without_success_flag_or_timer():
    process=Training();audit={}
    assert pause.recover_trainer(True,lambda:process,audit,sleep=lambda _:None,started=None)
    assert process.resumes==1 and audit['resume_verified'] and audit['pause_duration_seconds'] is None


def test_failed_resume_keeps_recovery_failure_explicit():
    process=Training(resume_error=PermissionError('fixture denial'));audit={'completed':True}
    assert not pause.recover_trainer(True,lambda:process,audit,sleep=lambda _:None)
    assert process.resumes==1 and audit['resume_verified'] is False and audit['completed'] is False


def test_no_pause_attempt_never_controls_process():
    assert pause.recover_trainer(False,lambda:pytest.fail('no process access allowed'),{})


class Owned:
    def __init__(self,pid,created=1.,children=()):
        self.pid=pid;self.created=created;self.descendants=list(children);self.running=True;self.calls=[]
    def create_time(self):return self.created
    def children(self,recursive=True):return self.descendants
    def is_running(self):return self.running
    def terminate(self):self.calls.append('terminate');self.running=False
    def kill(self):self.calls.append('kill');self.running=False


def wait_procs(items,timeout):
    return [x for x in items if not x.running],[x for x in items if x.running]


def test_cleanup_captures_venv_redirector_and_actual_nn_child():
    child=Owned(11);launcher=Owned(10,children=[child]);processes={10:launcher,11:child};owned={}
    pause.snapshot_tree(SimpleNamespace(pid=10),owned,process_factory=processes.__getitem__,forbidden=(99,100))
    pause.cleanup_tree(SimpleNamespace(pid=10),owned,process_factory=processes.__getitem__,wait_procs=wait_procs,forbidden=(99,100))
    assert owned=={10:1.,11:1.} and launcher.calls==['terminate'] and child.calls==['terminate']


def test_cleanup_never_controls_reused_or_protected_pid():
    reused=Owned(11,created=2.);launcher=Owned(10);processes={10:launcher,11:reused}
    pause.cleanup_tree(SimpleNamespace(pid=10),{10:1.,11:1.},process_factory=processes.__getitem__,wait_procs=wait_procs)
    assert not reused.calls
    protected=Owned(99);launcher.descendants=[protected]
    with pytest.raises(RuntimeError,match='descendant tree'):
        pause.cleanup_tree(SimpleNamespace(pid=10),{},process_factory=lambda pid:launcher,
                           wait_procs=wait_procs,forbidden=(99,))
    assert not protected.calls


def test_cleanup_access_denial_is_not_misreported_as_exited():
    def denied(pid):raise PermissionError('fixture')
    with pytest.raises(RuntimeError,match='descendant tree'):
        pause.cleanup_tree(SimpleNamespace(pid=10),{10:1.},process_factory=denied,wait_procs=wait_procs)


@pytest.mark.skipif(os.name!='nt',reason='Windows coordinator, fake controls only')
@pytest.mark.parametrize('probe',[False,True])
def test_main_interruption_resumes_and_probe_never_controls(monkeypatch,tmp_path,probe):
    directory=tmp_path/'output/block'
    class Fake(Training):
        pid=77
        def __init__(self):super().__init__(status='running');self.suspends=0
        def create_time(self):return 1.
        def cwd(self):return str(tmp_path)
        def cmdline(self):return ['fake-training']
        def suspend(self):
            self.suspends+=1
            assert not probe
            assert (directory/'pause-attempt.json').is_file()
            self.state='stopped'
            raise KeyboardInterrupt('fixture: successful suspend interrupted before return')
    process=Fake()
    guard=Guard();guard.wait=lambda timeout:0
    monkeypatch.setattr(pause,'ROOT',tmp_path)
    monkeypatch.setattr(sys,'prefix',str(tmp_path/'.venv-roads-export'))
    monkeypatch.setattr(pause,'verified_training',lambda *a,**kw:process)
    monkeypatch.setattr(pause.subprocess,'Popen',lambda *a,**kw:guard)
    monkeypatch.setattr(pause,'wait_ready',lambda *a,**kw:{'pid':44,'in_windows_job':False,'deadline_monotonic':1e15})
    monkeypatch.setattr(pause,'defer_keyboard_interrupts',lambda:nullcontext())
    monkeypatch.setitem(sys.modules,'psutil',SimpleNamespace())
    argv=['wrapper','--training-pid','77','--training-created','1','--output-dir',str(directory)]
    if probe:argv+=['--probe-watchdog']
    else:
        model=tmp_path/'model.onnx';model.write_bytes(b'fixture, not a neural model')
        metadata=model.with_suffix('.onnx.json');metadata.write_text('{}')
        frozen=tmp_path/'freeze.json'
        frozen.write_text(json.dumps({'winner':{'model_sha256':pause.sha(model),'metadata_sha256':pause.sha(metadata)}}))
        baseline=tmp_path/'models/factory/spatial_unet_v10h_sdd2-step300.onnx'
        baseline.parent.mkdir(parents=True)
        baseline.write_bytes(b'baseline fixture')
        baseline.with_suffix('.onnx.json').write_text('{}')
        monkeypatch.setattr(pause,'BASELINE_MODEL_SHA',pause.sha(baseline))
        monkeypatch.setattr(pause,'BASELINE_METADATA_SHA',pause.sha(baseline.with_suffix('.onnx.json')))
        argv+=['--after',str(model),'--freeze',str(frozen)]
    monkeypatch.setattr(sys,'argv',argv)
    if probe:pause.main()
    else:
        with pytest.raises(KeyboardInterrupt):pause.main()
    audit=json.loads((directory/'training-pause-audit.json').read_text())
    assert process.suspends==(0 if probe else 1) and process.resumes==(0 if probe else 1)
    assert (directory/'resume-disarmed.json').is_file()
    assert audit['completed']==probe
    if probe:assert audit['no_training_process_control']
    else:assert audit['pause_attempted'] and audit['resume_verified']
