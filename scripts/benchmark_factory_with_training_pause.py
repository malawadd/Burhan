"""Guarded factory checks; actual suspend requires explicit coordinator execution."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib,json,math,os
from pathlib import Path
import signal,subprocess,sys,time,uuid

ROOT=Path(__file__).resolve().parents[1]
TRAINING_NAME='evcivil-yolov8n-r3-adapted'
RESUME_MARGIN=15.
BASELINE_MODEL_SHA='dd5019cbcbc859c4eeb994a7639fb0c520ee3e3175e47a3f1544ec4c5dd60753'
BASELINE_METADATA_SHA='25d8d250c3abc0deab82994c3cc48f6f5d356058bbd74ab4903b9cd497f1ba78'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()

def verified_training(pid,created,*,process_factory=None,require_actual=False):
    if process_factory is None:
        import psutil
        process_factory=psutil.Process
    process=process_factory(pid)
    if abs(process.create_time()-created)>.01:raise RuntimeError('Training PID reused')
    if Path(process.cwd()).resolve()!=ROOT:raise RuntimeError('Different training workspace')
    command=process.cmdline()
    expected=ROOT/'scripts/train_construction_damage_round3.py'
    if not any((Path(w) if Path(w).is_absolute() else ROOT/w).resolve()==expected for w in command if not w.startswith('--')):
        raise RuntimeError('Not the exact workspace training script')
    if '--name' not in command or command.index('--name')+1>=len(command) or command[command.index('--name')+1]!=TRAINING_NAME:
        raise RuntimeError('Not the specific coordinated continuation')
    if require_actual and process.memory_info().rss<128*1024*1024:
        raise RuntimeError('Tiny launcher is not the actual Torch worker')
    return process

def write(path,value):
    """Atomic same-directory JSON replacement; no partial concurrent markers."""
    path=Path(path);temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
        os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)

def read_marker(path,token):
    try:
        value=json.loads(Path(path).read_text(encoding='utf-8'))
        return value if isinstance(value,dict) and value.get('token')==token else None
    except (OSError,ValueError,TypeError):return None

def windows_in_job():
    """Read this process's job membership without changing host policy."""
    if os.name!='nt':raise RuntimeError('Verified Windows watchdog required')
    import ctypes
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    kernel.IsProcessInJob.argtypes=[wintypes.HANDLE,wintypes.HANDLE,ctypes.POINTER(wintypes.BOOL)]
    kernel.IsProcessInJob.restype=wintypes.BOOL
    result=wintypes.BOOL()
    if not kernel.IsProcessInJob(kernel.GetCurrentProcess(),None,ctypes.byref(result)):
        raise ctypes.WinError(ctypes.get_last_error())
    return bool(result.value)

def wait_ready(guard,directory,token,*,timeout=15,clock=time.monotonic,sleep=time.sleep,descendant_pids=None):
    deadline=clock()+timeout
    while clock()<deadline:
        if guard.poll() is not None:raise RuntimeError('Watchdog exited before readiness')
        ready=read_marker(directory/'watchdog-ready.json',token)
        if ready:
            actual_pid=ready.get('pid')
            if type(actual_pid) is not int or ready.get('in_windows_job') is not False:
                raise RuntimeError('Watchdog did not verify independence')
            if actual_pid!=guard.pid:
                # Windows venv redirectors may spawn the real interpreter. Its
                # ready acknowledgement must belong to this launcher's tree.
                if descendant_pids is None:
                    import psutil
                    descendants={p.pid for p in psutil.Process(guard.pid).children(recursive=True)}
                else:descendants=set(descendant_pids())
                if actual_pid not in descendants:raise RuntimeError('Watchdog ready PID is not an owned interpreter descendant')
            if type(ready.get('deadline_monotonic')) not in (int,float) or not math.isfinite(ready['deadline_monotonic']) or ready['deadline_monotonic']<=clock()+RESUME_MARGIN:
                raise RuntimeError('Invalid watchdog deadline')
            return ready
        sleep(.1)
    raise RuntimeError('Watchdog readiness timeout; no pause allowed')

def remaining_timeout(requested,deadline,*,clock=time.monotonic):
    remaining=deadline-clock()-RESUME_MARGIN
    if remaining<1:raise RuntimeError('Pause budget exhausted; recovery margin reserved')
    return min(float(requested),remaining)

def cpu_seconds(process):
    value=process.cpu_times()
    return float(value.user+value.system)

def assert_quiescent(process,baseline,*,guard,directory,token,tolerance=.02):
    if guard.poll() is not None:raise RuntimeError('Watchdog died during pause')
    if read_marker(directory/'resume-watchdog.json',token):raise RuntimeError('Watchdog fired; timing invalid')
    if process.status()!='stopped':raise RuntimeError('Trainer is not suspended; timing invalid')
    used=cpu_seconds(process)-baseline
    if used>tolerance:raise RuntimeError(f'Training CPU advanced {used:.6f}s; timing invalid')
    return used

def snapshot_tree(launcher,owned,*,process_factory,forbidden=()):
    """Capture redirector and real interpreter descendants for this phase."""
    process=process_factory(launcher.pid)
    for item in [process,*process.children(recursive=True)]:
        if item.pid in forbidden:raise RuntimeError('Protected process in benchmark tree')
        owned.setdefault(item.pid,item.create_time())

def gone(error):
    return isinstance(error,ProcessLookupError) or type(error).__name__ in {'NoSuchProcess','ZombieProcess'}

def cleanup_tree(launcher,owned,*,process_factory,wait_procs,forbidden=()):
    """Only exact owned phase descendants, never arbitrary Python processes."""
    notes=[]
    try:snapshot_tree(launcher,owned,process_factory=process_factory,forbidden=forbidden)
    except Exception as error:
        if not gone(error):raise RuntimeError('Cannot verify benchmark descendant tree') from error
        notes.append(repr(error))
    targets=[]
    for pid,created in reversed(list(owned.items())):
        if pid in forbidden:raise RuntimeError('Refusing protected trainer/watchdog')
        try:
            item=process_factory(pid)
            if abs(item.create_time()-created)<=.01 and item.is_running():targets.append(item)
        except Exception as error:
            if not gone(error):raise RuntimeError('Cannot verify owned benchmark PID for cleanup') from error
            continue
    for item in targets:
        try:item.terminate()
        except Exception as error:notes.append(repr(error))
    _,alive=wait_procs(targets,timeout=3)
    for item in alive:
        try:
            if abs(item.create_time()-owned[item.pid])<=.01:item.kill()
        except Exception as error:notes.append(repr(error))
    _,alive=wait_procs(alive,timeout=3)
    if alive:raise RuntimeError('Owned benchmark children survived cleanup')
    return notes

@contextmanager
def defer_keyboard_interrupts():
    previous={}
    for name in ('SIGINT','SIGBREAK'):
        value=getattr(signal,name,None)
        if value is not None:
            previous[value]=signal.getsignal(value)
            signal.signal(value,signal.SIG_IGN)
    try:yield
    finally:
        for value,handler in previous.items():signal.signal(value,handler)

def recover_trainer(attempted,process_getter,audit,*,clock=time.monotonic,sleep=time.sleep,started=None):
    """Attempt bookkeeping precedes suspend, including interrupted calls."""
    if not attempted:return True
    try:
        current=process_getter()
        audit['pause_duration_seconds']=clock()-started if started is not None else None
        audit['status_before_resume']=current.status()
        current.resume()
        audit.update(resume_requested_utc=now(),status_after_resume=current.status())
        if current.status()=='stopped':raise RuntimeError('Trainer remains suspended')
        cpu0=cpu_seconds(current);sleep(.5)
        audit['training_cpu_seconds_in_resume_confirmation']=cpu_seconds(current)-cpu0
        audit['resume_verified']=True
        return True
    except BaseException as error:
        audit.update(resume_verified=False,resume_error=repr(error),completed=False)
        return False  # Never disarm independent watchdog after failed recovery.

def watchdog(args,directory):
    report={'token':args.watchdog,'training_pid':args.training_pid}
    try:
        if windows_in_job():raise RuntimeError('Watchdog remains in a Windows job')
        verified_training(args.training_pid,args.training_created)
        deadline=time.monotonic()+args.maximum_pause
        write(directory/'watchdog-ready.json',{'token':args.watchdog,'pid':os.getpid(),
              'in_windows_job':False,'ready_utc':now(),'deadline_monotonic':deadline})
        while time.monotonic()<deadline:
            if read_marker(directory/'resume-disarmed.json',args.watchdog):return
            time.sleep(.25)
        report['triggered_utc']=now()
        if read_marker(directory/'pause-attempt.json',args.watchdog):
            process=verified_training(args.training_pid,args.training_created)
            report['status_before']=process.status()
            process.resume()
            report.update(resume_requested=True,status_after=process.status())
        else:report['resume_requested']=False
    except BaseException as error:report['error']=repr(error)
    write(directory/'resume-watchdog.json',report)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-pid',type=int,required=True)
    parser.add_argument('--training-created',type=float,required=True)
    parser.add_argument('--after',type=Path);parser.add_argument('--freeze',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--maximum-pause',type=float,default=360);parser.add_argument('--watchdog')
    parser.add_argument('--probe-watchdog',action='store_true',help='Identity/readiness/Windows independence only; NEVER suspend or resume training')
    args=parser.parse_args();directory=args.output_dir.resolve()
    if not directory.is_relative_to((ROOT/'output').resolve()):raise RuntimeError('Output outside workspace')
    directory.mkdir(parents=True,exist_ok=True)
    if not 120<=args.maximum_pause<=600:raise RuntimeError('Watchdog bound must be120–600s')
    if args.watchdog:watchdog(args,directory);return
    import psutil
    if not args.probe_watchdog:
        if args.after is None or args.freeze is None:parser.error('--after and --freeze required')
        winner=json.loads(args.freeze.read_text(encoding='utf-8'))['winner']
        if sha(args.after)!=winner['model_sha256'] or sha(args.after.with_suffix('.onnx.json'))!=winner['metadata_sha256']:
            raise RuntimeError('Model/metadata differs from frozen winner')
        before=ROOT/'models/factory/spatial_unet_v10h_sdd2-step300.onnx'
        if sha(before)!=BASELINE_MODEL_SHA or sha(before.with_suffix('.onnx.json'))!=BASELINE_METADATA_SHA:
            raise RuntimeError('Preserved baseline model/metadata changed')
    getter=lambda:verified_training(args.training_pid,args.training_created)
    process=verified_training(args.training_pid,args.training_created,require_actual=True)
    if process.status()=='stopped':raise RuntimeError('Already stopped; refusing external suspension')
    audit_path=directory/'training-pause-audit.json'
    if audit_path.exists():raise RuntimeError('Preserve previous pause audit')
    token=uuid.uuid4().hex
    audit={'training_pid':process.pid,'training_created':process.create_time(),'training_command':process.cmdline(),
           'training_cwd':process.cwd(),'training_state':'Owned CPU fixed-work trainer; workers0; optimizer/RNG/memory retained',
           'maximum_pause_seconds':args.maximum_pause,'freeze_sha256':sha(args.freeze) if args.freeze else None,
           'model_sha256':sha(args.after) if args.after else None,'probe_only':args.probe_watchdog,
           'phases':[],'completed':False,'pause_attempted':False}
    guard=None;attempted=False;started=None;resume_ok=False
    try:
        if os.name!='nt':raise RuntimeError('Windows breakaway watchdog required')
        flags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_BREAKAWAY_FROM_JOB
        # Breakaway denial aborts before suspension; no weaker fallback.
        # Launch the actual interpreter, avoiding a venv redirector and its
        # additional child-job lifetime. Keep the same trusted psutil package
        # directory; the actual watchdog must still prove it is outside jobs.
        base_python=Path(sys._base_executable).resolve()
        package_directory=(Path(sys.prefix)/'Lib/site-packages').resolve()
        if package_directory!=(ROOT/'.venv-roads-export/Lib/site-packages').resolve():
            raise RuntimeError('Watchdog must use the trusted export environment package directory')
        script_path=str(Path(__file__).resolve())
        bootstrap=(f"import sys,runpy;sys.path.insert(0,{str(package_directory)!r});"
                   f"sys.argv=[{script_path!r},*sys.argv[1:]];"
                   f"runpy.run_path({script_path!r},run_name='__main__')")
        guard=subprocess.Popen([str(base_python),'-I','-S','-c',bootstrap,'--training-pid',str(args.training_pid),
              '--training-created',str(args.training_created),'--output-dir',str(directory),
              '--maximum-pause',str(args.maximum_pause),'--watchdog',token],cwd=ROOT,
              stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=flags)
        ready=wait_ready(guard,directory,token)
        audit.update(watchdog_pid=guard.pid,watchdog_ready=ready)
        if args.probe_watchdog:
            audit.update(completed=True,no_training_process_control=True)
            return  # finally disarms guard; no pause-attempt marker exists.
        deadline=ready['deadline_monotonic'];started=time.monotonic()
        write(directory/'pause-attempt.json',{'token':token,'attempt_utc':now()})
        attempted=True;audit['pause_attempted']=True
        process.suspend()
        audit.update(pause_started_utc=now(),status_after_suspend=process.status())
        baseline=cpu_seconds(process);audit['cpu_at_pause_seconds']=baseline
        time.sleep(.25)
        assert_quiescent(getter(),baseline,guard=guard,directory=directory,token=token)
        write(audit_path,audit)
        runtime_python=ROOT/'.venv/Scripts/python.exe'
        before=ROOT/'models/factory/spatial_unet_v10h_sdd2-step300.onnx'
        def isolated():
            remaining_timeout(1,deadline)
            return assert_quiescent(getter(),baseline,guard=guard,directory=directory,token=token)
        def run(name,script,arguments,timeout):
            isolated();limit=remaining_timeout(timeout,deadline)
            phase={'name':name,'started_utc':now(),'script':script,'arguments':list(map(str,arguments)),
                   'timeout_seconds':limit}
            audit['phases'].append(phase);write(audit_path,audit)
            print(json.dumps({'phase':name,'state':'running'}),flush=True)
            child=None;owned={};cleanup_ok=False
            try:
                with (directory/(name+'.log')).open('w',encoding='utf-8') as log:
                    with defer_keyboard_interrupts():
                        child=subprocess.Popen([str(runtime_python),'-X','utf8',str(ROOT/'scripts'/script),*map(str,arguments)],
                              cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                              creationflags=subprocess.CREATE_NO_WINDOW)
                        snapshot_tree(child,owned,process_factory=psutil.Process,forbidden=(process.pid,guard.pid,ready['pid']))
                    phase_deadline=time.monotonic()+limit
                    while True:
                        try:snapshot_tree(child,owned,process_factory=psutil.Process,forbidden=(process.pid,guard.pid))
                        except psutil.NoSuchProcess:
                            if child.poll() is None:raise
                        isolated();result=child.poll()
                        if result is not None:break
                        if time.monotonic()>=phase_deadline:raise subprocess.TimeoutExpired(script,limit)
                        time.sleep(.25)
                phase.update(completed_utc=now(),returncode=result)
                if result:raise RuntimeError(f'{name} failed; original log retained')
            finally:
                if child is not None:
                    with defer_keyboard_interrupts():
                        phase['cleanup_notes']=cleanup_tree(child,owned,process_factory=psutil.Process,
                             wait_procs=psutil.wait_procs,forbidden=(process.pid,guard.pid))
                        cleanup_ok=True
                phase['owned_processes']=[{'pid':pid,'created':created} for pid,created in owned.items()]
                phase['child_cleanup_verified']=cleanup_ok
                write(audit_path,audit)
            isolated()
        run('calibration-capacity','benchmark_factory_accuracy.py',['--before',before,'--after',args.after,
            '--split','calibration','--repeat',3,'--output',directory/'calibration-paired-profile-capacity.json'],120)
        capacity=json.loads((directory/'calibration-paired-profile-capacity.json').read_text(encoding='utf-8'))
        rate=capacity['results']['after']['process_ms']['serial_capacity_fps'];audit['capacity_gate_fps']=rate
        if type(rate) not in (int,float) or not math.isfinite(rate) or rate<=15:
            raise RuntimeError('Frozen winner failed finite>15/s capacity; original80 not scored')
        run('localization-audit','factory_localization_audit.py',['--model',args.after,'--output',directory/'after-localization-audit.json',
            '--visuals-dir',directory/'after-masks'],90)
        for tag,model in [('before',before),('after',args.after)]:
            run(tag+'-runtime','benchmark_runtime.py',['--varied','--fps',16,'--seconds',60,'--deadline',100,
                '--preview-fps',6,'--model',model,'--output',directory/(tag+'-runtime-16fps-60s.json'),'--trace'],120)
        audit['cpu_seconds_during_pause']=isolated();audit['completed']=True
    except BaseException as error:
        audit.update(completed=False,error=repr(error))
        raise
    finally:
        with defer_keyboard_interrupts():
            resume_ok=recover_trainer(attempted,getter,audit,started=started)
            audit['watchdog_fired']=bool(read_marker(directory/'resume-watchdog.json',token))
            if audit['watchdog_fired']:audit['completed']=False
            if not attempted or resume_ok:write(directory/'resume-disarmed.json',{'token':token,'completed_utc':now()})
            write(audit_path,audit)  # Failed recovery deliberately leaves guard armed.
            if guard is not None and (not attempted or resume_ok):
                try:guard.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    audit['watchdog_exit_pending']=True;write(audit_path,audit)
            print(json.dumps({'pause_audit':str(audit_path),'resume_verified':audit.get('resume_verified',not attempted),
                              'watchdog_armed':attempted and not resume_ok,'completed':audit.get('completed')}),flush=True)
        if attempted and not resume_ok:raise RuntimeError('Resume verification failed; independent guard remains armed')
        if audit['watchdog_fired']:raise RuntimeError('Watchdog fired; benchmark block invalidated')

if __name__=='__main__':main()

