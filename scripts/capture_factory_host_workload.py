"""Read-only process CPU sample; never pause, stop or modify host processes."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,time
import psutil

ROOT=Path(__file__).resolve().parents[1]


def sample():
    rows={}
    for process in psutil.process_iter(['pid','name','exe','num_threads','cpu_times','memory_info','create_time']):
        try:
            row=process.info;cpu=row['cpu_times'];memory=row.pop('memory_info')
            row['cpu_seconds']=cpu.user+cpu.system;row.pop('cpu_times')
            row['working_set_bytes']=memory.rss
            try:row['cwd']=process.cwd()
            except (psutil.Error,OSError):row['cwd']=None
            row['project_related']=bool(row['cwd'] and str(ROOT).lower() in row['cwd'].lower())
            rows[row['pid']]=row
        except (psutil.Error,OSError,AttributeError):continue
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    first=sample();started=time.perf_counter();time.sleep(1.0);last=sample();elapsed=time.perf_counter()-started
    rows=[]
    for pid,row in last.items():
        previous=first.get(pid)
        row['cpu_percent_one_logical_core']=max(0.,100*(row['cpu_seconds']-previous['cpu_seconds'])/elapsed) if previous and previous['create_time']==row['create_time'] else None
        rows.append(row)
    rows.sort(key=lambda r:r['cpu_percent_one_logical_core'] or 0,reverse=True)
    value={'captured_utc':datetime.now(timezone.utc).isoformat(),'sample_seconds':elapsed,'logical_cpus':psutil.cpu_count(),
        'scope':'Read-only workload snapshot; one100% unit is one logical core. Project-quiescent is not whole-host-exclusive. No command-line arguments or process control collected.',
        'active_processes':[r for r in rows if (r['cpu_percent_one_logical_core'] or 0)>=1],
        'project_processes':[r for r in rows if r['project_related']]}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(args.output.resolve()),'active':[(r['pid'],r['name'],round(r['cpu_percent_one_logical_core'],1)) for r in value['active_processes'][:10]],'project_pids':[r['pid'] for r in value['project_processes']]},indent=2))


if __name__=='__main__':main()
