"""Submit only paper reproduction prerequisites; training is preflight-gated."""
import argparse,fcntl,json,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--setup-job',required=True);a=p.parse_args();assert a.setup_job.isdigit()
specs=[
 ('data',['setup'],'@CPU_PARTITION@',4,'8G',180,0),
 ('subset',['setup'],'@CPU_PARTITION@',2,'4G',45,0),
 ('features',['subset'],'@GPU_PARTITIONS@',2,'16G',120,1),
 ('kmeans',['features'],'@CPU_PARTITION@',16,'128G',480,0),
 ('speakers',['data'],'@GPU_PARTITIONS@',2,'8G',240,1),
 ('stats',['data','kmeans','speakers'],'@CPU_PARTITION@',4,'32G',180,0),
 ('preflight',['stats'],'@PREFLIGHT_PARTITION@',4,'32G',90,1),
]
assert not (R/'STOPPED_BY_USER.json').exists()
with (R/'pipeline_jobs.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 path=R/'pipeline_jobs.json';jobs=json.loads(path.read_text()) if path.exists() else {'setup':{'job':a.setup_job}}
 assert jobs['setup']['job']==a.setup_job
 for name,deps,partition,cpus,mem,minutes,gpus in specs:
    if name in jobs:continue
    dependency='afterok:'+':'.join(jobs[d]['job'] for d in deps)
    cmd=['sbatch','--parsable','--account='+('@GPU_ACCOUNT@' if gpus else '@CPU_ACCOUNT@'),
      '--partition='+partition,'--nodes=1','--ntasks=1','--cpus-per-task='+str(cpus),'--mem='+mem,
      '--time='+str(minutes),'--dependency='+dependency,'--kill-on-invalid-dep=yes',
      '--job-name=paper_'+name,'--chdir='+str(R),'--output='+str(R/'logs/%x_%j.log')]
    if gpus:cmd+=['--gres=gpu:'+str(gpus)]
    if name=='data':cmd+=['--requeue']
    cmd+=[str(R/'stage.sh'),name]
    job=subprocess.check_output(cmd,text=True).strip().split(';')[0];assert job.isdigit()
    jobs[name]={'job':job,'dependencies':deps,'minutes':minutes,'gpus':gpus,'cpus':cpus,'memory':mem,'partition':partition,'submitted':time.time()}
    tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(jobs,indent=2));tmp.replace(path)
    print('SUBMITTED',name,job,flush=True)
