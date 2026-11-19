"""Resume complete official epochs, bounded at 90; no MSP/ABC dispatch."""
import argparse,fcntl,json,math,os,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent
def atomic(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(d,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)));tmp.replace(p)
def submit(target):
    assert 1<=target<=90
    assert not (R/'STOPPED_BY_USER.json').exists()
    audit=json.loads((R/'preflight/COMPLETE.json').read_text());assert audit['passed']
    tests=[t for t in audit['tests'] if t['epoch_mode']==(1 if target<=30 else 31)]
    seconds=max(t.get('warm_seconds',t['seconds']) for t in tests)
    # One full epoch, four ranks; allow validation, DDP, checkpoint I/O and cold imports.
    minutes=max(90,math.ceil((seconds*audit['sampler_batches']*1.5+3000)/900)*15)
    assert minutes<=2880,'Measured epoch exceeds partition limit; inspect before submitting'
    peak=max(t['peak_reserved_gib'] for t in audit['tests'])
    partition='@GPU_PARTITIONS@'  # All listed devices must satisfy preflight memory requirements.
    host_memory=max(32,math.ceil((audit['host_peak_rss_gib']*4*1.3+8)/8)*8)
    with (R/'train_jobs.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        p=R/'train_jobs.json';jobs=json.loads(p.read_text()) if p.exists() else {}
        key=str(target)
        if key in jobs:print('ALREADY_SUBMITTED',jobs[key],flush=True);return
        cmd=['sbatch','--parsable','--account=@GPU_ACCOUNT@','--partition='+partition,
             '--nodes=1','--ntasks=1','--cpus-per-task=16','--mem='+str(host_memory)+'G','--gres=gpu:4',
             '--time='+str(minutes),'--job-name=paper_epoch'+key,'--chdir='+str(R),
             '--output='+str(R/'logs/%x_%j.log'),str(R/'train_job.sh'),key]
        job=subprocess.check_output(cmd,text=True).strip().split(';')[0];assert job.isdigit()
        jobs[key]={'job':job,'target_epoch':target,'minutes':minutes,'gpus':4,'cpus':16,'host_memory':str(host_memory)+'G','partition':partition,
                   'reason':'Official run.sh uses four GPUs; global batch_bins retained and each batch sharded across ranks; full-epoch time estimated from GPU preflight'}
        atomic(p,jobs);print('TRAIN_SUBMITTED',json.dumps(jobs[key]),flush=True)
def run(target):
    assert not (R/'STOPPED_BY_USER.json').exists()
    assert (R/'preflight/COMPLETE.json').exists()
    import torch
    progress=R/'train/PROGRESS.json'
    prior=json.loads(progress.read_text())['epoch'] if progress.exists() else 0
    assert prior==target-1,(prior,target)
    if target==1:assert not (R/'train/checkpoint.pth').exists(),'Refuse undocumented initialization'
    started=time.time()
    subprocess.run([str(R/'recipe.sh'),'--stage','9','--stop_stage','9',
        '--python',str(R/'python.sh'),'--tok_args','--max_epoch '+str(target)],check=True)
    ck=torch.load(R/'train/checkpoint.pth',map_location='@CPU_PARTITION@',mmap=True,weights_only=False)
    assert ck['reporter']['epoch']==target
    assert len(ck['optimizers'])==2 and all(o['state'] for o in ck['optimizers'])
    lrs=[[g['lr'] for g in o['param_groups']] for o in ck['optimizers']]
    expected=1e-4 if target<30 else 1e-5
    assert all(abs(lr-expected)<1e-10 for x in lrs for lr in x),lrs
    atomic(progress,{'epoch':target,'full_epochs':target,'phase_completed':'frozen' if target<=30 else 'joint',
        'learning_rates_after_epoch':lrs,'total_count':ck['reporter']['stats'][target]['train']['total_count'],
        'job':os.environ.get('SLURM_JOB_ID'),'wall_seconds':time.time()-started})
    atomic(R/'train/reporter.json',ck['reporter'])
    del ck
    if target<90:submit(target+1)
    else:
        atomic(R/'TOKENIZER_TRAINING_COMPLETE.json',{'epochs':90,'frozen':30,'joint':60,
          'next':'Train a separate LJSpeech unit HiFiGAN and compare paper metrics; VCTK warmup CER is not Table3 WER.'})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['submit','run']);p.add_argument('target',type=int);a=p.parse_args()
    globals()[a.action](a.target)
