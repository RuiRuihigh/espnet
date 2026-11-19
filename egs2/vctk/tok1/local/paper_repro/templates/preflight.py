"""Check the unchanged official model with actual paper-recipe batch sizes."""
import gc,json,math,os,subprocess,time,collections,resource
from pathlib import Path
import numpy as np,soundfile as sf,kaldiio,torch
from espnet2.tasks.gan_tok import GANSpeechTokenizerTask as Task
from espnet2.samplers.num_elements_batch_sampler import NumElementsBatchSampler
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
R=Path(__file__).resolve().parent;recipe=R/'espnet/egs2/vctk/tok1'
assert not (R/'STOPPED_BY_USER.json').exists()
assert (R/'SETUP_OK.json').exists() and (R/'DATA_READY.json').exists()
assert (R/'stages/stats.done').exists()
subprocess.run(['git','diff','--exit-code','HEAD','--','espnet2','egs2/TEMPLATE'],cwd=R/'espnet',check=True)
out=R/'preflight';out.mkdir(exist_ok=True)
set_all_random_seed(0);torch.set_num_threads(2)
args=Task.get_parser().parse_args(['--config',str(recipe/'conf/train_paper.yaml'),
 '--token_list',str(recipe/'data/noinfo_token_list/bpe_unigram5000/tokens.txt'),
 '--bpemodel',str(recipe/'data/noinfo_token_list/bpe_unigram5000/bpe.model'),
 '--token_type','bpe','--use_preprocessor','true','--output_dir',str(out/'scratch')])
assert args.num_iters_per_epoch is None and args.max_epoch==90
assert args.tokenizer_conf['freeze_epochs']==30 and args.batch_bins==16000000
assert not hasattr(args,'prosody_conf')
torch.backends.cudnn.benchmark=args.cudnn_benchmark
torch.backends.cudnn.deterministic=args.cudnn_deterministic
shapes=[str(R/'stats/train'/name) for name in ['speech_shape','text_shape.bpe','spembs_shape']]
sampler=list(NumElementsBatchSampler(batch_bins=args.batch_bins,shape_files=shapes,
 min_batch_size=4,sort_in_batch=args.sort_in_batch,sort_batch=args.sort_batch,drop_last=args.drop_last_iter))
wav=dict(l.split(maxsplit=1) for l in (R/'dump/raw/tr_no_dev_sp/wav.scp').read_text().splitlines())
texts=dict(l.split(maxsplit=1) for l in (R/'dump/raw/tr_no_dev_sp/text').read_text().splitlines())
sp=kaldiio.load_scp(str(R/'dump/ecapa192/tr_no_dev_sp/ecapa192.scp'))
lengths={l.split()[0]:int(l.split()[1].split(',')[0]) for l in (R/'stats/train/speech_shape').read_text().splitlines()}
assert collections.Counter(u for b in sampler for u in b)==collections.Counter(wav.keys())
# ESPnet shards each global batch across ranks, not the list of batches.
global_sampler=sampler
sampler=[b[0::4] for b in global_sampler]
assert all(sampler)
indices=sorted({max(range(len(sampler)),key=lambda i:max(lengths[u] for u in sampler[i])),
                max(range(len(sampler)),key=lambda i:len(sampler[i])*max(lengths[u] for u in sampler[i])),
                len(sampler)//2})
model=Task.build_model(args).cuda();assert model.tokenizer.num_clusters==2000
assert set(model.objectives)=={'asr','reconstruction'}
pp=Task.build_preprocess_fn(args,True);collate=Task.build_collate_fn(args,True)
def batch(ids):
    xs=[]
    for u in ids:
        wave,sr=sf.read(wav[u],dtype='float32');assert sr==16000
        speaker=np.asarray(sp[u],dtype='float32').reshape(-1);assert speaker.shape==(192,)
        xs.append((u,pp(u,{'speech':wave,'text':texts[u],'spembs':speaker})))
    _,bt=collate(xs);return {k:v.cuda() for k,v in bt.items()}
results=[]
try:
 for epoch in [1,31]:
    model.train();model.tokenizer.set_epoch(epoch);opt=Task.build_optimizers(args,model)
    for idx in indices:
        bt=batch(sampler[idx]);torch.cuda.reset_peak_memory_stats();t=time.time()
        for o in opt:o.zero_grad(set_to_none=True)
        g=model(**bt);assert torch.isfinite(g['loss']);g['loss'].backward()
        encoder_grad=sum(float(p.grad.abs().sum()) for p in model.tokenizer.frontend.parameters() if p.grad is not None)
        centroid_grad=model.tokenizer.quantizer.centroids.grad
        centroid_grad=0.0 if centroid_grad is None else float(centroid_grad.abs().sum())
        if epoch==1:assert encoder_grad==0 and centroid_grad==0
        else:assert encoder_grad>0 and centroid_grad>0
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip);assert torch.isfinite(norm)
        opt[0].step()
        for o in opt:o.zero_grad(set_to_none=True)
        d=model(forward_generator=False,**bt);assert torch.isfinite(d['loss']);d['loss'].backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip);assert torch.isfinite(norm)
        opt[1].step()
        for o in opt:o.zero_grad(set_to_none=True)
        torch.cuda.synchronize();record={'epoch_mode':epoch,'sampler_batch':idx,'batch_size':len(sampler[idx]),'global_batch_size':len(global_sampler[idx]),
          'max_audio_seconds':max(lengths[u] for u in sampler[idx])/16000,
          'seconds':time.time()-t,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
          'encoder_grad_l1':encoder_grad,'centroid_grad_l1':centroid_grad}
        warm=time.time()
        for generator,optimizer in [(True,opt[0]),(False,opt[1])]:
            v=model(forward_generator=generator,**bt);assert torch.isfinite(v['loss']);v['loss'].backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip);assert torch.isfinite(norm)
            optimizer.step()
            for o in opt:o.zero_grad(set_to_none=True)
            del v
        torch.cuda.synchronize();record['warm_seconds']=time.time()-warm
        record['peak_reserved_gib']=torch.cuda.max_memory_reserved()/2**30
        results.append(record);print('PREFLIGHT_BATCH',json.dumps(record),flush=True)
        del bt,g,d;gc.collect();torch.cuda.empty_cache()
    del opt;model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
except Exception as e:
    (out/'FAILED.json').write_text(json.dumps({'error':repr(e),'completed_tests':results},indent=2));raise
peak=max(r['peak_reserved_gib'] for r in results)
assert peak+3 < torch.cuda.get_device_properties(0).total_memory/2**30,'Insufficient DDP memory headroom; do not alter paper batch automatically'
report={'passed':True,'official_batch_bins':args.batch_bins,'sampler_batches':len(sampler),'train_utterances':len(wav),
 'num_iters_per_epoch':None,'training_gpus':4,'preflight_rank':0,'batch_sharding':'Official batch[rank::4]; one rank measured, DDP headroom reserved','gpu':torch.cuda.get_device_name(),'tests':results,
 'host_peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
 'scratch_weights_discarded':True,'single_codebook':2000,'frozen_and_joint_backward_passed':True,
 'checkpoint_initialization':'Official pretrained WavLM + new Libri30h centroids; ASR/vocoder fresh, no MSP weights'}
(out/'COMPLETE.json').write_text(json.dumps(report,indent=2));print('PREFLIGHT_COMPLETE',flush=True)
print('PREFLIGHT PASSED. Review COMPLETE.json; then explicitly run: python train_chain.py submit 1', flush=True)
