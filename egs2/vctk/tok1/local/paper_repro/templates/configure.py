"""Keep upstream models intact; record paper-required recipe overrides."""
import hashlib,json,subprocess,os
from pathlib import Path
import yaml
R=Path(__file__).resolve().parent;B=Path('@BASE@')
REPO=R/'espnet';RECIPE=REPO/'egs2/vctk/tok1'
COMMIT='0aff6a8fa03d3b1fc3f358e835a7748490c8bd16'
def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
checkout_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
original=RECIPE/'conf/train_gan_default.yaml';c=yaml.safe_load(original.read_text())
assert hashlib.sha256(original.read_bytes()).hexdigest()=='c0f857709450d8fa4c9a055a7c5ebd95a93bc0033cae8ee29742259256932ced','Official baseline YAML changed; review before running'
assert c['tokenizer_conf']['freeze_epochs']==30 and c['max_epoch']==90
assert c['model_conf']['reconstruction_weight']==.1 and c['batch_bins']==16000000
assert c['reconstruction_objective_conf']['speaker_embed_dim']==192
assert c['optim_conf']['lr']==c['optim2_conf']['lr']==1e-4
assert c['scheduler_conf']['milestones']==c['scheduler2_conf']['milestones']==[30]
# Local copy of the SAME original WavLM-large checkpoint avoids downloads.
c['frontend_conf']['frontend_conf']={'upstream':'wavlm_local','path_or_url':str(B/'assets/wavlm_large.pt')}
c['frontend_conf']['download_dir']=str(R/'hub')
# The public default builds VCTK centroids; the paper instead uses Libri30h.
c['quantizer_conf']['centroid_path']=str(R/'initialization/km_2000.mdl')
(RECIPE/'conf/train_paper.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
db=RECIPE/'db.sh'
if db.is_symlink():db.unlink()
db.write_text('VCTK=@VCTK@\n')
local_path=RECIPE/'local/path.sh'
if local_path.is_symlink():local_path.unlink()
local_path.write_text('source @ROOT@/env.sh\n')
atomic(R/'PROVENANCE.json',{'upstream':'https://github.com/ondatk68/espnet/tree/feature/speech-tokenizer',
 'upstream_base_commit':COMMIT,'checkout_commit':checkout_commit,'paper':'https://arxiv.org/abs/2601.19781','original_yaml_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
 'training':'VCTK; 0.9/1.0/1.1 speed perturbation; official speaker-closed split; BPE5000',
 'initialization':'New seeded 30h subset of LibriSpeech train-clean-100; original WavLM-large layer21; official sklearn MiniBatchKMeans K=2000',
 'speaker_encoder':'speechbrain/spkrec-ecapa-voxceleb, 192 dimensions, per-utterance; paper choice overrides template ESPnet default',
 'schedule':{'frozen_epochs':30,'frozen_lr':1e-4,'joint_epochs':60,'joint_lr':1e-5,'alpha':.1,'gpus_official_run_sh':4},
 'unknowns':['Paper does not specify VCTK release or utterance split; shared VCTK0.80 supported by author recipe is used.',
             'Original 30h initialization utterance list and random seed are not public; deterministic seed0 subset will be archived.',
             'Software versions/hardware may differ; runtime audit is archived.'],
 'old_msp_checkpoint_used':False,'prosody_auxiliary_losses':False,'batch_consistency_patch_used':False,
 'evaluation':'Tokenizer training is distinct from paper Table3: separately train unit HiFiGAN on LJSpeech and evaluate WER with Whisper-large-v3, MCD, F0 RMSE, UTMOS.'})
print('CONFIGURED_PAPER_RECIPE',flush=True)
