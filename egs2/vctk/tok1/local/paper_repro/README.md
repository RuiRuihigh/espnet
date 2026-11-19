# Phonological Tokenizer paper reproduction

Reproduce the single-codebook tokenizer training from the
[Phonological Tokenizer paper](https://arxiv.org/abs/2601.19781), using the
[author's implementation](https://github.com/ondatk68/espnet/tree/feature/speech-tokenizer).
This directory packages the VCTK reproduction prepared on Delta, with paths and
Slurm settings configurable for another cluster. It does not contain the older
MSP pilot, auxiliary pitch/VUV heads, trained weights, or datasets.

**Validation status on 2026-11-19:** VCTK preparation, LibriSpeech feature extraction,
K-means, ECAPA extraction, and statistics completed on Delta. The one-GPU preflight
was still queued. Neither the preflight nor the 90-epoch training has passed yet;
there are no reproduction quality results. Portable environment installation and
multi-GPU execution must be validated on the destination cluster.

## Fixed experimental settings

| Component | Setting |
|---|---|
| Upstream base | `0aff6a8fa03d3b1fc3f358e835a7748490c8bd16` |
| Model configuration | Unchanged `conf/train_gan_default.yaml`; generated copy changes pretrained-file/cache and centroid paths only |
| SSL frontend | Original pretrained WavLM-Large, layer 21; not an already fine-tuned phonological checkpoint |
| Initialization corpus | New seed-0 subset of LibriSpeech `train-clean-100`, approximately 30 hours |
| Quantizer | One 2,000-entry codebook, differentiable K-means, approximately 50 tokens/s |
| K-means initialization | Author's `perform_kmeans.sh` / `learn_kmeans.py`, sklearn MiniBatchKMeans; not the old MSP custom Lloyd code |
| Tokenizer training corpus | VCTK, 16 kHz; speed factors 0.9, 1.0, 1.1 |
| Text supervision | Unigram BPE, 5,000 entries |
| Speaker conditioning | Frozen `speechbrain/spkrec-ecapa-voxceleb`, 192 dimensions, per utterance |
| Epochs 1–30 | Freeze SSL encoder and centroids; train ASR, HiFi-GAN and discriminator, learning rate 1e-4 |
| Epochs 31–90 | Unfreeze SSL encoder and centroids, learning rate 1e-5; speaker encoder remains frozen |
| Optimizers | Two Adam optimizers, betas (0.5, 0.9), weight decay 0; milestone 30, gamma 0.1; official optimizer reset on unfreeze |
| Batching | Four GPUs, global `batch_bins=16000000`, `batch_type=numel`, `accum_grad=1` |
| Duration | Full dataset epochs; `num_iters_per_epoch=None`, no 100-update cap |
| Other settings | Seed 0, FP32 (`use_amp=false`), best validation generator loss, keep best 3 |

The generator objective is `0.9 * ASR + 0.1 * reconstruction`.
ASR uses CTC weight 0.3 and attention weight 0.7, with label smoothing 0.1.
Reconstruction combines mel loss (45), feature matching (2), and generator
adversarial loss (1); the discriminator has its own objective and optimizer.
Do not add relative-pitch, V/UV, energy, or style losses to this reproduction.
The full architecture and loss settings remain in the original YAML, including
the 12-block E-Branchformer, 6-block Transformer decoder, HiFi-GAN, and quantizer
temperature schedule. The original YAML SHA256 is
`c0f857709450d8fa4c9a055a7c5ebd95a93bc0033cae8ee29742259256932ced`.

## Data and reproducibility boundaries

LibriSpeech is used **only for centroid initialization**; VCTK supplies the
tokenizer's paired speech/text training. Do not merge LibriSpeech into VCTK training.

The Delta run used legacy VCTK 0.80 (`wav48` plus `txt`). The pinned recipe also
supports 0.92 (`wav48_silence_trimmed`, mic1), but changing release is a recorded
experimental difference, not an exact repeat of the Delta input. The paper does
not identify the release or the original 30-hour LibriSpeech utterance list.

The official split is **speaker-closed**: for each speaker, the final five sorted
utterances are test, the preceding five are validation, and the remainder train.
Speakers overlap across these partitions; utterance IDs do not. This is different
from the speaker-disjoint MSP probing setup.

Observed post-preparation counts on Delta:

| Split | Utterances | Audio hours |
|---|---:|---:|
| VCTK train `tr_no_dev` | 43,004 | 42.871581 |
| VCTK speed-augmented train `tr_no_dev_sp` | 129,011 | 129.474886 |
| VCTK validation `dev` | 540 | 0.515973 |
| VCTK test `eval1` | 540 | 0.498717 |
| LibriSpeech initialization | 8,499 | 30.001646 |

`libri30h_ids.txt` archives our seed-0 selection, not the authors' unpublished IDs.
`libri_subset.py` regenerates a manifest with paths, samples and transcripts from
a complete `train-clean-100` directory. Compare its IDs with this file. Absolute
paths change the manifest hash, so compare utterance IDs and audio lengths too.
Obtain the corpora from their official providers or an authorized local copy;
no audio is distributed here. MSP is not required for this reproduction.

## Create a workspace on the new cluster

Use a dedicated checkout for each experiment; configuration writes recipe-local
`db.sh`, `local/path.sh`, and `conf/train_paper.yaml`. It does not modify models,
loss implementations, or trainers. The workspace contains a symlink to this checkout.

From `egs2/vctk/tok1`, substitute your actual paths, Slurm accounts and partitions:

```bash
python3 local/paper_repro/init_workspace.py \
  --root /scratch/USER/phonological_paper \
  --base /scratch/USER/phonological_assets \
  --vctk /datasets/VCTK-Corpus \
  --libri /datasets/LibriSpeech/train-clean-100 \
  --conda-sh /opt/miniforge3/etc/profile.d/conda.sh \
  --cpu-account YOUR_CPU_ACCOUNT --gpu-account YOUR_GPU_ACCOUNT \
  --cpu-partition cpu --gpu-partitions gpu \
  --preflight-partition gpu
```

The renderer checks syntax, refuses a nonempty workspace and never installs or
submits anything. Paths containing spaces or shell metacharacters are rejected.
`--base` holds environments, assets and caches. Supply partitions whose **every
eligible GPU** has enough memory; the old Delta model-name routing is not portable.
The reference preflight was requested on an 80 GB class A100. There is no measured
minimum yet. Do not assume that any 16/24/40 GB GPU can run the unchanged batch.

For non-Slurm clusters, port only the submission wrappers. Preserve four-rank
batch sharding, optimizers, schedule and model settings. No such launcher is
implemented here. Cluster migration does not justify shrinking the scientific batch.

## Build environments and fetch the pretrained frontend

On an allocated CPU node, with network access where required:

```bash
bash local/paper_repro/bootstrap_environment.sh /scratch/USER/phonological_paper
bash local/paper_repro/download_wavlm.sh /scratch/USER/phonological_paper
cd /scratch/USER/phonological_paper
source env.sh
```

Environment snapshots are from the actual Delta environments, not an upstream
supported lockfile. Main Python was 3.12, PyTorch/torchaudio 2.10.0+cu126, s3prl
0.4.18 from ESPnet commit `572af70a3e47c23292b6a3e93fa13dcf1e281111`.
The separate audio environment uses PyTorch/torchaudio 2.5.1+cu124, SpeechBrain
1.0.3, NumPy 1.26.4, huggingface-hub 0.36.2. The bootstrap refuses to overwrite
existing main/audio environments and finishes with `pip check`.

Required fixes are already included in the dependency snapshots: main environment
`resampy==0.4.3`; audio environment `kaldiio==2.18.1`, `librosa==0.11.0`,
`typeguard==2.13.3`. SoX, bc, libsndfile, bash, git, curl and Perl are needed.
The reference recipe used bundled `run.pl` inside Slurm allocations, not nested
`sbatch` calls. A full Kaldi build was not present on Delta.

Fresh installation has not been exercised on the destination cluster. If the
package resolver or CUDA driver is incompatible, diagnose and record the minimal
environment change. Do not silently upgrade the source branch or change objectives.
The WavLM download uses a fixed Hugging Face revision and verifies SHA256 before
renaming the file. Pre-cache ECAPA on a network-enabled node if GPU nodes lack
network access; its model ID is fixed, but an upstream snapshot revision was not
originally pinned, so record the revision used on the new cluster.

## Preprocessing and preflight

Run `setup.sh` under an allocated CPU job first. It writes configuration and
`SETUP_OK.json` after importing the official task from this checkout. For example,
adapt the Slurm flags below to the destination limits:

```bash
SETUP_JOB=$(sbatch --parsable --account=YOUR_CPU_ACCOUNT --partition=cpu \
  --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:45:00 \
  --chdir="$PAPER_ROOT" --output="$PAPER_ROOT/logs/setup_%j.log" "$PAPER_ROOT/setup.sh")
SETUP_JOB=${SETUP_JOB%%;*}
"$PAPER_MAIN/bin/python" submit_pipeline.py --setup-job "$SETUP_JOB"
```

This explicitly submits the prerequisite DAG:

```text
setup -> VCTK data stages 1–5 -> ECAPA stage 6 --+
setup -> Libri30h -> WavLM features -> K-means -+-> statistics stage 8 -> GPU preflight
```

Each stage can also be invoked as `bash stage.sh NAME` within a suitably allocated
job. Names: `data`, `subset`, `features`, `kmeans`, `speakers`, `stats`, `preflight`.
Completed stages write markers under `stages/`. Existing completed stages are
skipped. Interrupted sub-stages may rerun; this is not per-file resumability for
every official utility. Inspect artifacts before manually accepting a marker.

Reference resource requests, **not performance guarantees**:

| Stage | GPUs | CPUs | Host RAM | Time limit |
|---|---:|---:|---:|---:|
| Data | 0 | 4 | 8 GB | 3 h |
| Libri subset | 0 | 2 | 4 GB | 45 min |
| SSL features | 1 | 2 | 16 GB | 2 h |
| K-means | 0 | 16 | 128 GB | 8 h |
| ECAPA | 1 | 2 | 8 GB | 4 h |
| Statistics | 0 | 4 | 32 GB | 3 h |
| Preflight | 1 | 4 | 32 GB | 90 min |

Keep all walltimes explicit. Adjust resource-only settings using measurements;
do not submit duplicate copies to multiple queues. On Delta, clustering took
48m30s, ECAPA 2h59m57s and statistics 36m42s. These are observations, not estimates
of speed on another filesystem or GPU.

Before training, verify:

- `DATA_READY.json`: ID alignment, disjoint utterance partitions, counts, BPE size.
- `initialization/km_2000.mdl`: 2,000 finite centroids with the expected WavLM dimension.
- ECAPA `.scp` entries: all train/dev/test IDs, finite vectors of dimension 192.
- `stats` shape files and `stages/stats.done`: all required inputs present.
- `preflight/COMPLETE.json`: real frozen and unfrozen generator/discriminator
  backward steps pass; frozen encoder/centroids receive no gradients, joint mode
  receives gradients; finite losses, complete sampler coverage, memory margin.

The preflight evaluates rank 0 of the official four-way batch sharding on one GPU.
It is **not a four-GPU DDP test** and does not prove the slowest/largest rank fits.
Inspect all four ranks' longest/padded batches and validate NCCL plus real DDP
startup before trusting the full run. The recipe's `NCCL_SOCKET_IFNAME` default
may need adaptation to the new interconnect. Preflight scratch weights are discarded.

## Start and resume official training

Unlike the original Delta wrapper, this portable preflight does **not** auto-submit
training. Once preflight results, eligible GPU memory and destination limits are
reviewed, submit the first epoch explicitly:

```bash
source env.sh
"$PAPER_MAIN/bin/python" train_chain.py submit 1
```

This starts a **chain through epoch 90**: each completed full epoch verifies its
checkpoint and submits the next. Requests use four GPUs on one node, 16 CPUs,
host memory derived from preflight, and an explicit estimated epoch walltime.
Inspect the current 48-hour maximum and estimator in `train_chain.py` against the
new partition's actual limits before starting. It does not shorten epochs to fit.
Changing from one epoch per job to multiple epochs per job is a scheduling change
that still requires faithful resume and the same 30/60 schedule.

Important files: `train/checkpoint.pth`, `train/PROGRESS.json`,
`train/reporter.json`, `train_jobs.json`, `preflight/COMPLETE.json`, `PROVENANCE.json`.
Checkpoint saving uses an atomic temporary-file rename; model math is unchanged.
Resume restores the official reporter, both optimizers and schedulers; the
official reset at unfreezing must remain intact. Do not load an old MSP checkpoint.

If a job fails, inspect `sacct`, the log, checkpoint epoch and progress marker
before retrying. A recorded target in `train_jobs.json` blocks duplicate submission;
there is no automatic failed-job retry. Reconcile only the failed entry after
confirming no active job exists. If a checkpoint was saved but progress writing
failed, verify the checkpoint rather than rerunning an already completed epoch.
Creating `STOPPED_BY_USER.json` blocks future wrapper submissions; it does not
cancel an already running job.

## Evaluation scope

Training losses and VCTK ASR CER are diagnostics, not a reproduction of the paper's
generation table. The paper evaluates tokenizers with a separately trained
LJSpeech unit HiFi-GAN and metrics including Whisper-large-v3 WER, MCD, F0 RMSE,
and UTMOS. That downstream training/evaluation pipeline is **not implemented in
this package**. It must be separately specified and matched before claiming
paper-level reproduction. No MSP A/B/C result is evidence for this official run.

## Instructions for the receiving agent

1. Inspect this README, templates and upstream YAML before launching anything.
2. Inventory available corpora, GPUs, driver, disk quota, account and partition limits.
3. Create an isolated workspace and record all adaptations in `PORTING.json` or
   a companion change log. Preserve the scientific settings above.
4. Install/import-test environments, verify the model checksum and run prerequisite stages.
5. Compare counts/IDs and inspect errors; do not mark unfinished stages complete.
6. Pass GPU preflight and verify actual four-rank training startup, then train 90 full epochs.
7. Report job IDs, epoch/update progress, phase, checkpoint, finite losses, and
   remaining limitations. Never describe an unrun preflight or queued training as passed.
