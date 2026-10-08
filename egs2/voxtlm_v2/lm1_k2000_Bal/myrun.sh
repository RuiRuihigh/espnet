#!/usr/bin/env bash
#SBATCH --job-name=lm1_k2000_Bal
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal/log/myrun.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal/log/myrun.%j.err

set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal

. ./path.sh
. ./cmd.sh

# Stage 3's WavLM-large forward has no chunking, so a single ~120s utterance
# needs a ~2GB contiguous block -- borderline on the 10.9GB gpu nodes.
# First run without this: 14/16 speechlm/train shards OOM'd with ~1.5GB
# "reserved but unallocated" (fragmentation). Same var lm1's voxt_v2 run
# used (speechlm/train 16/16 succeeded there). Inherited by the array jobs
# slurm.pl submits (sbatch --export=ALL).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# This recipe reuses ../../voxtlm_v1/lm1_k1000_Bal's Stage 1/2 output as-is
# (data/ and dump/audio_raw are both symlinked straight there, not copied):
# discretization is the ONLY thing that differs between k=1000 self-trained
# kmeans and this recipe's fixed Phonological Tokenizer codebook, so the raw
# audio/text task splits (asr/tts/speechlm/textlm, D_Bal-subsampled) and the
# formatted wav.scp are identical either way. Do NOT run Stage 1/2 here --
# that would just redo (or worse, diverge from) lm1_k1000_Bal's own copy.
# Wait for lm1_k1000_Bal's Stage 1/2 to actually finish before Stage 3 below
# can do anything (data/ only has "librispeech" so far as of this writing).

##############################################################################
# Stage 3: discretize speech with the Phonological Tokenizer
# (WavLM-large layer 21, fixed released 2000-entry codebook) for asr/tts/
# speechlm. Unlike lm1_k1000_Bal's Stage 3, there's no kmeans training step
# (no learn_kmeans, no --cpu_cmd memory dance) and no A100 involved --
# feature extraction with the released checkpoint (~1.3GB WavLM-large) fits
# comfortably on the regular gpu partition (cmd.sh's default cuda_cmd),
# same as lm1_k1000_Bal's own Stage 3 hubert_base feature extraction.
# Idempotent per task (train/dev/test together): a task whose discrete units
# were already produced from the current released checkpoint/centroids
# (fingerprinted via md5sum, see lm.sh) is skipped automatically on rerun.
##############################################################################
# [Done: asr/speechlm/tts all labeled (speechlm/train 16/16 shards OK with
#  PYTORCH_CUDA_ALLOC_CONF above). Not rerunning.]
# ./run.sh --stage 3 --stop_stage 3

##############################################################################
# Stage 4: assemble the multitask text; Stage 5: train BPE on Stage 4's
# output (sentencepiece unigram, nbpe=10000, special tokens kept out of
# merges via data/nlsyms.txt). Both CPU-only, no GPU needed.
##############################################################################
# [Done, not rerunning -- BPE alone takes ~1h.]
# ./run.sh --stage 4 --stop_stage 5

##############################################################################
# Stage 6: collect stats (text_shape.bpe etc., used by Stage 7's batch
# sampler).
##############################################################################
# [Done, not rerunning.]
# ./run.sh --stage 6 --stop_stage 6

# [Same fix lm1_k1000_Bal's README documents]: discrete speech-unit
# sequences have no natural sentence boundary, and a handful of extremely
# long ones make Stage 7 OOM on the A100 (self-attention memory is O(L^2);
# batch_bins only buckets by linear token count). Filter out length>1500,
# backing up the original first (idempotent -- skips if already done).
# [Already applied: 1,279,026 -> 1,274,524; not touching the file again.]
# F=exp/lm_stats_en_bpe10000/train/text_shape.bpe
# [ -e "$F.orig" ] || cp "$F" "$F.orig"
# awk '{split($2,a,","); if (a[1]<=1500) print}' "$F.orig" > "$F"

##############################################################################
# Stage 7: train the language model, OPT-350M init (the paper's reported
# main-result config). This is the one step that genuinely needs the A100 --
# cuda_cmd stays on the regular gpu partition for everything above, and is
# only switched here (via sed on cmd.sh, since lm.sh re-sources it
# internally -- a plain `export` wouldn't stick), then switched back on exit
# via trap regardless of success/failure/interrupt.
#
# HF_HOME points at the shared hf_cache/ (facebook/opt-350m already cached
# there), with HF_HUB_OFFLINE=1 so it loads even though compute nodes have
# no internet. This is real training and will run for a long time -- watch
# train.log's train/valid acc curve and stop manually when it overfits
# (no early stopping is configured).
##############################################################################
# [Stage 7 done: stopped after resume at 21epoch (valid.acc.best.pth stuck
#  at 16epoch since epoch 16, valid_acc 0.313 -> plateaued/dipped to 0.311
#  for 5 straight epochs while train_acc kept climbing -- same overfit point
#  voxt_v2's earlier run of this exact data/tokenizer hit. Using 16epoch.pth
#  (= valid.acc.best.pth already) as the final checkpoint.]
# trap 'sed -i "s#export cuda_cmd=\"slurm.pl --config conf/slurm_clsp_a100.conf\"#export cuda_cmd=\"slurm.pl\"#" cmd.sh' EXIT
# sed -i 's#export cuda_cmd="slurm.pl"#export cuda_cmd="slurm.pl --config conf/slurm_clsp_a100.conf"#' cmd.sh
# . ./cmd.sh
# export HF_HOME="${PWD}/hf_cache"
# export HF_HUB_OFFLINE=1
# ./run.sh --stage 7 --stop_stage 7 --ngpu 1 --lm_config conf/tuning/train_transformer_opt350.yaml

##############################################################################
# Stage 8 (perplexity on textlm/speechlm test) + Stage 9-10 (ASR/TTS
# decoding), using the best (16epoch) checkpoint. None of this needs the
# A100: Stage 8 is a single forward pass; Stage 9-10's beam search is
# heavier but still just inference at batch_size=1 -- cmd.sh's default
# cuda_cmd (regular gpu partition) is already active from Stage 3-6 above,
# same as lm1_k1000_Bal's own Stage 8-10 run.
##############################################################################
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1
./run.sh --stage 8 --stop_stage 10 --ngpu 1 --lm_config conf/tuning/train_transformer_opt350.yaml --inference_lm valid.acc.best.pth
