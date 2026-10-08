#!/usr/bin/env bash
#SBATCH --job-name=k2000_tts_voc
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal/log/myrun_vocoder.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal/log/myrun_vocoder.%j.err

set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal

# parallelwavegan env, not ./path.sh's espnet env -- see local/
# eval_tts_vocoder.sh's own header comment for why.
source /opt/slurm_files/software/conda25/2025-08-21/etc/profile.d/conda.sh
conda activate parallelwavegan

##############################################################################
# TTS -> audio -> objective scores (CER via whisper_wer, MOS-style
# naturalness via pseudo_mos/utmos, + mcd_f0/discrete_speech/speaker),
# using the user-provided DiscreteSymbolHiFiGANGenerator vocoder checkpoint.
# Requires Stage 10's hyp.tok to already exist (lm1_k2000_Bal's own
# myrun.sh); local/eval_tts_vocoder.sh checks for it and exits with a clear
# error if it's missing rather than hanging or failing obscurely.
##############################################################################
checkpoint=/export/fs06/mliang17/phonologicaltokenizer_modify/ParallelWaveGAN/egs/ljspeech/phon_voc1/exp/train_nodev_ljspeech_hifigan_phon.v1/checkpoint-400000steps.pkl
bash local/eval_tts_vocoder.sh "${checkpoint}"
