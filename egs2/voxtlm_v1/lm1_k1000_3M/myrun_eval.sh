#!/usr/bin/env bash
#SBATCH --job-name=v1_k1k3M_eval
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_3M/log/myrun_eval.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_3M/log/myrun_eval.%j.err

set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_3M

. ./path.sh
. ./cmd.sh
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1

# checkpoint 训练结束后按 train.log 的 valid loss/acc 曲线选定,默认 valid.acc.best.pth
# (local/slm21_common.sh 和 local/eval_tts_vocoder.sh 里的路径也写的是这个,改之前要同步改)。
CKPT=${1:-valid.acc.best.pth}

# 与 lm1_k1000_Bal 的评测一致(同样的 km_1000 官方模型、同样的 hubert_voc1 声码器):
#   1) Stage 8 PPL  2) sLM21 sWUGGY/sBLIMP  3) Stage 9-10 ASR 全量 + TTS 100 句子集
#   4) vocoder + VERSA
# sLM21 的 km 是官方外部路径,不依赖本 recipe 的 exp/kmeans(本 recipe 没有)。

##############################################################################
# 1) Stage 8: PPL
##############################################################################
./run.sh --stage 8 --stop_stage 8 --ngpu 1 --nclusters 1000 \
    --lm_config conf/tuning/train_transformer_opt350.yaml \
    --inference_lm "${CKPT}"

##############################################################################
# 2) sLM21
##############################################################################
bash local/eval_slm21_text.sh
bash local/eval_slm21_speech.sh
bash local/eval_slm21_score.sh

##############################################################################
# 3) Stage 9 ASR 全量 + Stage 10 TTS(100 句子集)
##############################################################################
./run.sh --stage 9 --stop_stage 10 --ngpu 1 --nclusters 1000 \
    --lm_config conf/tuning/train_transformer_opt350.yaml \
    --inference_lm "${CKPT}" \
    --lm_test_text_tts dump/raw/test/text.tts.subset100

##############################################################################
# 4) TTS 客观评分(hubert_voc1 声码器,VERSA 在 A100 上打分)
##############################################################################
VOCODER_CKPT=/export/fs06/mliang17/phonologicaltokenizer_modify/ParallelWaveGAN/egs/ljspeech/hubert_voc1/exp/train_nodev_ljspeech_hifigan_hubert.v1/checkpoint-400000steps.pkl
(
    unset PYTHONPATH
    export PATH="${PWD}/utils:${PATH}"
    # shellcheck disable=SC1091
    source /opt/slurm_files/software/conda25/2025-08-21/etc/profile.d/conda.sh
    conda activate parallelwavegan
    bash local/eval_tts_vocoder.sh "${VOCODER_CKPT}" text.tts.subset100
)

##############################################################################
echo "======================================================================"
echo "全部评测跑完。结果位置:"
echo "  PPL      : exp/lm_train_transformer_opt350_en_bpe10000/perplexity_test_*/"
echo "  ASR WER  : exp/lm_train_transformer_opt350_en_bpe10000/decode_test_asr/decode_lm_asr/score_wer/result.txt"
echo "  sLM21    : exp/slm21_eval/submission/ (数字见本日志 eval_slm21_score.sh 输出)"
echo "  TTS 客观分: exp/lm_train_transformer_opt350_en_bpe10000/decode_test_tts/text.tts.subset100/vocoder_eval/*/versa_eval/avg_result.txt"
echo "======================================================================"
