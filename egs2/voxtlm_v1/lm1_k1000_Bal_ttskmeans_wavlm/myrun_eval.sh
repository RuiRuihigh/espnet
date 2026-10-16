#!/usr/bin/env bash
#SBATCH --job-name=v1_k1k_wavlm_eval
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_ttskmeans_wavlm/log/myrun_eval.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_ttskmeans_wavlm/log/myrun_eval.%j.err

set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_ttskmeans_wavlm

. ./path.sh
. ./cmd.sh
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1

# 2026-10-12 第 19 epoch 中途手动停止训练(共 18 个完整 epoch);valid.acc.best.pth -> 16epoch.pth。
# 曲线:valid acc 从第 10 epoch 起停在 0.333(17/18 epoch 降到 0.332),valid loss 在 11/12 epoch
# 最低(2.501)之后一路回升到 2.535(18 epoch),train acc 还在涨(0.461 -> 0.485),已经过拟合。
# 按其它 recipe 的惯例用 valid.acc.best.pth(16epoch,valid loss 2.508,离最低点很近)。
# checkpoint 默认 valid.acc.best.pth
# (local/slm21_common.sh 和 local/eval_tts_vocoder.sh 里的路径也写的是这个,改之前要同步改)。
CKPT=${1:-valid.acc.best.pth}

# 由 lm1_k1000_3M/myrun_eval.sh 拷贝而来,口径一致:
#   1) Stage 8 PPL  2) sLM21 sWUGGY/sBLIMP  3) Stage 9 ASR 全量(8 路并行,普通 gpu 分区)
# TTS 不在这里跑,统一用 myrun_tts_fallback.sh 的 rand100 三级回退(fb_final)。
# 本脚本不会改 cmd.sh,可以和 fallback 同时跑(fallback 的 VERSA 用单独的 A100 命令,也不改 cmd.sh)。
# sLM21 用本 recipe 自己的 exp/kmeans/wavlm_large_21_1000clusters(WavLM-large 第21层),
# local/eval_slm21_speech.sh 已经改好;必须用本实验自己的 kmeans,单元编号才和 LM 对得上。
# 提交前确认 cmd.sh 第 84 行 cuda_cmd 是 "slurm.pl"(普通 gpu),否则 ASR 会去要 8 张 A100。

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
# 3) Stage 9 ASR 全量
##############################################################################
./run.sh --stage 9 --stop_stage 9 --ngpu 1 --nclusters 1000 \
    --lm_config conf/tuning/train_transformer_opt350.yaml \
    --inference_lm "${CKPT}"

##############################################################################
echo "======================================================================"
echo "全部评测跑完。结果位置:"
echo "  PPL      : exp/lm_train_transformer_opt350_en_bpe10000/perplexity_test_*/"
echo "  ASR WER  : exp/lm_train_transformer_opt350_en_bpe10000/decode_test_asr/decode_lm_asr/score_wer/result.txt"
echo "  sLM21    : exp/slm21_eval/submission/ (数字见本日志 eval_slm21_score.sh 输出)"
echo "  TTS      : 不在本脚本,见 myrun_tts_fallback.sh(decode_test_tts/text.tts.fb_final/)"
echo "======================================================================"
