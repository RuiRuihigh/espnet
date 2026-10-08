#!/usr/bin/env bash
#SBATCH --job-name=sqa_v1_forget_resume3
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_sqa/log/myrun_sqa_v1_forgetting_resume3.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_sqa/log/myrun_sqa_v1_forgetting_resume3.%j.err

# resume2(job 1809021)的三级回退解码 + vocoder decode 都已经成功(102个wav在
# exp/sqa_v1_ft/decode_test_tts/text.tts.fb_final/vocoder_eval/checkpoint-400000steps/wav/),
# 只有 Step3 VERSA 打分挂了,两个原因都已经在 local/eval_tts_vocoder.sh 里修好:
#   1) sed+trap 切 A100 的正则对不上本目录 cmd.sh 的 ${SQA_CUDA_CMD:-slurm.pl}
#      这一行,静默失效,VERSA 实际跑在普通 GPU(10.9GB)上 OOM
#   2) data/test/speech/tts/wav.scp 里是相对路径(downloads/LibriTTS/...),在
#      本目录下解析不到 -- 已经软链接 downloads/LibriTTS 过来解决
# local/summarize_fallback.py 还要拿"纯 beam5、不回退"(text.tts.rand100)的
# VERSA 结果去跟 fb_final 对比,之前只跑过 fb_final,这里补上 rand100 那份
# (复用 Step1 已经生成的 100 条 wav 对应的 hyp.tok,cost 很低,100句)。
# 这个脚本只重跑 eval_tts_vocoder.sh(它自己的 Step1/2 很快,几秒钟内完成,不用
# 额外跳过),不重跑前面的解码。
set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_sqa
. ./path.sh
. ./cmd.sh
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1

VOCODER_CKPT=/export/fs06/mliang17/phonologicaltokenizer_modify/ParallelWaveGAN/egs/ljspeech/phon_voc1/exp/train_nodev_ljspeech_hifigan_phon.v1/checkpoint-400000steps.pkl
(
    unset PYTHONPATH
    export PATH="${PWD}/utils:${PATH}"
    # shellcheck disable=SC1091
    source /opt/slurm_files/software/conda25/2025-08-21/etc/profile.d/conda.sh
    conda activate parallelwavegan
    bash local/eval_tts_vocoder.sh "${VOCODER_CKPT}" text.tts.rand100
    bash local/eval_tts_vocoder.sh "${VOCODER_CKPT}" text.tts.fb_final
)
python3 local/summarize_fallback.py
