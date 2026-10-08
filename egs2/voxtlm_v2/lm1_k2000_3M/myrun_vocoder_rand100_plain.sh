#!/usr/bin/env bash
#SBATCH --job-name=3M_vocoder_rand100
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_3M/log/myrun_vocoder_rand100_plain.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_3M/log/myrun_vocoder_rand100_plain.%j.err

# myrun_eval_full.sh 跑完三级回退后调 summarize_fallback.py 时报错退出了:
# 它要对比"纯beam5(未回退)"的VERSA结果,但流程本身没单独给 text.tts.rand100
# 跑过vocoder+VERSA(只给 fb_final 跑了)。hyp.tok已经有了(Stage10解码时自动
# 生成),这里补上vocoder+VERSA这一步,再重跑一次summarize_fallback.py。
set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_3M

VOCODER_CKPT=/export/fs06/mliang17/phonologicaltokenizer_modify/ParallelWaveGAN/egs/ljspeech/phon_voc1/exp/train_nodev_ljspeech_hifigan_phon.v1/checkpoint-400000steps.pkl
[ -f "${VOCODER_CKPT}" ] || { echo "vocoder checkpoint 不存在: ${VOCODER_CKPT}" >&2; exit 1; }

(
    unset PYTHONPATH
    export PATH="${PWD}/utils:${PATH}"
    # shellcheck disable=SC1091
    source /opt/slurm_files/software/conda25/2025-08-21/etc/profile.d/conda.sh
    conda activate parallelwavegan
    bash local/eval_tts_vocoder.sh "${VOCODER_CKPT}" text.tts.rand100
)
python3 local/summarize_fallback.py
