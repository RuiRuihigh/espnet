#!/usr/bin/env bash
#SBATCH --job-name=opt1.3b_vocoder_new20
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_opt1.3B/log/myrun_vocoder_new20.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_opt1.3B/log/myrun_vocoder_new20.%j.err

# Stage 10 的beam5解码(myrun_tts_beam5_new20.sh)已经给这20句生成了
# score_tts/hyp.tok。这里跑vocoder+VERSA,把最短的几句(51/56/58/65/68步)
# 的CER/WER跟步数对照一下,看是不是也有premature-EOS导致漏字的问题(跟原来
# 那8句同样的诊断思路)。
set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_opt1.3B

VOCODER_CKPT=/export/fs06/mliang17/phonologicaltokenizer_modify/ParallelWaveGAN/egs/ljspeech/phon_voc1/exp/train_nodev_ljspeech_hifigan_phon.v1/checkpoint-400000steps.pkl
[ -f "${VOCODER_CKPT}" ] || { echo "vocoder checkpoint 不存在: ${VOCODER_CKPT}" >&2; exit 1; }

(
    unset PYTHONPATH
    export PATH="${PWD}/utils:${PATH}"
    # shellcheck disable=SC1091
    source /opt/slurm_files/software/conda25/2025-08-21/etc/profile.d/conda.sh
    conda activate parallelwavegan
    bash local/eval_tts_vocoder.sh "${VOCODER_CKPT}" text.tts.new20
)

echo
echo "==== avg_result.txt ===="
cat exp/lm_train_transformer_opt1.3b_en_bpe10000/decode_test_tts/text.tts.new20/vocoder_eval/checkpoint-400000steps/versa_eval/avg_result.txt
