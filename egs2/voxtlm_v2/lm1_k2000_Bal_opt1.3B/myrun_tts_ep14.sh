#!/usr/bin/env bash
#SBATCH --job-name=opt1.3b_tts_ep14
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_opt1.3B/log/myrun_tts_ep14.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_opt1.3B/log/myrun_tts_ep14.%j.err

# 用14epoch.pth跑TTS测试: 同一批rand100(100句),beam5,penalty=0,maxlen=1200,
# 跟7epoch的plain beam5基线(WER 9.96%/CER 8.14%)设置完全一致,直接可比。
# 解码在普通gpu分区,VERSA打分由 local/eval_tts_vocoder.sh 自己切到1张A100。
set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_opt1.3B
. ./path.sh
. ./cmd.sh
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1

VOCODER_CKPT=/export/fs06/mliang17/phonologicaltokenizer_modify/ParallelWaveGAN/egs/ljspeech/phon_voc1/exp/train_nodev_ljspeech_hifigan_phon.v1/checkpoint-400000steps.pkl
[ -f "${VOCODER_CKPT}" ] || { echo "vocoder checkpoint 不存在: ${VOCODER_CKPT}" >&2; exit 1; }

PROMPT=exp/penalty_sweep/text.tts.rand100_ep14
TAG=$(basename ${PROMPT})

./run.sh --stage 10 --stop_stage 10 --ngpu 1 --gpu_inference true --inference_nj 8 \
    --lm_config conf/tuning/train_transformer_opt1.3b.yaml \
    --inference_lm 14epoch.pth \
    --lm_inference_tts_config conf/decode_lm_tts_maxlen1200.yaml \
    --lm_test_text_tts "${PROMPT}"

(
    unset PYTHONPATH
    export PATH="${PWD}/utils:${PATH}"
    # shellcheck disable=SC1091
    source /opt/slurm_files/software/conda25/2025-08-21/etc/profile.d/conda.sh
    conda activate parallelwavegan
    bash local/eval_tts_vocoder.sh "${VOCODER_CKPT}" "${TAG}"
)

echo "==== 14epoch TTS (rand100, beam5) avg_result ===="
cat exp/lm_train_transformer_opt1.3b_en_bpe10000/decode_test_tts/${TAG}/vocoder_eval/checkpoint-400000steps/versa_eval/avg_result.txt
