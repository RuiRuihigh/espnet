#!/usr/bin/env bash
#SBATCH --job-name=v1k2000_tts100
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal/log/myrun_tts_subset100.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal/log/myrun_tts_subset100.%j.err

set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal

. ./path.sh
. ./cmd.sh

# 跟隔壁 lm1_k1000_Bal/myrun_tts_subset100.sh 同一套逻辑、同一份 100 句清单
# (local/tts_eval_subset/keys.txt,三个 recipe 内容完全一致,md5校验过)。
# 输出落在 exp/.../decode_test_tts/text.tts.subset100/,跟全量的
# text.tts/(Stage10之前被取消、目前没有完整hyp.tok)是两个独立目录,互不影响。
##############################################################################
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1
./run.sh --stage 10 --stop_stage 10 --ngpu 1 --nclusters 2000 \
    --lm_config conf/tuning/train_transformer_opt350.yaml \
    --inference_lm valid.acc.best.pth \
    --lm_test_text_tts dump/raw/test/text.tts.subset100

##############################################################################
VOCODER_CKPT=/export/fs06/mliang17/phonologicaltokenizer_modify/ParallelWaveGAN/egs/ljspeech/hubert_voc_v1Balk2000/exp/train_nodev_ljspeech_hifigan_hubert.v1/checkpoint-400000steps.pkl

(
    unset PYTHONPATH
    export PATH="${PWD}/utils:${PATH}"
    # shellcheck disable=SC1091
    source /opt/slurm_files/software/conda25/2025-08-21/etc/profile.d/conda.sh
    conda activate parallelwavegan
    bash local/eval_tts_vocoder.sh "${VOCODER_CKPT}" text.tts.subset100
)

echo "======================================================================"
echo "100句子集跑完。结果:"
echo "  exp/lm_train_transformer_opt350_en_bpe10000/decode_test_tts/text.tts.subset100/vocoder_eval/*/versa_eval/avg_result.txt"
echo "======================================================================"
