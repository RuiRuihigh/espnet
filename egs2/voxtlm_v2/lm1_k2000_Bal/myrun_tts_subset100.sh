#!/usr/bin/env bash
#SBATCH --job-name=v2k2000_tts100
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal/log/myrun_tts_subset100.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal/log/myrun_tts_subset100.%j.err

set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal

. ./path.sh
. ./cmd.sh

# 跟 voxtlm_v1 两个 recipe 下的 myrun_tts_subset100.sh 同一套逻辑、同一份
# 100 句清单(local/tts_eval_subset/keys.txt,md5 三边一致)。这边全量
# text.tts/ 已经跑完(9957条 hyp.tok + wav + avg_result.txt 都在),这次
# 输出落在独立的 decode_test_tts/text.tts.subset100/,不会碰、不会覆盖
# 已有的全量结果。
# run.sh 自己内部 nclusters=2000 是固定值(音系学 tokenizer 只有这一个配置,
# 不像 v1 那样可选),不用像 v1 k2000 那样额外传 --nclusters。
##############################################################################
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1
./run.sh --stage 10 --stop_stage 10 --ngpu 1 \
    --lm_config conf/tuning/train_transformer_opt350.yaml \
    --inference_lm valid.acc.best.pth \
    --lm_test_text_tts dump/raw/test/text.tts.subset100

##############################################################################
VOCODER_CKPT=/export/fs06/mliang17/phonologicaltokenizer_modify/ParallelWaveGAN/egs/ljspeech/phon_voc1/exp/train_nodev_ljspeech_hifigan_phon.v1/checkpoint-400000steps.pkl

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
