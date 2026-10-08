#!/usr/bin/env bash
#SBATCH --job-name=opt1.3b_beam5_new20
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_opt1.3B/log/myrun_tts_beam5_new20.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_opt1.3B/log/myrun_tts_beam5_new20.%j.err

# 原来的8句"提前停"是从text.tts.rand100(100句)里筛出来的。这里换另外20句
# (exp/penalty_sweep/text.tts.new20,从全部9957句里排掉rand100用过的那100句
# 之后随机抽的,跟rand100完全不重叠),用同样的beam5配置
# (conf/decode_lm_tts_maxlen1200.yaml)解码,看premature-EOS的比例是不是能
# 在一个全新样本上复现(rand100上是8/100=8%)。
set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_opt1.3B
. ./path.sh
. ./cmd.sh
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1

trap 'sed -i "s#export cuda_cmd=\"slurm.pl --config conf/slurm_clsp_a100.conf\"#export cuda_cmd=\"slurm.pl\"#" cmd.sh' EXIT
sed -i 's#export cuda_cmd="slurm.pl"#export cuda_cmd="slurm.pl --config conf/slurm_clsp_a100.conf"#' cmd.sh
. ./cmd.sh

MAXLEN=1200
LM_EXP=exp/lm_train_transformer_opt1.3b_en_bpe10000
PROMPT=exp/penalty_sweep/text.tts.new20
[ -s "${PROMPT}" ] || { echo "缺 ${PROMPT}" >&2; exit 1; }

./run.sh --stage 10 --stop_stage 10 --ngpu 1 --gpu_inference true --inference_nj 1 \
    --lm_config conf/tuning/train_transformer_opt1.3b.yaml --inference_lm valid.acc.best.pth \
    --lm_inference_tts_config conf/decode_lm_tts_maxlen1200.yaml \
    --lm_test_text_tts "${PROMPT}"

echo
echo "==== beam5 在新20句上的steps ===="
python3 local/penalty_sweep_report.py "${LM_EXP}/decode_test_tts/text.tts.new20" ${MAXLEN} $(awk '{print $1}' "${PROMPT}")
