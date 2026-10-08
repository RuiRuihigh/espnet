#!/usr/bin/env bash
#SBATCH --job-name=sqa_v1_test
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_sqa/log/myrun_sqa_v1_test.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_sqa/log/myrun_sqa_v1_test.%j.err

# 用法: sbatch myrun_sqa_v1_test.sh [15epoch.pth]
#       对照实验: LM_EXP=exp/sqa_v1p2_ft D=dump/v1p2/raw SKIP_P1=true sbatch myrun_sqa_v1_test.sh 12epoch.pth
# 只跑 SQA test,checkpoint 直接指定(默认 15epoch = valid loss 最低 1.172),
# 不做 myrun_sqa_v1_eval.sh 里的 dev 扫 epoch 和 PPL。
#   1) Part II test_p2(2620 题)按续写概率打分 A/B/L;主指标 A-calib,另报 A-norm、A-calib_rand
#   2) Part I test_p1(2620 题)beam5 生成 + BLEU/ROUGE-L,附"答案=转写"基线
set -e
set -u
set -o pipefail

CKPT=${1:-15epoch.pth}

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_sqa
. ./path.sh
. ./cmd.sh
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1

LM_EXP=${LM_EXP:-exp/sqa_v1_ft}
D=${D:-dump/v1/raw}
SKIP_P1=${SKIP_P1:-false}
TAG=$(basename ${CKPT} .pth)

# 两部分互不依赖,并行跑
local/run_mc_loglik.sh --lm_exp ${LM_EXP} --ckpt ${CKPT} --data ${D}/test_p2 \
    --out_dir ${LM_EXP}/mc_loglik/test_p2/${TAG} --variants A,B,L &
p2=$!
if ! ${SKIP_P1}; then
    local/eval_sqa.sh --part 1 --lm_exp ${LM_EXP} --data ${D}/test_p1 \
        --decode_config conf/decode_sqa_p1.yaml --inference_lm ${CKPT} --nj 8 &
    p1=$!
fi
wait ${p2}
${SKIP_P1} || wait ${p1}

echo "==== test Part II (${TAG})"; cat ${LM_EXP}/mc_loglik/test_p2/${TAG}/result.txt
${SKIP_P1} || echo "==== test Part I (${TAG})"; ${SKIP_P1} || cat ${LM_EXP}/decode_sqa/${TAG}_decode_sqa_p1/score_sqa/result.txt
