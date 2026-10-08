#!/usr/bin/env bash
#SBATCH --job-name=sqa_v1_eval
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_sqa/log/myrun_sqa_v1_eval.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_sqa/log/myrun_sqa_v1_eval.%j.err

# 用法: sbatch myrun_sqa_v1_eval.sh            # 扫所有 epoch
#       EPOCHS="4 8 12 16" sbatch myrun_sqa_v1_eval.sh   # 只扫指定 epoch
# 训练(myrun_sqa_v1_train.sh)结束或手动停掉之后跑。全部推理走普通 gpu 分区。
#
# 流程(先在 dev 上选,再上 test,避免在 test 上挑方法/挑 checkpoint):
#   0) 原始 13epoch 在 dev_p2 上的 A 打分(dev 上的 zero-shot 参照)
#   1) 每个 epoch 在 dev_p2(17 个留出说话人,2013 题)上做 A 打分,按 A-calib 选 best epoch
#   2) best epoch 在 test_p2(2620 题)上做 A/B/L 全部打分;主指标 A-calib,同时报 A-norm、A-calib_rand
#   3) best epoch 做 Part I test 生成(beam5),ROUGE-L/BLEU,附"答案=转写"基线
#   4) best epoch 的 textLM/speechLM PPL(Stage 8)。ASR WER 默认不跑(RUN_ASR=true 才跑)
set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_sqa
. ./path.sh
. ./cmd.sh
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1

SRC=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal
LM_EXP=exp/sqa_v1_ft
D=dump/v1/raw
ZS=exp/zeroshot_13epoch
RUN_ASR=${RUN_ASR:-false}
if [ -z "${EPOCHS:-}" ]; then
    EPOCHS=$(ls ${LM_EXP} | grep -E '^[0-9]+epoch\.pth$' | sed 's/epoch.pth//' | sort -n | tr '\n' ' ')
fi
acc_of() { awk -F'\t' -v v="$2" -v s="$3" '$1==v && $2==s {print $3}' "$1"; }

##############################################################################
# 0) zero-shot 参照(dev)
##############################################################################
local/run_mc_loglik.sh --lm_exp ${ZS} --ckpt 13epoch.pth --data ${D}/dev_p2 \
    --out_dir ${ZS}/mc_loglik/v1_dev_p2/13epoch --variants A

##############################################################################
# 1) dev 扫 epoch,按 A-calib 选 best
##############################################################################
for ep in ${EPOCHS}; do
    local/run_mc_loglik.sh --lm_exp ${LM_EXP} --ckpt ${ep}epoch.pth --data ${D}/dev_p2 \
        --out_dir ${LM_EXP}/mc_loglik/dev_p2/${ep}epoch --variants A
done
{
    printf "ckpt\tA-sum\tA-norm\tA-calib\tA-calib_rand\n"
    r=${ZS}/mc_loglik/v1_dev_p2/13epoch/result.txt
    printf "13epoch(zs)\t%s\t%s\t%s\t%s\n" "$(acc_of $r A sum)" "$(acc_of $r A norm)" "$(acc_of $r A calib)" "$(acc_of $r A calib_rand)"
    for ep in ${EPOCHS}; do
        r=${LM_EXP}/mc_loglik/dev_p2/${ep}epoch/result.txt
        printf "%sepoch\t%s\t%s\t%s\t%s\n" ${ep} "$(acc_of $r A sum)" "$(acc_of $r A norm)" "$(acc_of $r A calib)" "$(acc_of $r A calib_rand)"
    done
} > ${LM_EXP}/mc_loglik/dev_p2/sweep.tsv
BEST=$(awk -F'\t' 'NR>1 && $1!~/zs/ {if ($4>m) {m=$4; b=$1}} END {print b}' ${LM_EXP}/mc_loglik/dev_p2/sweep.tsv)
echo "${BEST}.pth" > ${LM_EXP}/best_dev_ckpt
echo "==== dev sweep (best by A-calib: ${BEST})"; column -t -s$'\t' ${LM_EXP}/mc_loglik/dev_p2/sweep.tsv

##############################################################################
# 2) test Part II
##############################################################################
local/run_mc_loglik.sh --lm_exp ${LM_EXP} --ckpt ${BEST}.pth --data ${D}/test_p2 \
    --out_dir ${LM_EXP}/mc_loglik/test_p2/${BEST} --variants A,B,L

##############################################################################
# 3) test Part I(生成 + ROUGE)
##############################################################################
local/eval_sqa.sh --part 1 --lm_exp ${LM_EXP} --data ${D}/test_p1 \
    --decode_config conf/decode_sqa_p1.yaml --inference_lm ${BEST}.pth --nj 8

##############################################################################
# 4) 遗忘: PPL(Stage 8);ASR WER 可选
##############################################################################
stop=8; ${RUN_ASR} && stop=9
./lm.sh --stage 8 --stop_stage ${stop} --skip_train true \
    --lang en --token_type bpe --nbpe 10000 \
    --dumpdir ${SRC}/dump --train_set train --valid_set dev --test_sets test \
    --lm_exp "${LM_EXP}" --inference_lm "${BEST}.pth" \
    --gpu_inference true --inference_nj 8 \
    --lm_inference_asr_config conf/decode_lm_asr.yaml \
    --lm_test_text_asr ${SRC}/dump/raw/test/text.asr \
    --lm_test_text_textlm ${SRC}/dump/raw/test/text.textlm \
    --lm_test_text_speechlm ${SRC}/dump/raw/test/text.speechlm

echo "==== test Part II (${BEST})"; cat ${LM_EXP}/mc_loglik/test_p2/${BEST}/result.txt
echo "==== test Part I (${BEST})"; cat ${LM_EXP}/decode_sqa/${BEST}_decode_sqa_p1/score_sqa/result.txt
echo "==== PPL"; cat ${LM_EXP}/perplexity_test_textlm/*/ppl ${LM_EXP}/perplexity_test_unitlm/*/ppl
