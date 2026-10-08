#!/usr/bin/env bash
#SBATCH --job-name=sqa_v1_eval
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_sqa/log/myrun_sqa_v1_eval.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_sqa/log/myrun_sqa_v1_eval.%j.err

# 用法: sbatch myrun_sqa_v1_eval.sh [valid.acc.best.pth]
# 训练(myrun_sqa_v1_train.sh)结束或手动停掉之后跑。全部推理走普通 gpu 分区。
#
# 不再跑 dev A-calib 扫描挑 checkpoint(见 conf/tuning/finetune_sqa_v1_opt350.yaml
# 的改动说明):keep_nbest_models 已改成只存 valid acc 最好的 5 个,直接用
# valid.acc.best.pth。流程:
#   1) test_p2(2620 题)上做 A/B/L 全部打分;主指标 A-calib,同时报 A-norm、A-calib_rand
#   2) test Part I 生成(beam5),ROUGE-L/BLEU,附"答案=转写"基线
#   3) textLM/speechLM PPL(Stage 8)遗忘测试。ASR WER 默认不跑(RUN_ASR=true 才跑)
set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_sqa
. ./path.sh
. ./cmd.sh
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1

SRC=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal
LM_EXP=exp/sqa_v1_ft
D=dump/v1/raw
CKPT=${1:-valid.acc.best.pth}
TAG=$(basename ${CKPT} .pth)
RUN_ASR=${RUN_ASR:-false}

##############################################################################
# 1) test Part II
##############################################################################
local/run_mc_loglik.sh --lm_exp ${LM_EXP} --ckpt ${CKPT} --data ${D}/test_p2 \
    --out_dir ${LM_EXP}/mc_loglik/test_p2/${TAG} --variants A,B,L

##############################################################################
# 2) test Part I(生成 + ROUGE)
##############################################################################
local/eval_sqa.sh --part 1 --lm_exp ${LM_EXP} --data ${D}/test_p1 \
    --decode_config conf/decode_sqa_p1.yaml --inference_lm ${CKPT} --nj 8

##############################################################################
# 3) 遗忘: PPL(Stage 8);ASR WER 可选
##############################################################################
stop=8; ${RUN_ASR} && stop=9
./lm.sh --stage 8 --stop_stage ${stop} --skip_train true \
    --lang en --token_type bpe --nbpe 10000 \
    --dumpdir ${SRC}/dump --train_set train --valid_set dev --test_sets test \
    --lm_exp "${LM_EXP}" --inference_lm "${CKPT}" \
    --gpu_inference true --inference_nj 8 \
    --lm_inference_asr_config conf/decode_lm_asr.yaml \
    --lm_test_text_asr ${SRC}/dump/raw/test/text.asr \
    --lm_test_text_textlm ${SRC}/dump/raw/test/text.textlm \
    --lm_test_text_speechlm ${SRC}/dump/raw/test/text.speechlm

echo "==== test Part II (${TAG})"; cat ${LM_EXP}/mc_loglik/test_p2/${TAG}/result.txt
echo "==== test Part I (${TAG})"; cat ${LM_EXP}/decode_sqa/${TAG}_decode_sqa_p1/score_sqa/result.txt
echo "==== PPL"; cat ${LM_EXP}/perplexity_test_textlm/*/ppl ${LM_EXP}/perplexity_test_unitlm/*/ppl
