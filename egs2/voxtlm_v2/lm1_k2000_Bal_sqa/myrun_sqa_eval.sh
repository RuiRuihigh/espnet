#!/usr/bin/env bash
#SBATCH --job-name=sqa_eval
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_sqa/log/myrun_sqa_eval.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_sqa/log/myrun_sqa_eval.%j.err

# 用法: sbatch myrun_sqa_eval.sh 2 [valid.acc.best.pth]
# 训练(myrun_sqa_train.sh)结束或手动停掉之后再跑。解码都是 batch=1 的 beam search,
# 普通 gpu 分区就够(不设 SQA_CUDA_CMD,cmd.sh 默认 slurm.pl)。
set -e
set -u
set -o pipefail

PART=${1:?usage: myrun_sqa_eval.sh <1|2> [inference_lm]}
INFERENCE_LM=${2:-valid.acc.best.pth}

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_sqa

. ./path.sh
. ./cmd.sh
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1

SRC=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal
LM_EXP=exp/sqa_part${PART}_ft
TEST=dump/part${PART}/raw/test

if [ "${PART}" = 2 ]; then
    SQA_CONFIGS="conf/decode_sqa_p2_letter.yaml"   # 准确率只需要 "the answer is option x"
    # 想看完整解析(慢很多,2620 条 × 最多250 token × beam5)就加上:
    # SQA_CONFIGS+=" conf/decode_sqa_p2_full.yaml"
else
    SQA_CONFIGS="conf/decode_sqa_p1.yaml"
fi

##############################################################################
# 0) finetune 前的基线: 原始 16epoch.pth 直接做同样的 prompt(只做一次)。
# 预期基本解析不出选项或接近随机,主要用来证明提升来自 finetune。
# exp/zeroshot_16epoch 里只放软链接,解码结果写在本目录,不碰 lm1_k2000_Bal/exp。
# config.yaml 里 token_list/bpemodel 是相对路径,在本目录解析到同样内容的拷贝。
##############################################################################
ZS=exp/zeroshot_16epoch
if [ ! -e "${ZS}/16epoch.pth" ]; then
    mkdir -p "${ZS}"
    ln -s ${SRC}/exp/lm_train_transformer_opt350_en_bpe10000/16epoch.pth "${ZS}/16epoch.pth"
    ln -s ${SRC}/exp/lm_train_transformer_opt350_en_bpe10000/config.yaml "${ZS}/config.yaml"
fi
for cfg in ${SQA_CONFIGS}; do
    if [ ! -s "${ZS}/decode_sqa/16epoch_$(basename ${cfg} .yaml)/score_sqa/result.txt" ]; then
        local/eval_sqa.sh --part "${PART}" --lm_exp "${ZS}" --data "${TEST}" \
            --decode_config "${cfg}" --inference_lm 16epoch.pth --nj 8
    fi
done

##############################################################################
# 1) finetune 后的 SQA 测试集(test-clean 2620 条)
#   Part II: accuracy(随机 25%,永远选C 32.1%),另有混淆矩阵和 pred_letter
#   Part I : BLEU-4 / ROUGE-L
# 结果: ${LM_EXP}/decode_sqa/<ckpt>_<config>/score_sqa/result.txt
##############################################################################
for cfg in ${SQA_CONFIGS}; do
    local/eval_sqa.sh --part "${PART}" --lm_exp "${LM_EXP}" --data "${TEST}" \
        --decode_config "${cfg}" --inference_lm "${INFERENCE_LM}" --nj 8
done

##############################################################################
# 2) 遗忘评测: 原任务的 Stage 8(textLM/speechLM PPL)+ Stage 9(ASR WER),
# 跟 lm1_k2000_Bal 的 Stage 8/9 同一套测试文件和打分(拿 lm1_k2000_Bal 里 16epoch 的
# textLM/speechLM PPL、ASR WER 做对照)。
# --dumpdir 指向 lm1_k2000_Bal/dump(Stage 9 从 ${data_feats}/test/lm_text 取 ASR 参考,
# 只读);输出写在 ${LM_EXP} 下。TTS(Stage 10 + vocoder)比较重,需要时再单独跑。
##############################################################################
./lm.sh --stage 8 --stop_stage 9 --skip_train true \
    --lang en --token_type bpe --nbpe 10000 \
    --dumpdir ${SRC}/dump --train_set train --valid_set dev --test_sets test \
    --lm_exp "${LM_EXP}" --inference_lm "${INFERENCE_LM}" \
    --gpu_inference true --inference_nj 8 \
    --lm_inference_asr_config conf/decode_lm_asr.yaml \
    --lm_test_text_asr ${SRC}/dump/raw/test/text.asr \
    --lm_test_text_textlm ${SRC}/dump/raw/test/text.textlm \
    --lm_test_text_speechlm ${SRC}/dump/raw/test/text.speechlm

echo "==== SQA ===="
for cfg in ${SQA_CONFIGS}; do
    for d in ${ZS}/decode_sqa/16epoch_$(basename ${cfg} .yaml) \
             ${LM_EXP}/decode_sqa/$(basename ${INFERENCE_LM} .pth)_$(basename ${cfg} .yaml); do
        echo "--- ${d}"; cat "${d}/score_sqa/result.txt"
    done
done
