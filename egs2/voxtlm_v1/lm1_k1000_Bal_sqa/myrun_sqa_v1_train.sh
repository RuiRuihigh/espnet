#!/usr/bin/env bash
#SBATCH --job-name=sqa_v1_train
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_sqa/log/myrun_sqa_v1_train.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_sqa/log/myrun_sqa_v1_train.%j.err

# 用法: sbatch myrun_sqa_v1_train.sh
set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_sqa

. ./path.sh
. ./cmd.sh

# V1: 按 A(cloze)打分方式重新设计的训练。上一版(myrun_sqa_train.sh 2)的问题:
#   目标是 "the answer is option x because <~150 token 解析>",决定对错的只有 1 个字母 token;
#   test 贪心生成 34.9%,而按续写概率打分(A-calib)41.1%,原始 13epoch 不训练 A-norm 就有 37.8%,
#   finetune 对理解的提升很小。
# V1 的格式(训练=测试,loss 全在答案内容上,没有字母/选项/解析):
#   <startofspeech>语音<startoftext> question: q<generatetext> 答案原文
#   Part I : 答案 = LibriSQA 开放式答案(平均 ~38 BPE token)
#   Part II: 答案 = 正确选项原文(平均 ~16 BPE token);测试时对 4 个选项原文算
#            log P(选项|语音,问题),A-calib 再减去语音段为空时的分数
# 两部分同一批 train-clean-360 语音、不同问题,合在一起训;dev 用两部分共享的
# 17 个留出说话人(dev 语音不会通过另一部分漏进训练,已检查说话人交集为0)。
#
# replay 重新配比(答案变短后 SQA 的 loss token 从 1640万 降到 ~570万,不调的话只占30%):
#   asr 2.5万 / textlm 2.5万 / tts 1万 / unitlm 5千 -> loss token 占比
#   sqa1 37.0% + sqa2 15.6% = 52.6%, asr 13.0%, unitlm 14.9%, tts 11.4%, textlm 8.1%
#   (统计在 dump/v1/raw/prep_report.txt)
#
# 训练超参同上一版(从 13epoch.pth 初始化, lr 1e-4, warmup 300, patience 4 按 valid loss),
# 区别是保留每个 epoch 的 checkpoint(keep_nbest_models 30),之后由 myrun_sqa_v1_eval.sh
# 用 dev 上的 A-calib 准确率挑 checkpoint,而不是 valid acc。

DUMP=dump/v1
LM_STATS=exp/lm_stats_sqa_v1
LM_EXP=exp/sqa_v1_ft
LM_CONFIG=conf/tuning/finetune_sqa_v1_opt350.yaml
INIT=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal/exp/lm_train_transformer_opt350_en_bpe10000/13epoch.pth

common_opts="--lang en --token_type bpe --nbpe 10000 --dumpdir ${DUMP} \
    --train_set train --valid_set dev --lm_config ${LM_CONFIG} \
    --lm_stats_dir ${LM_STATS} --lm_exp ${LM_EXP}"

##############################################################################
# Step 1: 数据(纯 CPU,约4分钟;已生成就跳过)
##############################################################################
if [ ! -f "${DUMP}/raw/lm_train.txt" ]; then
    python local/prepare_librisqa.py --parts 1 2 --p2_format option_text \
        --replay_counts asr=25000,tts=10000,textlm=25000,unitlm=5000 --out "${DUMP}/raw"
fi

##############################################################################
# Step 2: Stage 6 统计 + 1500 长度过滤兜底(同上一版)
##############################################################################
if [ ! -f "${LM_STATS}/train/text_shape.bpe.orig" ]; then
    ./lm.sh --stage 6 --stop_stage 6 --nj 16 ${common_opts}
    F=${LM_STATS}/train/text_shape.bpe
    cp "$F" "$F.orig"
    awk '{split($2,a,","); if (a[1]<=1500) print}' "$F.orig" > "$F"
fi

##############################################################################
# Step 3: Stage 7 finetune(A100,通过 SQA_CUDA_CMD 传给本目录 cmd.sh)
##############################################################################
export SQA_CUDA_CMD="slurm.pl --config conf/slurm_clsp_a100.conf"
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1
./lm.sh --stage 7 --stop_stage 7 --ngpu 1 ${common_opts} \
    --lm_args "--init_param ${INIT}"
