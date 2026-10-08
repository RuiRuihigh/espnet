#!/usr/bin/env bash
#SBATCH --job-name=sqa_train
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_sqa/log/myrun_sqa_train.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_sqa/log/myrun_sqa_train.%j.err

# 用法: sbatch myrun_sqa_train.sh 2     # 先跑 Part II(单选题)
#       sbatch myrun_sqa_train.sh 1     # Part I(开放问答),脚本已备好,暂时不跑
set -e
set -u
set -o pipefail

PART=${1:?usage: myrun_sqa_train.sh <1|2>}

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_sqa

. ./path.sh
. ./cmd.sh

# 在 lm1_k1000_Bal 的 13epoch.pth(valid.acc.best)上 finetune 一个 spoken QA 任务:
#   <startofspeech> D(语音单元) <startoftext> question: ... [options: a. .. d. ..] <generatetext> 答案
# lm_multitask 把第一个 <generatetext> 之前的全部 mask 掉,loss 只落在答案上,模型代码不用改。
#
# 数据: LibriSQA (downloads/LibriSQA, HF ZihanZhao/LibriSQA)。train = LibriSpeech
# train-clean-360 全部 104,014 条,test = test-clean 全部 2,620 条。语音单元直接复用
# lm1_k1000_Bal/dump/raw/{train,test}/speech/asr/token(官方 km_1000),不重抽 HuBERT。
#
# BPE/词表: data/en_token_list 是从 lm1_k1000_Bal 复制的独立拷贝(md5 一致),
# 不是软链接(吸取 09-19 k2000 覆盖 BPE 的教训)。这里绝对不跑 Stage 4/5。
# BPE 里没有大写和数字 -> prepare_librisqa.py 全部转小写、数字转英文单词,
# 仍有 <unk> 的训练样本直接丢掉(Part II 丢了 58 条)。
#
# 防遗忘: 训练集里混入原四个任务(asr/tts/textlm/unitlm)各 2.5 万条 replay,
# SQA 约占一半。dev 只放 SQA(按说话人留出 18 个 train-clean-360 说话人, 2058 条),
# 所以 valid acc 就是答案 token 的准确率;原任务的遗忘在 myrun_sqa_eval.sh 里单独测。

DUMP=dump/part${PART}
LM_STATS=exp/lm_stats_sqa_part${PART}
LM_EXP=exp/sqa_part${PART}_ft
LM_CONFIG=conf/tuning/finetune_sqa_opt350.yaml
INIT=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal/exp/lm_train_transformer_opt350_en_bpe10000/13epoch.pth

common_opts="--lang en --token_type bpe --nbpe 10000 --dumpdir ${DUMP} \
    --train_set train --valid_set dev --lm_config ${LM_CONFIG} \
    --lm_stats_dir ${LM_STATS} --lm_exp ${LM_EXP}"

##############################################################################
# Step 1: 生成 ${DUMP}/raw/{lm_train.txt,dev/lm_text,test/*}。纯 CPU,约3分钟。
# 已生成就跳过(要重做就先删 ${DUMP})。
#   Part II: 训练集选项顺序随机打乱并同步改写解析里的 "Option X"(原始答案分布
#   C 32% / D 16% 不均衡,打乱后四个字母各约25%);解析里有无法安全改写的字母引用
#   的样本(约2%)保持原顺序。test 保持原顺序,和 LibriSQA 论文可比。
#   结果统计在 ${DUMP}/raw/prep_report.txt。
##############################################################################
if [ ! -f "${DUMP}/raw/lm_train.txt" ]; then
    python local/prepare_librisqa.py --parts "${PART}" --out "${DUMP}/raw"
fi

##############################################################################
# Step 2: Stage 6 统计 text_shape(给 batch sampler 用)。
# SQA 最长 927 BPE token,replay 在准备阶段已按 <=1500 过滤;这里仍沿用原 recipe
# 的 1500 过滤写法兜底(幂等,原文件备份为 .orig)。
##############################################################################
if [ ! -f "${LM_STATS}/train/text_shape.bpe.orig" ]; then
    ./lm.sh --stage 6 --stop_stage 6 --nj 16 ${common_opts}
    F=${LM_STATS}/train/text_shape.bpe
    cp "$F" "$F.orig"
    awk '{split($2,a,","); if (a[1]<=1500) print}' "$F.orig" > "$F"
fi

##############################################################################
# Step 3: Stage 7 finetune(A100)。
# --init_param 只加载模型权重,优化器/调度器从头开始(lr 峰值 1e-4, warmup 300 步)。
# 每 epoch = 2000 minibatch ≈ 100 次更新,约 4 个 epoch 过一遍 SQA+replay。
# valid acc 选 best,valid loss 连续 4 个 epoch 不降自动停(patience 4),最多 30 epoch。
# A100 通过 SQA_CUDA_CMD 传给本目录的 cmd.sh(cmd.sh 里 cuda_cmd 默认读这个变量),
# 不再用 sed 改 cmd.sh。
# lm.sh 的 --resume true: 中断后重提本脚本会从 ${LM_EXP}/checkpoint.pth 续训。
##############################################################################
export SQA_CUDA_CMD="slurm.pl --config conf/slurm_clsp_a100.conf"
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1
./lm.sh --stage 7 --stop_stage 7 --ngpu 1 ${common_opts} \
    --lm_args "--init_param ${INIT}"
