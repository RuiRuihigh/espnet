#!/usr/bin/env bash
#SBATCH --job-name=sqa_v1p2_train
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_sqa/log/myrun_sqa_v1p2_train.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_sqa/log/myrun_sqa_v1p2_train.%j.err

# 用法: sbatch myrun_sqa_v1p2_train.sh
set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_sqa

. ./path.sh
. ./cmd.sh

# 对照实验 V1-P2only: 只去掉 Part I,其他全部和 V1(myrun_sqa_v1_train.sh)相同,
# 用来回答"加入 Part I 对 Part II 有没有害"。
#   相同: 格式(prompt 只有问题, 目标=正确选项原文)、Part II 训练行(与 V1 逐行一致, 101,988 条)、
#         dev 说话人(--dev_speakers_from V1 的 dev, 17 人)、test_p2(与 V1 逐字节一致)、
#         初始化(13epoch.pth)、超参(同一个 conf/tuning/finetune_sqa_v1_opt350.yaml)
#   不同: 没有 Part I;replay 按比例缩小(asr/textlm 7.3k, tts 2.9k, unitlm 1.5k),
#         让 SQA 仍占 ~52% loss token(V1 52.6%, 这里 52.5%; asr 12.7 / unitlm 15.4 / tts 11.4 / textlm 8.0)
# 注意: 总数据量约为 V1 的 45%,每个 epoch(固定 2000 minibatch)覆盖的数据比例更高;
#       valid loss 只含 Part II,和 V1 的 valid loss 不可直接比。
#       checkpoint 选取规则同 V1:valid loss 最低的 epoch。

DUMP=dump/v1p2
LM_STATS=exp/lm_stats_sqa_v1p2
LM_EXP=exp/sqa_v1p2_ft
LM_CONFIG=conf/tuning/finetune_sqa_v1_opt350.yaml
INIT=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal/exp/lm_train_transformer_opt350_en_bpe10000/13epoch.pth

common_opts="--lang en --token_type bpe --nbpe 10000 --dumpdir ${DUMP} \
    --train_set train --valid_set dev --lm_config ${LM_CONFIG} \
    --lm_stats_dir ${LM_STATS} --lm_exp ${LM_EXP}"

##############################################################################
# Step 1: 数据(纯 CPU,约4分钟;已生成就跳过)
##############################################################################
if [ ! -f "${DUMP}/raw/lm_train.txt" ]; then
    python local/prepare_librisqa.py --parts 2 --p2_format option_text --part_dirs \
        --dev_speakers_from dump/v1/raw/dev/lm_text \
        --replay_counts asr=7300,tts=2900,textlm=7300,unitlm=1500 --out "${DUMP}/raw"
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
