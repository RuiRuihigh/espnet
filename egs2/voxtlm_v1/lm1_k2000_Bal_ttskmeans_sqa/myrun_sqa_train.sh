#!/usr/bin/env bash
#SBATCH --job-name=k2000_sqa_train
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans_sqa/log/myrun_sqa_train.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans_sqa/log/myrun_sqa_train.%j.err

# 用法: sbatch myrun_sqa_train.sh
set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans_sqa

. ./path.sh
. ./cmd.sh

# k2000 版本的 SQA V1 finetune:完全照搬 lm1_k1000_Bal_sqa 的 V1(myrun_sqa_v1_train.sh),
# 只把底座换成 lm1_k2000_Bal_ttskmeans(hubert_base L6, k=2000, 只用 tts 训的 kmeans;OPT-350M),
# 初始化用它的 17epoch.pth(valid.acc.best,也是 valid loss 最低 2.550)。
#   格式: <startofspeech>语音<startoftext> question: q<generatetext> 答案原文
#         Part I 答案 = 开放式答案;Part II 答案 = 正确选项原文;Part I + II 混合训练
#   数据: 语音单元用 lm1_k2000_Bal_ttskmeans/dump/raw 的 k2000 token,BPE 用它自己的
#         bpe_unigram10000(已复制到 data/en_token_list, md5 f2439be...),和 k1000 的不同
#   dev : 与 k1000 V1 完全相同的 17 个说话人(--dev_speakers_from),dev 条目逐条一致
#   replay: 与 k1000 V1 相同条数 asr 2.5万 / textlm 2.5万 / tts 1万 / unitlm 5千
#           loss token 占比: sqa1 35.5% + sqa2 14.9% = 50.4%(k1000 为 52.6%,k2000 的语音序列略长)
#   超参: 与 k1000 V1 相同(conf/tuning/finetune_sqa_v1_opt350.yaml, lr 1e-4, warmup 300,
#         valid loss 连续不降早停);区别: 只保留 valid loss 最好的 5 个 checkpoint,
#         valid.loss.best.pth 指向 valid loss 最低的那个(测评用它)。
# 测评: myrun_sqa_test.sh ft(finetune 后) / myrun_sqa_test.sh zs(17epoch zero-shot)

DUMP=dump/v1
LM_STATS=exp/lm_stats_sqa_v1
LM_EXP=exp/sqa_v1_ft
LM_CONFIG=conf/tuning/finetune_sqa_v1_opt350.yaml
INIT=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans/exp/lm_train_transformer_opt350_en_bpe10000/17epoch.pth

common_opts="--lang en --token_type bpe --nbpe 10000 --dumpdir ${DUMP} \
    --train_set train --valid_set dev --lm_config ${LM_CONFIG} \
    --lm_stats_dir ${LM_STATS} --lm_exp ${LM_EXP}"

##############################################################################
# Step 1: 数据(纯 CPU,约4分钟;已生成就跳过)
##############################################################################
if [ ! -f "${DUMP}/raw/lm_train.txt" ]; then
    python local/prepare_librisqa.py --parts 1 2 --p2_format option_text \
        --src_dump ../lm1_k2000_Bal_ttskmeans/dump/raw \
        --dev_speakers_from ../lm1_k1000_Bal_sqa/dump/v1/raw/dev/lm_text \
        --replay_counts asr=25000,tts=10000,textlm=25000,unitlm=5000 --out "${DUMP}/raw"
    # Part I test: 与 k1000 相同的 500 条
    mkdir -p ${DUMP}/raw/test_p1_sub500
    for f in lm_text text.sqa transcript; do
        awk 'NR==FNR {k[$1]; next} $1 in k' ../lm1_k1000_Bal_sqa/dump/v1/raw/test_p1_sub500/text.sqa \
            ${DUMP}/raw/test_p1/$f > ${DUMP}/raw/test_p1_sub500/$f
    done
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
