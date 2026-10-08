#!/usr/bin/env bash
#SBATCH --job-name=v1_k2kttskm_opt1.3B
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans_opt1.3B/log/myrun.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans_opt1.3B/log/myrun.%j.err

set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans_opt1.3B

. ./path.sh
. ./cmd.sh

# 跟 lm1_k2000_Bal_ttskmeans 完全一样的数据(D_Bal,hubert_base 第6层,tts 10%
# 子集训的 kmeans),唯一的变量是 Stage 7 用 OPT-1.3B 初始化,不再是 OPT-350M。
#
# 这个实验只换模型配置,数据/切分/kmeans/BPE/统计量一概不变,所以不需要跑
# Stage 1-6 —— 它们的产物(downloads/、data/、dump/)整个目录直接软链接自
# lm1_k2000_Bal_ttskmeans(只读复用),exp/lm_stats_en_bpe10000(Stage 6 的
# 统计量,Stage 7 的 batch sampler 要读)也是软链接。exp/ 本身没有整体链接,
# 因为 Stage 7 要在这里新建自己的训练目录(exp/lm_train_transformer_opt1.3b_
# en_bpe10000,按 lm_config 的文件名自动命名,不会跟 350M 那份撞名)。
#
# conf/tuning/train_transformer_opt1.3b.yaml 是从 train_transformer_opt350.yaml
# 复制的,只改了一行 lm_conf.opt_name: facebook/opt-350m -> facebook/opt-1.3b,
# 其余(batch_bins/accum_grad/学习率等)原样不动。facebook/opt-1.3b 已经在共享
# 的 hf_cache 里缓存好(config.json/model.safetensors/pytorch_model.bin 都在),
# 不需要联网下载。
#
# 【没有把握的地方,提交前要知道】
# OPT-1.3B 参数量是 350M 的约3.7倍,batch_bins/accum_grad 完全没动,显存会不会
# 够没有验证过(这个 cluster 的 A100 是 40G 还是 80G 我也没查到确切型号)。
# 如果 Stage 7 一开始就 OOM,大概率需要把 batch_bins 调小、accum_grad 相应调大
# 来维持差不多的等效 batch size,或者开梯度检查点(gradient checkpointing,
# 如果 HuggingfaceOPTModel 支持的话)。这个改动我没有先做,想先看它到底会不会
# OOM,不确定的情况下先别过度调参。

##############################################################################
# Stage 7: 训练语言模型,OPT-1.3B 初始化。
#
# 全流程只有这一步用 A100,跑完(或被手动叫停后)用 trap 把 cuda_cmd 改回
# 普通 GPU。训练不会自动停,要盯着 train.log 的 valid acc 判断过拟合。
##############################################################################
trap 'sed -i "s#export cuda_cmd=\"slurm.pl --config conf/slurm_clsp_a100.conf\"#export cuda_cmd=\"slurm.pl\"#" cmd.sh' EXIT
sed -i 's#export cuda_cmd="slurm.pl"#export cuda_cmd="slurm.pl --config conf/slurm_clsp_a100.conf"#' cmd.sh
. ./cmd.sh
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1
./run.sh --stage 7 --stop_stage 7 --ngpu 1 --nclusters 2000 \
    --lm_config conf/tuning/train_transformer_opt1.3b.yaml
