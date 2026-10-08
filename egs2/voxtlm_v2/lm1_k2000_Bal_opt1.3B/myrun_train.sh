#!/usr/bin/env bash
#SBATCH --job-name=opt1.3b_train
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_opt1.3B/log/myrun_train.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_opt1.3B/log/myrun_train.%j.err

# 用法: sbatch myrun_train.sh
#
# 对齐 lm1_k2000_Bal/myrun.sh 里已经跑完的 Stage 7(同样的
# batch_bins/accum_grad/lr/scheduler,唯一区别是 conf/tuning/
# train_transformer_opt1.3b.yaml 里 opt_name 换成了 facebook/opt-1.3b),
# 只是从头单独训练 —— OPT-350M 的 embed_tokens/lm_head 是按咱们自己的 BPE 词表
# 随机初始化的(espnet2/lm/huggingface_pretrained_opt_lm.py 里
# pop("decoder.embed_tokens.weight")),跟 opt_name 用哪个无关,所以 lm1_k2000_Bal
# 已有的 16epoch.pth 没法当 --init_param 复用,这里是真正意义上的从零训练。
#
# 数据(dump/raw、data/en_token_list)、Stage6 统计(exp/lm_stats_en_bpe10000)全部
# 软链接自 lm1_k2000_Bal,没有重跑 Stage 1-6,也不会写回那边(exp/ 在本目录是
# 本地真目录,新 checkpoint 落在本地的 exp/lm_train_transformer_opt1.3b_en_bpe10000)。
#
# OPT-1.3b 权重已经手动预下载进共享 hf_cache(跟 opt-350m 在同一个缓存里,纯新增,
# 没动原有内容),HF_HUB_OFFLINE=1 离线模式验证过能正常加载。
#
# 显存是这次最大的未知数: OPT-350M 在同样的 batch_bins=10000/accum_grad=20/ngpu=1
# 下峰值约 30GB(A100 80GB,量数据来自 SQA finetune 的实测)。OPT-1.3B 参数量约
# 4倍(3.3亿->13.2亿,已验证 hidden=2048/24层/32头),Adam 优化器状态、激活值都会
# 明显变大,有没有 80GB 装不下的风险没实测过。如果一提交就 OOM,先把下面
# --lm_args 里的 batch_bins 砍半、accum_grad 翻倍(保持有效 batch size 不变)再试,
# 不要改 lr/scheduler,否则就不是同一套超参数的对照实验了。
#
# 没有配置早停(跟 lm1_k2000_Bal 一样),需要自己盯着 train.log 的 valid acc/loss,
# 该收敛就手动停(lm1_k2000_Bal 自己在 16epoch 附近过拟合,21epoch 后手动停的)。
set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_opt1.3B
. ./path.sh
. ./cmd.sh
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1

##############################################################################
# Stage 7: LM 训练(A100)。跟 lm1_k2000_Bal/myrun.sh 一样用 sed+trap 临时切到
# A100 分区,退出时(不管成功/失败/被 scancel)都尽量切回去 —— 注意 scancel 父
# 任务本身可能来不及跑 trap(之前 vocoder 评测那次就踩过这个坑),如果手动中断
# 记得跑一下 `grep cuda_cmd cmd.sh` 确认有没有卡在 A100 配置上。
##############################################################################
trap 'sed -i "s#export cuda_cmd=\"slurm.pl --config conf/slurm_clsp_a100.conf\"#export cuda_cmd=\"slurm.pl\"#" cmd.sh' EXIT
sed -i 's#export cuda_cmd="slurm.pl"#export cuda_cmd="slurm.pl --config conf/slurm_clsp_a100.conf"#' cmd.sh
. ./cmd.sh
./run.sh --stage 7 --stop_stage 7 --ngpu 1 --lm_config conf/tuning/train_transformer_opt1.3b.yaml
