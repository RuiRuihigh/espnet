#!/usr/bin/env bash
#SBATCH --job-name=v1_k2k_wavlm_lm
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans_wavlm/log/myrun_train.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans_wavlm/log/myrun_train.%j.err

# 2026-10-11 只跑 Stage 7(LM 训练)。Stage 1-6 已由 myrun.sh(1819222)跑完,
# text_shape.bpe 的 >1500 过滤也已做完(.orig 是备份),这里不再重复。
#
# A100 任务带 24 小时时限(--time 24:00:00),好让 backfill 有机会插空,优先级本身不变。
# 到 24 小时 A100 任务被 slurm 杀掉,本脚本随之报错结束,不自动续训。
# 之后要接着训的话重新 sbatch 本脚本即可:lm.sh 带 --resume true,从 checkpoint.pth 接着训;
# 但 slurm.pl 会覆盖 train.log,重提之前先手动把它备份一下。
# 24 小时内训练不会自己停(max_epoch 250),要看 valid acc 手动 scancel 本任务和它的 A100 子任务。
set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans_wavlm

. ./path.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1

NCLUSTERS=2000
A100_CMD="slurm.pl --config conf/slurm_clsp_a100.conf --time 24:00:00"

# lm.sh 会重新 source cmd.sh,所以只能临时改 cmd.sh,退出时(包括被 scancel)改回普通 slurm.pl
trap "sed -i 's#export cuda_cmd=\"${A100_CMD}\"#export cuda_cmd=\"slurm.pl\"#' cmd.sh" EXIT
sed -i "s#export cuda_cmd=\"slurm.pl\"#export cuda_cmd=\"${A100_CMD}\"#" cmd.sh
grep -q "export cuda_cmd=\"${A100_CMD}\"" cmd.sh || { echo "cmd.sh 里的 cuda_cmd 没改成功" >&2; exit 1; }

./run.sh --stage 7 --stop_stage 7 --ngpu 1 --nclusters "${NCLUSTERS}" \
    --lm_config conf/tuning/train_transformer_opt350.yaml
