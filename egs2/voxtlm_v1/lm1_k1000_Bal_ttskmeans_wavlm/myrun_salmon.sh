#!/usr/bin/env bash
#SBATCH --job-name=v1k1000w_salmon
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_ttskmeans_wavlm/log/myrun_salmon.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_ttskmeans_wavlm/log/myrun_salmon.%j.err

set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_ttskmeans_wavlm

. ./path.sh
. ./cmd.sh

##############################################################################
# SALMon(声学一致性 / 声学-语义对齐)评测,8 个子任务各 200 对,共 3200 条音频。
# 纯似然比较,不生成、不用 vocoder。流程跟 voxtlm_v2/lm1_k2000_Bal/myrun_salmon.sh
# 完全一样,只是离散化换成本 recipe 的 wavlm_large 第21层 + km1000
# (跟训练数据、sLM21 语音轨同一个 kmeans),LM 用 valid.acc.best.pth。
#
# 数据直接按绝对路径只读 v2 那份 downloads/SALMon(不在本 recipe 的
# downloads/ 下建软链接——lm1_k1000_Bal/downloads 被 opt1.3B 共用)。
# 只写 exp/salmon_eval;不改 cmd.sh;kmeans 和 ckpt/ 只读。
#
# 结果:exp/salmon_eval/results.tsv(acc_sum 是官方口径)
##############################################################################
bash local/eval_salmon.sh --stage 1 --stop_stage 5 --nj 8
