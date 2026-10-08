#!/usr/bin/env bash
#SBATCH --job-name=k2000_salmon
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal/log/myrun_salmon.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal/log/myrun_salmon.%j.err

set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal

. ./path.sh
. ./cmd.sh

##############################################################################
# SALMon(声学一致性 / 声学-语义对齐)评测,8 个子任务各 200 对,共 3200 条音频。
# 纯似然比较:正例/负例两条 wav 各算整句 log-likelihood,正例更高算对,
# 随机水平 50%。不生成、不用 vocoder。
#
# 跟 sLM21 语音轨完全同一套离散化和打分(Phonological Tokenizer 第21层 +
# km2000,unit2cjk 不去重,valid.acc.best.pth),所以数字能跟 sLM21 放在一起看。
# 详见 local/eval_salmon.sh 开头的注释。
#
# 数据已在登录节点下好(downloads/SALMon/hf,stage 1 有 .complete 标记,
# 计算节点没网也不会再去下载),wav 和 datadir 也已经准备好(stage 2),
# 重跑是幂等的。这里从 stage 2 开始跑到 5。
#
# 只写 downloads/SALMon 和 exp/salmon_eval 两个新目录;不改 cmd.sh
# (cuda_cmd 在脚本内临时覆盖成普通 gpu 分区),不碰任何软链接目录。
#
# 结果:exp/salmon_eval/results.tsv(acc_sum 是要报的官方口径)
##############################################################################
bash local/eval_salmon.sh --stage 2 --stop_stage 5 --nj 8
