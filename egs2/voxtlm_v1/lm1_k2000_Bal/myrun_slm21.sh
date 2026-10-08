#!/usr/bin/env bash
#SBATCH --job-name=v1k2000_slm21
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal/log/myrun_slm21.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal/log/myrun_slm21.%j.err

set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal

. ./path.sh
. ./cmd.sh

# 单独拆出来跑,跟 myrun_eval.sh 分开:sLM21 只需要训好的 LM checkpoint
# (valid.acc.best.pth = 18epoch.pth),跟 Stage 8-10/vocoder 完全独立,不用等
# Stage 10 用新 maxlen=900 重跑完再排队。三个 recipe(k1000/v1 k2000/v2 k2000)
# 里只有这份还没跑过 sLM21,先单独补上。
#
# sWUGGY(词汇)+ sBLIMP(句法),文本轨 + 语音轨(用本实验自己训的
# exp/kmeans/hubert_base_6_2000clusters 打标签)+ 最终打分,三步顺序执行。
# 只跑 dev(论文报的就是 dev)。语音轨最重,dev 的 80000+50400=130400 条音频
# 全部要过特征提取+打标签,k1000 那边这步约 25 分钟。
bash local/eval_slm21_text.sh
bash local/eval_slm21_speech.sh
bash local/eval_slm21_score.sh

echo "sLM21 跑完,结果在 exp/slm21_eval/submission/,准确率见上面 eval_slm21_score.sh 的输出。"
