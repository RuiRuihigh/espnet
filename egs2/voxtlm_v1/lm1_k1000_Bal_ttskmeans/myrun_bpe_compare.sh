#!/usr/bin/env bash
#SBATCH --job-name=k1000tts_bpecmp
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_ttskmeans/log/myrun_bpe_compare.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_ttskmeans/log/myrun_bpe_compare.%j.err

set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_ttskmeans

. ./path.sh

# 复刻 /export/fs06/mliang17/espnet/egs2/voxtlm_v1/lm1/reports/bpe_size_comparison_k1000.md
# 的做法:local/compare_bpe_sizes.py 对每个 BPE 词表大小临时训一个 sentencepiece
# 模型(设置跟 lm.sh Stage 5 一致:unigram、character_coverage=1.0、nlsyms 保留
# 特殊token),再用它重新切分 dump/raw/lm_train.txt,统计每个任务(asr/textlm/
# tts/unitlm)以及总体的 token 长度(min/mean/median/P95/max/总token数)。
# 全部输出在本 recipe 自己的目录下,不碰已有的 data/ 和 exp/lm_*:
#   临时模型缓存: exp/bpe_compare/k1000/   (可以随时删,重跑会复用已训好的)
#   报告:         reports/bpe_size_comparison_k1000.md
# nbpe 列表跟老 lm1 那份 k1000 报告(以及 lm1_k1000_Bal 里那份)完全一致,方便直接对照。
python3 local/compare_bpe_sizes.py \
    --nclusters 1000 \
    --nbpe-list 3000,5000,7000,9000,10000 \
    --num-threads 8 \
    --out reports/bpe_size_comparison_k1000.md
