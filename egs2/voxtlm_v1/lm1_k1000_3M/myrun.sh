#!/usr/bin/env bash
#SBATCH --job-name=v1_k1k3M
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_3M/log/myrun.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_3M/log/myrun.%j.err

set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_3M

. ./path.sh
. ./cmd.sh

# lm1_k1000_Bal 的 D_3M 版本:hubert_base 第6层 + 论文作者发布的官方 km_1000(外部模型,
# 不重新拟合)+ OPT-350M 初始化,唯一变量是 speechlm/textlm 的训练规模(D_Bal -> D_3M)。
#
# 目录复用(见 lm1_k1000_3M 的骨架搭建):
#   downloads/*             同 lm1_k2000_3M_ttskmeans(绝对路径软链接,只读)
#   data/{librilight,librispeech,libritts,vctk}  -> voxtlm_v2/lm1_k2000_3M 同名目录
#   data/{train,dev,test}   真实拷贝(3M 切分),Stage 4 会写 train/bpe_text,不能共享
#   dump/audio_raw/asr|tts  清单真实拷贝(与 Bal 逐字节一致),data/ 软链接到 v2 3M
#   dump/audio_raw/speechlm/train  3M 过滤后清单(2,990,192 条)真实拷贝
#   dump/extracted/{asr,tts}       Bal 的标签 + .labeled_km1000_<md5> 标记,Stage 3 会跳过
#   dump/extracted/speechlm        空目录,Stage 3 会对 3M 清单重新打标签
#   dump/raw/test                  真实拷贝自 Bal(含 text.tts.rand100 / subset100)
#
# Stage 3 只给 speechlm 打标签:asr/tts 的标记文件 md5 与当前 km 模型一致,lm.sh
# 会识别为已完成。speechlm 3M 约 300 万条,是整个流程最慢的一步。
##############################################################################

##############################################################################
# Stage 3: speechlm 3M 打标签(官方 km_1000,只读,不拟合)
##############################################################################
KM_DIR=/export/fs06/mliang17/shared_kmeans/hubert_base_l6_k1000_official
md5sum ${KM_DIR}/km_1000.mdl  # 只打印,应为 d46744528bf7f8dba295d560925fc1cf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
./run.sh --stage 3 --stop_stage 3 --nclusters 1000 --learn_kmeans false --km_dir "${KM_DIR}"

##############################################################################
# Stage 4: 组装多任务文本;Stage 5: BPE(sentencepiece unigram, nbpe=10000)
##############################################################################
./run.sh --stage 4 --stop_stage 5 --nclusters 1000

##############################################################################
# Stage 6: 统计量 + 长度过滤(>1500 的序列在 OPT 位置编码上限附近,Stage 7 会 OOM)
##############################################################################
./run.sh --stage 6 --stop_stage 6 --nclusters 1000
F=exp/lm_stats_en_bpe10000/train/text_shape.bpe
[ -e "$F.orig" ] || cp "$F" "$F.orig"
awk '{split($2,a,","); if (a[1]<=1500) print}' "$F.orig" > "$F"

##############################################################################
# Stage 7: OPT-350M 训练,A100 只在这一步,trap 无论成败都切回普通 gpu
##############################################################################
trap 'sed -i "s#export cuda_cmd=\"slurm.pl --config conf/slurm_clsp_a100.conf\"#export cuda_cmd=\"slurm.pl\"#" cmd.sh' EXIT
sed -i 's#export cuda_cmd="slurm.pl"#export cuda_cmd="slurm.pl --config conf/slurm_clsp_a100.conf"#' cmd.sh
. ./cmd.sh
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1
./run.sh --stage 7 --stop_stage 7 --ngpu 1 --nclusters 1000 \
    --lm_config conf/tuning/train_transformer_opt350.yaml
