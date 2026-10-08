#!/usr/bin/env bash
#SBATCH --job-name=v1_k2000_Bal
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal/log/myrun.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal/log/myrun.%j.err

set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal

. ./path.sh
. ./cmd.sh

# 这份 recipe 的定位:跟隔壁 lm1_k1000_Bal 完全一样的流程和数据(D_Bal,
# hubert_base 第6层特征),唯一的变量是离散单元数 k=1000 -> k=2000,用来看
# 词表粒度对 speechLM/ASR/TTS 的影响。
#
# 注意跟 voxtlm_v2/lm1_k2000_Bal 区分开:那个是"音系学tokenizer"(微调过的
# WavLM-large 第21层 + 给定中心),跟这个不是一回事。这里是纯 HuBERT+kmeans。
#
# 目录复用情况(哪些是软链接、为什么安全):
#   downloads/*            -> 语料本体和 k1000 已切好的 *_segmented,只读
#   dump/audio_raw/{asr,speechlm,tts} -> k1000 的清单,只读
#       (dump/audio_raw 本身是本地真实目录,所以 Stage 3 建 kmeans 池子时
#        写出来的 kmeans_pool 落在本目录,不会污染 k1000)
#   data/                  本地真实拷贝(8.7G),不跟任何人共享
#       -- 教训:之前 voxtlm_v2/lm1_k2000_Bal 把 data/ 软链接到 k1000,
#          两边 Stage 4/5 互相覆盖 bpe.model,把 k1000 训练到一半的任务搞崩了。
#          BPE、token_list 这种"每个实验各不相同的产物"一律不共享。
#   ckpt / hf_cache        -> 老 checkout,只读
#
# 不需要跑 Stage 1(VAD切分+D_Bal抽样)和 Stage 2(wav dump):它们的产物就是
# 上面复制/链接过来的 data/ 和 dump/audio_raw。
#
# 也不需要跑 local/filter_long_utts.sh:链接过来的 speechlm/train 清单已经是
# k1000 过滤掉 >120秒之后的 299,122 条。而且那个脚本是原地改清单的,对着软
# 链接跑会写到 k1000 的目录里去,千万别在这里跑。

##############################################################################
# Stage 3: 训练 k=2000 的 kmeans,并给 asr/tts/speechlm 打标签。
#
# 跟 k1000 不同,这一步必须自己训 kmeans:论文作者只放出了 k=1000 和 k=200
# 的模型(https://huggingface.co/soumi-maiti,没有 k2000),所以这里用
# --learn_kmeans true(lm.sh 的默认值),--nclusters 2000。
# km_dir 不指定时,lm.sh 会自动用 exp/kmeans/hubert_base_6_2000clusters。
#
# 这一步内部依次做三件事:
#   1) 建平衡池子:从 asr/tts/speechlm 各抽约600万帧(50Hz)音频到
#      dump/audio_raw/kmeans_pool
#   2) 提 hubert_base 第6层特征到 dump/extracted/kmeans_pool(约50GB,普通GPU)
#   3) 拟合 MiniBatchKMeans:要把整个池子的特征读进内存(约55GB+),普通分区
#      给不了,所以只把这一步临时切到 conf/slurm_kmeans.conf(gpu-a100 节点
#      借内存 200G,不占GPU),跑完用 trap 改回来。
#      注意 perform_kmeans.sh 里只有 learn_kmeans.py 用 cpu_cmd(=train_cmd),
#      打标签那两步用的是 cuda_cmd,所以 cuda_cmd 保持普通 GPU 不动。
#
# 打完标签后可以跟 k1000 比一下:同一条音频,k2000 的单元序列长度应该一样
# (都是50Hz一帧一个单元),只是单元编号范围变成 0-1999。
#
# [注意] 以后要做 TTS 评测的话,vocoder 必须重新用 k2000 的标签训一个,
# k1000 那个 vocoder 在这里不能用(单元编号含义完全不同)。
##############################################################################
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
trap 'sed -i "s#export train_cmd=\"slurm.pl --config conf/slurm_kmeans.conf\"#export train_cmd=\"slurm.pl\"#" cmd.sh' EXIT
sed -i 's#export train_cmd="slurm.pl"#export train_cmd="slurm.pl --config conf/slurm_kmeans.conf"#' cmd.sh
. ./cmd.sh
./run.sh --stage 3 --stop_stage 3 --nclusters 2000 --learn_kmeans true
trap - EXIT
sed -i 's#export train_cmd="slurm.pl --config conf/slurm_kmeans.conf"#export train_cmd="slurm.pl"#' cmd.sh
. ./cmd.sh

##############################################################################
# Stage 4: 组装多任务文本;Stage 5: 用 Stage 4 产物训 BPE
# (sentencepiece unigram, nbpe=10000)。都是纯CPU、本地跑。
# 词表总量不变(还是10000),但语音侧的基元从1000个变成2000个,BPE 会把更多
# 额度分给语音 token,压缩率和序列长度跟 k1000 会不一样——这正是要看的东西。
# nclusters 在这两步也要传,因为 lm.sh 用它拼 pseudo_labels 的文件名。
##############################################################################
./run.sh --stage 4 --stop_stage 5 --nclusters 2000

##############################################################################
# Stage 6: 统计信息收集(算 text_shape.bpe,给 Stage 7 的 batch sampler 用)。
##############################################################################
./run.sh --stage 6 --stop_stage 6 --nclusters 2000

# 过滤掉长度 >1500 的序列:离散语音单元序列没有天然句子边界,极少数超长序列
# 会让 Stage 7 OOM(self-attention 显存 O(L^2)),而且超过 OPT 的2048位置上限
# 会直接报 index out of bounds。备份原始文件,脚本本身幂等。
# k2000 的压缩率跟 k1000 不同,被砍掉的条数大概率不一样,跑完可以对比下
# (k1000: 1,279,028 -> 1,276,237)。
F=exp/lm_stats_en_bpe10000/train/text_shape.bpe
[ -e "$F.orig" ] || cp "$F" "$F.orig"
awk '{split($2,a,","); if (a[1]<=1500) print}' "$F.orig" > "$F"

##############################################################################
# Stage 7: 训练语言模型,OPT-350M 初始化,跟 k1000 同一份配置,便于对比。
# 全流程只有这一步用 A100,跑完(或被手动叫停后)用 trap 把 cuda_cmd 改回
# 普通 GPU。训练不会自动停,要盯着 train.log 的 valid acc 判断过拟合。
##############################################################################
trap 'sed -i "s#export cuda_cmd=\"slurm.pl --config conf/slurm_clsp_a100.conf\"#export cuda_cmd=\"slurm.pl\"#" cmd.sh' EXIT
sed -i 's#export cuda_cmd="slurm.pl"#export cuda_cmd="slurm.pl --config conf/slurm_clsp_a100.conf"#' cmd.sh
. ./cmd.sh
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1
./run.sh --stage 7 --stop_stage 7 --ngpu 1 --nclusters 2000 \
    --lm_config conf/tuning/train_transformer_opt350.yaml
