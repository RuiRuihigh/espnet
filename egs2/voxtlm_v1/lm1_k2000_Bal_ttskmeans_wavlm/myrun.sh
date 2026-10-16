#!/usr/bin/env bash
#SBATCH --job-name=v1_k2k_ttskm_wavlm
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans_wavlm/log/myrun.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans_wavlm/log/myrun.%j.err

set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans_wavlm

. ./path.sh
. ./cmd.sh

# 跟 lm1_k2000_Bal_ttskmeans 完全一样(D_Bal、k=2000、kmeans 只用 tts train 的 10% 子集训、
# OPT-350M、同一份训练配置),唯一的变量是 kmeans 的输入特征:
#
#   lm1_k2000_Bal_ttskmeans        hubert_base 第6层   (768维, 50Hz)
#   这里(_ttskmeans_wavlm)         wavlm_large 第21层  (1024维, 50Hz, 原版 WavLM-Large 权重)
#
# 本地 lm.sh 只改了一行默认值 kmeans_feature="wavlm_large/21",所以下面每一步都不用额外传参,
# km_dir 自动是 exp/kmeans/wavlm_large_21_2000clusters。
# 跟 voxtlm_v2 的区别:v2 用的是 Sony Phonological Tokenizer(在 WavLM-Large 上微调过的权重 +
# 它自带的码本),这里是原版 WavLM-Large 第21层 + 自己在 tts 子集上训的 kmeans。
#
# WavLM-Large 权重:ckpt/ 是本地真实目录,里面是 s3prl 缓存名的 wavlm_large.pt(从
# voxt_v2/.../lm1/downloads/s3prl_cache 拷来,计算节点不用联网下载)。
#
# 目录复用情况(哪些是软链接、为什么安全),跟 lm1_k2000_Bal_ttskmeans 相同:
#   downloads/*             -> 同一批语料软链接,只读
#   data/{librispeech,librilight,libritts,vctk}
#                           -> lm1_k2000_Bal 的同名目录,只读
#   dump/audio_raw/*/*/data -> k1000 dump 出来的 flac,只读
#   hf_cache                -> 共享的 OPT-350M 权重,只读(HF_HUB_OFFLINE=1)
#   真实拷贝(会被写的):data/{train,dev,test}(没拷 bpe_text,Stage 4 重新生成)、
#   dump/audio_raw/*/* 的清单文件(含 tts/train_subset0.1,跟 hubert 版同一个子集)、
#   conf/、local/、lm.sh、cmd.sh、run.sh、ckpt/。dump/extracted、dump/raw、exp/ 从空开始。
#
# 资源估计(相对 hubert 版):特征 1024 维,tts 10% 子集约 1.07e7 帧 x 1024 x 4B ≈ 44GB
# (hubert 版约 33GB),learn_kmeans 全部读进内存,slurm_kmeans.conf 的 200G 够用;
# 打标签时 WavLM-Large 比 hubert_base 慢约 3 倍。
#
# [注意] 要做 TTS 评测,vocoder 必须用这里的 k2000 标签重新训一个,hubert 版的
# hubert_voc_v1Balk2000_ttskmeans 不能用(单元编号含义不同)。
##############################################################################
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
trap 'sed -i "s#export train_cmd=\"slurm.pl --config conf/slurm_kmeans.conf\"#export train_cmd=\"slurm.pl\"#" cmd.sh' EXIT
sed -i 's#export train_cmd="slurm.pl"#export train_cmd="slurm.pl --config conf/slurm_kmeans.conf"#' cmd.sh
. ./cmd.sh
./run.sh --stage 3 --stop_stage 3 --nclusters 2000 --learn_kmeans true --kmeans_train_task tts
trap - EXIT
sed -i 's#export train_cmd="slurm.pl --config conf/slurm_kmeans.conf"#export train_cmd="slurm.pl"#' cmd.sh
. ./cmd.sh

##############################################################################
# Stage 4: 组装多任务文本;Stage 5: 用 Stage 4 产物训 BPE
# (sentencepiece unigram, nbpe=10000)。都是纯CPU、本地跑。
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
# 被砍掉的条数跟 lm1_k2000_Bal 大概率不一样(BPE 压缩率不同),跑完可以对比
# (k1000: 1,279,028 -> 1,276,237)。
F=exp/lm_stats_en_bpe10000/train/text_shape.bpe
[ -e "$F.orig" ] || cp "$F" "$F.orig"
awk '{split($2,a,","); if (a[1]<=1500) print}' "$F.orig" > "$F"

##############################################################################
# Stage 7: 训练语言模型,OPT-350M 初始化,跟 lm1_k2000_Bal 同一份配置,便于对比。
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
