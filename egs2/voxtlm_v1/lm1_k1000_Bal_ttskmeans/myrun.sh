#!/usr/bin/env bash
#SBATCH --job-name=v1_k1k_ttskm
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_ttskmeans/log/myrun.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_ttskmeans/log/myrun.%j.err

set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_ttskmeans

. ./path.sh
. ./cmd.sh

# 这份 recipe 是 lm1_k2000_Bal_ttskmeans 的 k=1000 版本:数据(D_Bal,
# hubert_base 第6层)、流程、配置全部照搬,唯一的变量是 k 从 2000 换成 1000。
# kmeans 的训法照搬上游(RuiRuihigh/espnet@voxtlm_dev):
#
#   lm1_k1000_Bal       没有自己训 kmeans,直接用论文作者发布的官方 km_1000.mdl
#                       (HF soumi-maiti/voxtlm-k1000,md5 d4674452...)
#   这里(_ttskmeans)    自己训,只用 tts 的 10% train 子集,照搬上游的做法
#
# 上游的 lm.sh 是对 dump/audio_raw/ 下每个任务各跑一遍
# perform_kmeans.sh --stage 1 --stop-stage 4,但 stage 2 每次都写同一个
# ${km_dir}/km_1000.mdl,所以 asr -> speechlm -> tts 三次训练里,前两次的模型会
# 被后面的直接覆盖,真正留下来给下游用的只有最后一个 tts 的。既然如此就别白跑
# 那两次了:本地化的 lm.sh 里 Stage 3a 只训 tts 那一个(--kmeans_train_task tts),
# 然后拿它给 asr/tts/speechlm 三个任务统一打标签。learn_kmeans.py 是确定性的,
# 所以模型跟上游最终那个逐字节一致,只是省掉两轮特征抽取+拟合。
#
# 有意思的一点:官方 km_1000.mdl 很可能就是上游用同样的做法(tts 10% 子集)训出来
# 的,所以 Stage 3 跑完后可以拿这里的 km_1000.mdl 跟官方的比 md5,看能不能复现
# (下面 Stage 3 之后有一行只打印、不会因为不一致而报错的检查)。如果碰巧完全一致,
# 那 hubert_voc1(用官方 km_1000 标签训的 vocoder)就能直接拿来评这个 recipe 的 TTS;
# 不一致的话就得像 k2000 那边一样另训一个专属 vocoder。
#
# 目录复用情况(哪些是软链接、为什么安全):
#   downloads/*            -> 语料本体和已切好的 *_segmented,只读
#   data/{librispeech,librilight,libritts,vctk}
#                          -> lm1_k1000_Bal 的同名目录,Stage 1 的产物,之后只读
#   dump/audio_raw/*/*/data -> lm1_k1000_Bal dump 出来的 flac,只读,体积大
#
#   下面这些是本地真实文件/目录,绝不能共享:
#   data/{train,dev,test}  Stage 4 会往 data/train/bpe_text 写(train/dev/test 里的
#                          speech/text 清单是从 lm1_k1000_Bal 真实拷贝过来的)
#   data/en_token_list     Stage 5 的 BPE 模型和 token_list,每个实验都不一样
#   dump/audio_raw/*/*/    清单文件(wav.scp/utt2spk/...)是真实拷贝,来源是
#                          lm1_k1000_Bal(speechlm/train 已经是过滤掉 >120秒后的
#                          299,122 条)。这里 Stage 3 只跑到 perform_kmeans 的
#                          stage 4,往 datadir 写 text.km.* + 调 fix_data_dir.sh
#                          原地重写的是它的 stage 5,所以其实碰不到;真实拷贝是为了
#                          保险。之前 voxtlm_v2/lm1_k2000_Bal 把 data/ 软链接到 k1000,
#                          两边互相覆盖 bpe.model,把 k1000 训到一半的任务搞崩过,
#                          教训在此。
#                          (perform_kmeans 的 stage 1 会在这里建
#                           tts/train_subset0.1,也是本地目录)
#   dump/extracted, dump/raw, exp/
#                          标签、统计量、模型,全是本实验特有的(此刻都是空的,
#                          Stage 3 开始才会生成)
#
# 不需要跑 Stage 1(VAD切分+D_Bal抽样)和 Stage 2(wav dump):它们的产物就是
# 上面复制/链接过来的 data/ 和 dump/audio_raw。
#
# 也不要在这里跑 local/filter_long_utts.sh:拷过来的 speechlm/train 清单已经是
# 过滤后的 299,122 条,而且那个脚本是原地改清单的。

##############################################################################
# Stage 3: 训 kmeans(只用 tts),并给 asr/tts/speechlm 三个任务打标签。
#
# 内部依次做两件事:
#   1) 抽 tts train 的 10% 子集(约 39,866 条 / 1.07e7 帧)的 hubert_base 第6层
#      特征到 dump/extracted/tts/hubert_base/layer6/train_subset0.1,约 33GB,
#      普通 GPU 就够(cuda_cmd)。这些特征跟 k 无关(k2000 那边也抽过一份),
#      但为了不跟别的实验共享可写目录,这里独立重抽一遍,不复用。
#   2) 拟合 MiniBatchKMeans(1000 类):learn_kmeans.py 的 --percent -1 会把整个
#      子集读进内存(约 33GB+),普通分区给不了,所以只把这一步临时切到
#      conf/slurm_kmeans.conf(gpu-a100 节点借内存 200G,不占GPU),跑完用 trap
#      改回来。perform_kmeans.sh 里只有 learn_kmeans.py 用 cpu_cmd(=train_cmd),
#      打标签那两步用 cuda_cmd,所以 cuda_cmd 保持普通 GPU 不动。
#
# km_dir 不指定时 lm.sh 会自动用 exp/kmeans/hubert_base_6_1000clusters。
# 打完标签后可以跟 lm1_k1000_Bal(官方 kmeans)比一下:单元序列长度应该完全一样
# (都是50Hz一帧一个单元),但编号分布会不同 —— 这正是要看的东西。
#
# [注意] 要做 TTS 评测的话,vocoder 必须用这里的 k1000 标签训出来的。
# lm1_k1000_Bal 用的 hubert_voc1 是官方 km_1000 的标签训的、k2000 那两个 vocoder
# 是 2000 类的,在这里都不能直接用(除非上面 md5 检查显示模型跟官方完全一致)。
##############################################################################
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
trap 'sed -i "s#export train_cmd=\"slurm.pl --config conf/slurm_kmeans.conf\"#export train_cmd=\"slurm.pl\"#" cmd.sh' EXIT
sed -i 's#export train_cmd="slurm.pl"#export train_cmd="slurm.pl --config conf/slurm_kmeans.conf"#' cmd.sh
. ./cmd.sh
./run.sh --stage 3 --stop_stage 3 --nclusters 1000 --learn_kmeans true --kmeans_train_task tts
trap - EXIT
sed -i 's#export train_cmd="slurm.pl --config conf/slurm_kmeans.conf"#export train_cmd="slurm.pl"#' cmd.sh
. ./cmd.sh

# 只打印、不校验:自训的 km_1000.mdl 跟官方发布的是否逐字节一致。joblib/sklearn
# 版本不同也可能导致 pickle 内容不同,所以不一致不代表模型有问题,不要因此中断。
md5sum exp/kmeans/hubert_base_6_1000clusters/km_1000.mdl \
    /export/fs06/mliang17/shared_kmeans/hubert_base_l6_k1000_official/km_1000.mdl || true

##############################################################################
# Stage 4: 组装多任务文本;Stage 5: 用 Stage 4 产物训 BPE
# (sentencepiece unigram, nbpe=10000)。都是纯CPU、本地跑。
# nclusters 在这两步也要传,因为 lm.sh 用它拼 pseudo_labels 的文件名。
##############################################################################
./run.sh --stage 4 --stop_stage 5 --nclusters 1000

##############################################################################
# Stage 6: 统计信息收集(算 text_shape.bpe,给 Stage 7 的 batch sampler 用)。
##############################################################################
./run.sh --stage 6 --stop_stage 6 --nclusters 1000

# 过滤掉长度 >1500 的序列:离散语音单元序列没有天然句子边界,极少数超长序列
# 会让 Stage 7 OOM(self-attention 显存 O(L^2)),而且超过 OPT 的2048位置上限
# 会直接报 index out of bounds。备份原始文件,脚本本身幂等。
# 被砍掉的条数可以跟 lm1_k1000_Bal 对比(1,279,028 -> 1,276,237)。
F=exp/lm_stats_en_bpe10000/train/text_shape.bpe
[ -e "$F.orig" ] || cp "$F" "$F.orig"
awk '{split($2,a,","); if (a[1]<=1500) print}' "$F.orig" > "$F"

##############################################################################
# Stage 7: 训练语言模型,OPT-350M 初始化,跟 lm1_k1000_Bal 同一份配置,便于对比。
# 全流程只有这一步用 A100,跑完(或被手动叫停后)用 trap 把 cuda_cmd 改回
# 普通 GPU。训练不会自动停,要盯着 train.log 的 valid acc 判断过拟合。
##############################################################################
trap 'sed -i "s#export cuda_cmd=\"slurm.pl --config conf/slurm_clsp_a100.conf\"#export cuda_cmd=\"slurm.pl\"#" cmd.sh' EXIT
sed -i 's#export cuda_cmd="slurm.pl"#export cuda_cmd="slurm.pl --config conf/slurm_clsp_a100.conf"#' cmd.sh
. ./cmd.sh
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1
./run.sh --stage 7 --stop_stage 7 --ngpu 1 --nclusters 1000 \
    --lm_config conf/tuning/train_transformer_opt350.yaml
