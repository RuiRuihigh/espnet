#!/usr/bin/env bash
#SBATCH --job-name=v2_k2000_3M
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_3M/log/myrun.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_3M/log/myrun.%j.err

set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_3M

. ./path.sh
. ./cmd.sh

# 从 ../lm1_k2000_Bal 复制出来的:音系学 tokenizer(微调过的 WavLM-large
# 第21层 + Sony 发布的固定 2000-entry codebook)不变,数据规模从 D_Bal
# (speechlm/textlm 各30万)换成 D_3M(各300万,论文 Table 2)。
# 这个 sbatch job 本身是 cpu 分区的控制脚本,真正的重活靠
# train_cmd/cuda_cmd(slurm.pl)再往外提交子任务、控制脚本阻塞等它们跑完
# 才往下走,所以下面这些 stage 严格顺序执行。cpu 分区没有时间上限。
#
# GPU 策略:全程默认普通 gpu 分区,只有 Stage 7 训练那一步临时切到 A100
# (trap 保证不管成功/失败/中断都会改回来)。跟这份实验其它三个已经跑完的
# recipe(v1 k1000_Bal、v1 k2000_Bal、v2 k2000_Bal)一致。

# Stage 3 WavLM-large 的前向没有分块,单条 ~120s 的音频要约2GB连续显存,
# 在10.9GB的卡上容易因为碎片化 OOM(v2 Bal 第一次跑 16 个分片里 14 个因为
# "reserved but unallocated"碎片化失败)。这个环境变量緩解碎片化。
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

##############################################################################
# Stage 1: 数据准备,--local_data_opts "--data_config 3m" 把
# speechlm_train_size/textlm_train_size 都设成300万(对应论文 D_3M)。
# LibriLight small/medium/large 的 VAD 切分(local/data.sh 内部子阶段1)
# 复用 downloads/*_segmented 里已经切好的结果(软链接到
# v1/lm1_k1000_Bal,已验证三边 cut_by_vad.py md5一致、有"已切过就跳过"
# 保护,不会真的重新切或者覆盖共享数据),真正会花时间的是子阶段2的300万
# 条抽样(比 Bal 的30万条量大10倍)。
#
# [已跑完/待跑,见下方状态标记,跑完后把这行改成注释并记录结果]
##############################################################################
./run.sh --stage 1 --stop_stage 1 --local_data_opts "--data_config 3m"

##############################################################################
# Stage 2: wav dump(把 data/ 里的 wav.scp 重新格式化/采样率转换,产物落在
# dump/audio_raw/)。纯 CPU,量比 Bal 大但结构不变。
##############################################################################
./run.sh --stage 2 --stop_stage 2

##############################################################################
# Stage 3: 用音系学 tokenizer 给 asr/tts/speechlm 打离散单元标签。
# 用的是固定发布的 codebook(exp/kmeans/phonological_tokenizer_2000clusters,
# 软链接自 Bal,不是在我们数据上训练的),没有 kmeans 训练这一步,也不用
# A100,跟 Bal 当时一样走普通 gpu 分区。run.sh 里已经固定
# max_wav_duration=120,会在这一步之前自动过滤掉过长的 speechlm/train
# 音频,不需要像 v1 k1000_Bal 那样额外跑 filter_long_utts.sh。
# 按任务(train/dev/test)做了幂等标记(md5指纹),重跑会跳过已经打完标签的。
##############################################################################
./run.sh --stage 3 --stop_stage 3

##############################################################################
# Stage 4: 组装多任务文本;Stage 5: 用 Stage 4 产物训 BPE
# (sentencepiece unigram, nbpe=10000,特殊token走data/nlsyms.txt保留不参与
# 合并)。两步都是纯CPU、本地跑。Bal 那次 BPE 单独就要约1小时,3M 数据量大
# 10倍,预计会更久。
##############################################################################
./run.sh --stage 4 --stop_stage 5

##############################################################################
# Stage 6: 统计信息收集(算 text_shape.bpe 等,给 Stage 7 的 batch sampler用)。
##############################################################################
./run.sh --stage 6 --stop_stage 6

# 离散语音单元序列没有天然句子边界,极少数超长序列会让 Stage 7 在 A100 上
# OOM(self-attention显存 O(L^2),batch_bins只按token数线性分桶)。过滤掉
# 长度>1500的,备份原始文件以防误操作(脚本本身幂等)。
F=exp/lm_stats_en_bpe10000/train/text_shape.bpe
[ -e "$F.orig" ] || cp "$F" "$F.orig"
awk '{split($2,a,","); if (a[1]<=1500) print}' "$F.orig" > "$F"

##############################################################################
# Stage 7: 训练语言模型。用 OPT-350M 初始化(论文报告的主结果配置)。
#
# 这是全流程里唯一真正需要大显存、该用 A100 的地方,cuda_cmd 平时(Stage1-6)
# 都保持默认的普通GPU,只在这一步临时升级到 A100,跑完(或被手动叫停后)
# 用 trap 改回默认。
#
# HF_HOME 指向复用的 hf_cache/(已缓存 facebook/opt-350m),配合
# HF_HUB_OFFLINE=1 保证计算节点没有外网也能跑。这一步是真·训练,会跑很久,
# 数据量是 Bal 的10倍,不代表训练轮数(epoch)也变多——num_iters_per_epoch
# 是配置里固定的,每轮吃到的数据量不变,只是同样多的epoch能看到更多不同的
# 数据。到了合适的时候要手动看 train.log 的 train/valid acc 曲线判断要不要
# 提前叫停,不会自动停。
##############################################################################
trap 'sed -i "s#export cuda_cmd=\"slurm.pl --config conf/slurm_clsp_a100.conf\"#export cuda_cmd=\"slurm.pl\"#" cmd.sh' EXIT
sed -i 's#export cuda_cmd="slurm.pl"#export cuda_cmd="slurm.pl --config conf/slurm_clsp_a100.conf"#' cmd.sh
. ./cmd.sh
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1
./run.sh --stage 7 --stop_stage 7 --ngpu 1 --lm_config conf/tuning/train_transformer_opt350.yaml
