#!/usr/bin/env bash
#SBATCH --job-name=v1_k2k3M_ttskm
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_3M_ttskmeans/log/myrun.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_3M_ttskmeans/log/myrun.%j.err

set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_3M_ttskmeans

. ./path.sh
. ./cmd.sh

# 这份 recipe 是 lm1_k2000_Bal_ttskmeans 的 D_3M 版本:hubert_base 第6层 + 只用
# tts 10% train 子集训 kmeans(照搬上游做法)不变,数据规模从 D_Bal(speechlm/
# textlm 各30万)换成 D_3M(各300万,论文 Table 2)。
#
# 目录复用情况(哪些是软链接、为什么安全):
#   downloads/*             -> 语料本体和已经切好的 *_segmented,只读
#   data/{librilight,librispeech,libritts,vctk}
#                           -> voxtlm_v2/lm1_k2000_3M 的同名目录(它自己跑
#                              Stage1 子阶段1 的产物,3M抽样的源头),之后只读
#   dump/audio_raw/*/*/data -> voxtlm_v2/lm1_k2000_3M dump 出来的 flac,只读,
#                              体积大(主要是 tts,asr/speechlm 的 wav.scp 直接
#                              指向 downloads/ 下的原始文件,data/ 子目录本身没
#                              什么内容)
#   exp/kmeans/hubert_base_6_2000clusters
#                           -> lm1_k2000_Bal_ttskmeans 已经训好的 kmeans,见下面
#                              Stage 3 的说明,只读复用,不重新拟合
#
#   下面这些是本地真实文件/目录,绝不能共享(教训见 v1 k1000_Bal 的 BPE 被
#   共享 data/ 覆盖事故):
#   data/{train,dev,test}   Stage 4 会往 data/train/bpe_text 写(train/dev/test
#                           里的 speech/text 清单已经是从 v2/lm1_k2000_3M 真实
#                           拷贝过来的 3M 抽样结果,dev/test 都是全量、不受
#                           Bal/3M 影响)
#   data/en_token_list      Stage 5 的 BPE 模型和 token_list,每个实验都不一样
#   dump/audio_raw/*/*/     清单文件(wav.scp/utt2spk/...)是真实拷贝,来源是
#                           v2/lm1_k2000_3M。speechlm/train 用的是它 Stage 3a
#                           自带的 max_wav_duration=120 过滤后的版本(dump/
#                           audio_raw_filtered,3,000,000 -> 2,990,192),过滤
#                           标准(样本数/16000<=120)跟 v1 这边用的
#                           local/filter_long_utts.sh 完全一样,不需要在这里
#                           再跑一遍。asr/tts train 不受 D_Bal/D_3M 影响,是
#                           全量(281,241 / 398,665,跟 Bal ttskmeans 那边逐字节
#                           一致,已核对)。
#   dump/extracted, dump/raw, exp/lm_*
#                           标签、统计量、模型,全是本实验特有的
#
# 不需要跑 Stage 1(VAD切分+D_3M抽样)和 Stage 2(wav dump):它们的产物就是
# 上面复制/链接过来的 data/ 和 dump/audio_raw,voxtlm_v2/lm1_k2000_3M 已经
# 跑完了(那次跑了约31.5小时,主要是 LibriLight large 的清单整理慢)。
#
# 也不要在这里跑 local/filter_long_utts.sh:拷过来的 speechlm/train 清单已经
# 是过滤后的 2,990,192 条。

##############################################################################
# Stage 3: 给 asr/tts/speechlm 三个任务打离散单元标签,不重新训 kmeans。
#
# 关键点:tts/train 这个任务本身不受 D_Bal/D_3M 影响(论文设计里只有 speechlm/
# textlm 的 train 切分会变,asr/tts 和所有 dev/test 全部保持全量),已经核对过
# 这里的 dump/audio_raw/tts/train/{wav.scp,utt2num_samples} 跟
# lm1_k2000_Bal_ttskmeans 的逐字节相同。kmeans 只在 tts 的 10% train 子集上训
# (--kmeans_train_task tts),输入完全一样,learn_kmeans.py 又是确定性的,所以
# 重新拟合出来的模型必然跟 lm1_k2000_Bal_ttskmeans 那份逐字节一致 —— 与其真的
# 再花约40分钟+提约33GB特征重新拟合一遍,不如直接只读复用它已经训好的模型:
#   --km_dir ../lm1_k2000_Bal_ttskmeans/exp/kmeans/hubert_base_6_2000clusters
#   --learn_kmeans false
# (exp/kmeans/hubert_base_6_2000clusters 本身已经软链接到那份目录,见上面的
# 目录复用说明;这里传绝对路径给 --km_dir 只是让 lm.sh 自己也能找到,不依赖
# 相对路径解析)
#
# 只用普通 gpu(cuda_cmd 默认值),不需要临时切 A100 —— 那一步只有真正拟合
# kmeans(learn_kmeans.py 把全部特征读进内存)才需要,这里跳过了拟合,只做
# 特征提取+打标签。
#
# 3M 规模下 speechlm/train 有 2,990,192 条要打标签,比 Bal 的 299,122 条多约
# 10 倍,预计这一步会明显更慢。
##############################################################################
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
KM_DIR=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans/exp/kmeans/hubert_base_6_2000clusters
md5sum exp/kmeans/hubert_base_6_2000clusters/km_2000.mdl  # 只打印,确认复用的模型确实在(软链接指向 Bal ttskmeans)
./run.sh --stage 3 --stop_stage 3 --nclusters 2000 --learn_kmeans false --km_dir "${KM_DIR}"

##############################################################################
# Stage 4: 组装多任务文本;Stage 5: 用 Stage 4 产物训 BPE
# (sentencepiece unigram, nbpe=10000)。都是纯CPU、本地跑,3M 规模下语料量比
# Bal 大约10倍,BPE 训练本身会明显变慢(Bal 那次单独约1小时)。
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
F=exp/lm_stats_en_bpe10000/train/text_shape.bpe
[ -e "$F.orig" ] || cp "$F" "$F.orig"
awk '{split($2,a,","); if (a[1]<=1500) print}' "$F.orig" > "$F"

##############################################################################
# Stage 7: 训练语言模型,OPT-350M 初始化,跟 lm1_k2000_Bal_ttskmeans 同一份配置,
# 便于对比 Bal vs 3M 的效果。
#
# 全流程只有这一步用 A100,跑完(或被手动叫停后)用 trap 把 cuda_cmd 改回
# 普通 GPU。训练不会自动停,要盯着 train.log 的 valid acc 判断过拟合。
# num_iters_per_epoch 是配置里固定的(不因数据池变大而变),所以每轮训练本身
# 的耗时跟 Bal 差不多,只是同样多的epoch能看到更多不同的数据、理论上不容易
# 像小数据那样过拟合。
##############################################################################
trap 'sed -i "s#export cuda_cmd=\"slurm.pl --config conf/slurm_clsp_a100.conf\"#export cuda_cmd=\"slurm.pl\"#" cmd.sh' EXIT
sed -i 's#export cuda_cmd="slurm.pl"#export cuda_cmd="slurm.pl --config conf/slurm_clsp_a100.conf"#' cmd.sh
. ./cmd.sh
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1
./run.sh --stage 7 --stop_stage 7 --ngpu 1 --nclusters 2000 \
    --lm_config conf/tuning/train_transformer_opt350.yaml
