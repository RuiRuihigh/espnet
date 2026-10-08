#!/usr/bin/env bash
#SBATCH --job-name=lm1_3m
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_3m/log/myrun.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_3m/log/myrun.%j.err

set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_3m

. ./path.sh
. ./cmd.sh

# ============================================================================
# D_3M 配置(VoxtLM 论文 Table 2):speechlm=3000000 / textlm=3000000
# 这份 recipe 是从 lm1(D_Bal,30万)复制出来的,只复制了必要的东西:
#
#   [软链接复用,不会改动的大件]
#     downloads/*                   外部语料,只读
#     data/{librispeech,libritts,vctk,librilight}  Stage 1 产物(7.9G),
#                                   各语料全量池子,跟 data_config 无关
#     dump/audio_raw/tts            LibriTTS 重采样后的 flac(52G)。tts 不参
#                                   与 --data_config 抽样,内容完全一致
#     dump/extracted/{asr,tts}      asr/tts 的离散单元标签(1.1G/716M),同上
#     ckpt                          老 checkout 的 ckpt(目前没有任何脚本引用它,
#                                   纯粹是跟着 lm1 一起带过来的)
#
#   [复制过来的小件]
#     hf_cache/                     facebook/opt-350m 的 HF 缓存(1.9G)。故意
#                                   *复制*而不是软链接:Stage 7 会把它当 HF_HOME,
#                                   transformers/huggingface_hub 即使在 OFFLINE
#                                   模式下也可能往缓存目录写 .locks 之类的东西,
#                                   复制一份就彻底不会碰到别的 checkout
#     exp/kmeans/                   已经拟合好的 km_1000.mdl(4.8M)。故意复用
#                                   同一个 kmeans 模型,保证 Bal/3M 两组实验用
#                                   的是同一套离散单元词表,数据量才是唯一变量
#     data/en_token_list/bpe_unigram10000/{bpe.model,bpe.vocab,tokens.txt}
#                                   BPE 也复用:local/prepare_bpe_text.py 只挑
#                                   lm_text 里 asr/tts 开头的行,而 asr/tts 不
#                                   受 --data_config 影响,所以 BPE 本来就一样
#                                   (train.txt 947M 是 spm 的训练输入副本,跳过
#                                    Stage 5 就用不到,没复制)
#
#   [需要重跑的]
#     data/{train,dev,test}、dump/audio_raw/speechlm、dump/extracted/speechlm、
#     dump/raw、exp/lm_stats_*、exp/lm_train_*
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1(local/data.sh stage 1):各语料数据准备。
# 不用跑 —— data/{librispeech,libritts,vctk,librilight} 已经软链接到 lm1 的产物。
# 注意:这几个是软链接,千万别在这里跑 Stage 1,会写穿到 lm1 里去。
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# D_3M 抽样:从全量池子里抽 300万条 speechlm / 300万行 textlm,生成
# data/{train,dev,test}。纯本地 CPU(shuf+head+subset_data_dir.sh),
# dev/test 保持全量,只有 train 变大。比 bal 慢一些但也就十几分钟量级。
# ---------------------------------------------------------------------------
./local/data.sh --stage 2 --stop_stage 2 --data_config 3m

# ---------------------------------------------------------------------------
# Stage 2:wav dump(重新格式化 data/*/speech/* 里的 wav.scp -> dump/audio_raw)。
#
# [必读]dump/audio_raw/tts 是软链接指向 lm1 的 52G flac。lm.sh 的 Stage 2 是
# 无条件重跑的(format_wav_scp.sh 里没有任何"已存在就跳过"的逻辑),直接跑会
# 顺着软链接把 lm1 的目录整个重写一遍 —— 内容虽然一样,但白白烧 52G IO,而且
# 在 lm1 还在训练的时候去动它的目录很不干净。
#
# 所以这里先把三个 split 的 tts 数据目录临时挪走,让 Stage 2 的循环扫不到 tts,
# 只处理 asr 和 speechlm;trap 保证不管成功/失败/被打断,退出前都挪回来
# (Stage 3/4 还要用 data/*/speech/tts)。这是 lm1 里 Stage 3 改 cmd.sh 用过的
# 同一个套路。
#
# asr 的 wav.scp 只是指回 downloads/,不涉及重编码,本地重跑很便宜(83M),
# 就不搞特殊处理了,让 lm1_3m 自己有一份。
# ---------------------------------------------------------------------------
unhide_tts() {
    for d in train dev test; do
        if [ -d "data/${d}/speech/.tts_hidden" ]; then
            rm -rf "data/${d}/speech/tts"
            mv "data/${d}/speech/.tts_hidden" "data/${d}/speech/tts"
        fi
    done
}
# 也捕获 HUP/INT/TERM:bash 收到未捕获的信号时不会跑 EXIT trap,如果被 scancel
# 掉而 tts 还藏着,Stage 3 会因为扫不到 data/*/speech/tts 而整个漏掉 tts 任务。
trap unhide_tts EXIT HUP INT TERM
for d in train dev test; do
    if [ -d "data/${d}/speech/tts" ]; then
        mv "data/${d}/speech/tts" "data/${d}/speech/.tts_hidden"
    fi
done
./run.sh --stage 2 --stop_stage 2
unhide_tts
trap - EXIT

# ---------------------------------------------------------------------------
# Stage 3:用已有的 kmeans 模型给语音打离散单元标签。
#
# --learn_kmeans false:不重新拟合 kmeans。lm.sh 在 km_dir 为空时默认指向
# exp/kmeans/hubert_base_6_1000clusters/,那份 km_1000.mdl 已经从 lm1 复制过来
# (md5 = d6332e8e94092dd7d7b9718852f25ee0)。这样既省掉 7 小时+的拟合(还有
# 那个吃 ~120GB 内存的 learn_kmeans.py),也保证跟 D_Bal 用同一套 token 词表。
#
# 因为 md5 一致,dump/extracted/{asr,tts}/.labeled_km1000_<md5> 这两个标记
# 文件会被认出来,asr/tts 整个任务直接跳过,不会写进软链接的目录 —— 实际只有
# speechlm 的 300万条需要抽特征+打标签,这是这一步真正的开销(GPU,很久)。
#
# cuda_cmd 现在默认就是 conf/slurm.conf 的普通 gpu 分区(见本 recipe 自己那份
# cmd.sh 的改动)。hubert_base 才 95M 参数,几 GB 显存就够,不用碰 A100 ——
# lm1 里第一次提交(job 1794424)就是因为默认跑去 gpu-a100,16 个子任务大半卡在
# QOSMaxGRESPerUser PENDING,得 scancel 重来。A100 留给 Stage 7。
# ---------------------------------------------------------------------------
./run.sh --stage 3 --stop_stage 3 --learn_kmeans false

# ---------------------------------------------------------------------------
# Stage 4:组装多任务文本(生成 dump/raw/{train,dev,test}/lm_text 等)。
#
# [为什么没有 Stage 5]:Stage 5 做的事只有三件(见 lm.sh 该段):把 bpe_text
# 切出 ${bpedir}/train.txt、跑 spm_train 生成 bpe.model/bpe.vocab、再从 vocab
# 生成 tokens.txt。这三样已经从 lm1 复制过来了:
#   data/en_token_list/bpe_unigram10000/{bpe.model,bpe.vocab,tokens.txt}
# 而 BPE 的训练语料 bpe_text 只含 lm_text 里 asr/tts 开头的行(见
# local/prepare_bpe_text.py),asr/tts 不受 --data_config 影响,所以 bal/3m/set
# 三种配置训出来的 BPE 本来就是同一个 —— 重训只是花时间得到一样的结果,
# 而且会让两组实验的 token 词表存在"理论上可能不一致"的风险。
# 下游只用到 ${bpemodel}(=bpe.model)和 ${token_list}(=tokens.txt),
# Stage 6/7 都齐了;train.txt 只是 spm 的输入副本,跳过 Stage 5 就用不到。
# ---------------------------------------------------------------------------
./run.sh --stage 4 --stop_stage 4

# ---------------------------------------------------------------------------
# Stage 6:统计信息收集(text_shape.bpe,给 Stage 7 的 batch sampler 用)。
# 数据量是 bal 的 ~4 倍(lm_text 从 128万条涨到 ~660万条),会比 lm1 那次慢。
# ---------------------------------------------------------------------------
./run.sh --stage 6 --stop_stage 6

# [沿用 lm1 踩过的坑]:离散语音单元序列没有天然句子边界,VAD 切分不当会留下
# 几万 token 的超长片段;self-attention 显存是 O(L^2),而 batch_type=length 的
# batch_bins 只按 token 数线性分桶,不过滤直接训在 A100 上也会 OOM。
# 过滤掉长度 >1500 的(bal 那次约丢 0.4%),备份原始文件以防误操作。
F=exp/lm_stats_en_bpe10000/train/text_shape.bpe
[ -e "$F.orig" ] || cp "$F" "$F.orig"
awk '{split($2,a,","); if (a[1]<=1500) print}' "$F.orig" > "$F"

# ---------------------------------------------------------------------------
# Stage 7:训练语言模型,OPT-350M 初始化(论文主结果配置)。
# HF_HOME 指向本目录自己那份 hf_cache/(facebook/opt-350m 权重已缓存),
# 配合 HF_HUB_OFFLINE=1 保证计算节点没外网也不会去联网下载。
# 数据量是 bal 的 ~4 倍,每个 epoch 会长很多,注意 checkpoint 和过拟合拐点
# (bal 那次最优在 17 epoch)。
# ---------------------------------------------------------------------------
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1
# 只有这一步切到 A100:OPT-350M 真训练需要大显存;而且 lm.sh:822 会给 cuda_cmd
# 传 --name,conf/slurm.conf 里没有 "option name=*",slurm.pl 会直接 die,
# 只有 conf/slurm_clsp_a100.conf 描述了这个 option。
export CUDA_CMD_CONF=conf/slurm_clsp_a100.conf
./run.sh --stage 7 --stop_stage 7 --ngpu 1 --lm_config conf/tuning/train_transformer_opt350.yaml
