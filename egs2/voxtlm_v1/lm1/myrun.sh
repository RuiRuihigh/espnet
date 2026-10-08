#!/usr/bin/env bash
#SBATCH --job-name=stage9
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1/log/myrun.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1/log/myrun.%j.err

set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1

. ./path.sh
. ./cmd.sh

# Stage 1: 数据准备(librispeech/libritts/vctk/librilight 各自准备 + 合并成 data/train,dev,test)
# downloads/ 下的语料已经软链接到老 checkout,标记文件都在,不会重新下载
# [已跑完 job 1791616,exit 0,data/train,dev,test 都生成好了,不用重跑]
# ./run.sh --stage 1 --stop_stage 1

# D_Bal 抽样:从已经合并好的全量池子(speechlm 1736万条 / textlm 4042万条)里
# 随机抽 30万条,复现论文 D_Bal 配置。只是对已有 wav.scp/text 做 shuf+head+
# subset_data_dir.sh,纯本地CPU操作,不用GPU,几分钟内能跑完。
# [已跑完 job 1793936,exit 0,speechlm/textlm train 都精确300000条,不用重跑]
# ./local/data.sh --stage 2 --stop_stage 2 --data_config bal

# Stage 2: wav dump(重新格式化各个 data 目录里的 wav.scp)
# 环境已确认 resampy 0.4.3 / h5py 3.16.0 都在共享 venv 里,不缺依赖。
# 这个 sbatch job 本身只是控制脚本,真正的重活是靠 train_cmd(slurm.pl)
# 再往外提交 nj=32 的并行子任务,不用在这里加大资源。
# [已跑完 job 1793949,exit 0,dump/audio_raw/{asr,tts,speechlm}/{train,dev,test}
#  都生成好了,条数跟 data/ 里的对得上,不用重跑]
# ./run.sh --stage 2 --stop_stage 2

# Stage 3: kmeans 聚类 + 给 asr/tts/speechlm 三个任务打离散单元标签
# (hubert_base layer6, nclusters=1000, 跟老 checkout 那份配置一致)
#
# [踩过的坑,记录一下]:run.sh 内部在调用 perform_kmeans.sh 之前还有个
# "建平衡采样池子"的前置步骤,之前想抄近道直接手动调 perform_kmeans.sh
# 跳过了这步,结果读到一个还没生成的池子目录直接报错(job 1794395)。
#
# [这次的做法]:查过 perform_kmeans.sh 源码——因为咱们 --use_gpu 恒为
# true,脚本里所有要GPU的步骤(特征提取、打标签)都走 cuda_cmd --gpu 1,
# cpu_cmd(=train_cmd)全脚本只在拟合kmeans那一行(learn_kmeans.py,吃
# ~120GB内存的元凶)单独被用到,而且那行不带 --gpu。所以临时把 train_cmd
# 指向大内存的 conf/slurm_kmeans.conf,只会精准影响这一个内存瓶颈点。
#
# [踩过的第二个坑]:第一次提交(job 1794424)时 cuda_cmd 还是默认的
# conf/slurm_clsp_a100.conf,导致 hubert_base 特征提取(dump_features.sh,
# nj=16数组任务)全部跑去了 gpu-a100 分区——hubert_base 才95M参数,几GB
# 显存就够,根本不需要A100,而且直接把A100配额占满卡住了(16个子任务里
# 大半个 PENDING QOSMaxGRESPerUser)。已经 scancel 掉重来。这次把
# cuda_cmd 也临时降级到普通 gpu 分区(跟 README 里 Stage 8-10 临时切分区
# 是同一个套路),A100 留给后面真正需要大显存的 OPT-350M 训练用。
#
# bash 的 trap ... EXIT 后注册的会覆盖前一条,不会叠加,所以两个改回去的
# sed 合并写在同一条 trap 里,保证不管成功/失败/中断,退出前都恢复原样。
#
# [已跑完 job 1794455,exit 0,耗时7h23m。km_1000.mdl 生成好了,trap 也
#  确认正常把 cmd.sh 恢复原样了。asr/tts/speechlm 三个任务的 token 标签
#  条数(281241/398665/300000)都跟 wav.scp 对得上,不用重跑]
# trap '
# sed -i "s#export train_cmd=\"slurm.pl --config conf/slurm_kmeans.conf\"#export train_cmd=\"slurm.pl\"#" cmd.sh
# sed -i "s#export cuda_cmd=\"slurm.pl\"#export cuda_cmd=\"slurm.pl --config conf/slurm_clsp_a100.conf\"#" cmd.sh
# ' EXIT
# sed -i 's#export train_cmd="slurm.pl"#export train_cmd="slurm.pl --config conf/slurm_kmeans.conf"#' cmd.sh
# sed -i 's#export cuda_cmd="slurm.pl --config conf/slurm_clsp_a100.conf"#export cuda_cmd="slurm.pl"#' cmd.sh
# . ./cmd.sh
# ./run.sh --stage 3 --stop_stage 3

# Stage 4: 组装多任务文本(assemble train/dev/test 各任务的文本,生成
# data/train/lm_text 等,供 Stage 5 训 BPE 用)。
# Stage 5: 用 Stage 4 产物训 BPE(sentencepiece unigram,nbpe=10000,
# 特殊 token 走 data/nlsyms.txt 保留不参与合并)。
# 两步都是纯 CPU、本地跑,不用 GPU,数据量不大(bpe_text 只有 asr+tts
# 68万行),预计几分钟内跑完,合并一次提交。
# [已跑完 job 1794988,exit 0,耗时56分钟。lm_text(train)1,279,906条
#  精确等于 300000+300000+281241+398665,tokens.txt 正好10000条,不用重跑]
# ./run.sh --stage 4 --stop_stage 5

# Stage 6: 统计信息收集(算 text_shape.bpe 等,给 Stage 7 的 batch sampler
# 用)。纯 CPU,数据量不大,预计几分钟内跑完。
# [已跑完,exit 0,不用重跑]
# ./run.sh --stage 6 --stop_stage 6

# [README 踩过的坑]:Stage 6 出来的 exp/lm_stats_en_bpe10000/train/text_shape.bpe
# 里有极少数超长序列(离散语音单元序列没有天然的句子边界,VAD 切分不当
# 可能留下几万token长的片段),self-attention 显存是 O(L^2),batch_type=length
# 的 batch_bins 只是按 token 数线性分桶,完全没考虑这个二次方开销,不过滤
# 直接训 Stage 7 在 A100 上也会 OOM。这里过滤掉长度>1500的(约丢0.4%,
# 可忽略),备份一份原始的以防误操作(脚本本身幂等,重跑不会覆盖备份)。
# [已跑完,不用重跑]
# F=exp/lm_stats_en_bpe10000/train/text_shape.bpe
# [ -e "$F.orig" ] || cp "$F" "$F.orig"
# awk '{split($2,a,","); if (a[1]<=1500) print}' "$F.orig" > "$F"

# Stage 7: 训练语言模型。用 OPT-350M 初始化(论文报告的主结果配置,不是
# from-scratch消融基线)。
# [手动叫停,job 1795182(控制脚本)+1795218(A100训练本体)已 scancel]
# 跑到 epoch 22(连续跑了1天6小时),拉 train.log 逐epoch的 train/valid
# acc/loss 发现:valid_acc 在 epoch 12-13 见顶(0.295),valid_loss 同步
# 见底(2.741),之后连续9个epoch(14->22)valid_acc掉到0.290、valid_loss
# 持续爬升到2.803,而train_acc全程单调上升到0.462没有放缓——典型过拟合,
# D_Bal(30万条)这个规模的数据集在epoch12-13左右过拟合是合理的(train.yaml
# 里 max_epoch=250 是给论文全量D_Set规模调的)。ESPnet自己的
# valid.acc.best.pth 软链接已经正确指向 13epoch.pth,不用手动挑。
# export HF_HOME="${PWD}/hf_cache"
# export HF_HUB_OFFLINE=1
# ./run.sh --stage 7 --stop_stage 7 --ngpu 1 --lm_config conf/tuning/train_transformer_opt350.yaml

# [踩过的坑,第一次提交(job 1797956)时忘了照搬 Stage 3 那次的原则:
# A100 只留给真正需要大显存的训练(Stage 7),推理/解码这种不需要大显存
# 的步骤应该走普通 gpu 分区。Stage 9 的 beam search 解码(lm_inference.sh)
# 默认走 cuda_cmd(=A100),已经卡在 QOSMaxGRESPerUser 配额上了,scancel
# 重来。Stage 8 那次自己就走了普通gpu分区(原因不明,可能用的是别的cmd
# 变量),这次干脆把 cuda_cmd 也临时降级,两边都保险。用法跟 Stage 3 一样:
# 临时改 cmd.sh、trap 保证退出前改回去。]
trap 'sed -i "s#export cuda_cmd=\"slurm.pl\"#export cuda_cmd=\"slurm.pl --config conf/slurm_clsp_a100.conf\"#" cmd.sh' EXIT
sed -i 's#export cuda_cmd="slurm.pl --config conf/slurm_clsp_a100.conf"#export cuda_cmd="slurm.pl"#' cmd.sh
. ./cmd.sh

# Stage 8: 用最优 checkpoint(13epoch,即 valid.acc.best.pth)算 textLM/
# speechLM 的困惑度(PPL)。必须显式传 --inference_lm valid.acc.best.pth——
# 默认的 valid.acc.ave.pth 只有训练跑到 max_epoch 或触发早停才会生成,
# 咱们是手动提前停的,不会有这个文件。这一步是单条前向推理(batch_size=1,
# fp32),README 里提过不需要A100,11GB的普通GPU就够。
# [已跑完,exit 0(job 1798010里跑成功的那部分),结果:
#  textLM PPL=27.33, speechLM(unitLM) PPL=50.86,不用重跑]
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1
# ./run.sh --stage 8 --stop_stage 8 --ngpu 1 --lm_config conf/tuning/train_transformer_opt350.yaml --inference_lm valid.acc.best.pth

# Stage 9: ASR 解码 + 算 WER。跟 Stage 8 是完全分开的两个 stage,同样要
# 显式传 --inference_lm valid.acc.best.pth(理由同上)。用的是真·beam
# search 自回归解码(不是Stage8那种单次前向),默认 --inference_nj 8,
# 负载比训练小得多,同样不需要A100。
#
# [踩过的坑]:第一次提交(job 1798010)8个解码子任务里6个都分到了节点c15,
# 清一色报"CUDA unknown error"(CUDA初始化失败),另外2个分到c17/c20的
# 顺利跑完——不是分区路由问题,是c15这台机器GPU当前有毛病(SLURM自己还
# 没标记down)。已经把c15加进 conf/slurm.conf 的gpu排除列表,这次重跑。
./run.sh --stage 9 --stop_stage 9 --ngpu 1 --lm_config conf/tuning/train_transformer_opt350.yaml --inference_lm valid.acc.best.pth

# [注意]:Stage 9 自己跑出来的 sclite 打分是 test-clean+test-other 合并
# 算的一个 WER,但论文(Table 3/7)是分开报 test-clean / test-other 的。
# 不用重新解码,直接用刚生成的 hyp.trn/ref.trn 按 uttid 列表拆开重新跑
# sclite 就行(纯本地计算,不占GPU,秒级完成)。
SCOREDIR=exp/lm_train_transformer_opt350_en_bpe10000/decode_test_asr/decode_lm_asr/score_wer
mkdir -p "${SCOREDIR}/clean" "${SCOREDIR}/other"
awk '{print "(asr_"$1")"}' data/librispeech/asr/test_clean/text | sort > "${SCOREDIR}/test_clean_ids.txt"
awk '{print "(asr_"$1")"}' data/librispeech/asr/test_other/text | sort > "${SCOREDIR}/test_other_ids.txt"
for split in clean other; do
    idfile="${SCOREDIR}/test_${split}_ids.txt"
    grep -F -f "${idfile}" "${SCOREDIR}/hyp.trn" > "${SCOREDIR}/${split}/hyp.trn"
    grep -F -f "${idfile}" "${SCOREDIR}/ref.trn" > "${SCOREDIR}/${split}/ref.trn"
    sclite -r "${SCOREDIR}/${split}/ref.trn" trn -h "${SCOREDIR}/${split}/hyp.trn" trn \
        -i rm -o all stdout > "${SCOREDIR}/${split}/result.txt"
done
