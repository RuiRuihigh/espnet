#!/usr/bin/env bash
#SBATCH --job-name=lm1_k1000_Bal
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal/log/myrun.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal/log/myrun.%j.err

set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal

. ./path.sh
. ./cmd.sh

# 这份 recipe 的定位:lm1 那次(D_Bal, k=1000)因为 downloads/*_segmented 被
# 另一次运行(用了后来才加的"长VAD片段强制切块"修复)意外覆盖,speechLM 的
# 切分粒度跟论文/历史结果不可比,而且这次已经把 local/librilight/cut_by_vad.py
# 改回旧版逻辑并加了"已切过就跳过"的幂等保护。这是一次干净重跑,从 Stage 1
# 一路跑到 Stage 7(训练)。这个 sbatch job 本身是 cpu 分区的控制脚本,真正的
# 重活都是靠 train_cmd/cuda_cmd(slurm.pl)再往外提交子任务、控制脚本阻塞
# 等它们跑完才往下走——所以下面这些 stage 是严格顺序执行的,前一个不完全
# 跑完,不会开始下一个。cpu 分区没有时间上限,可以放心跑完整条流水线
# (大概率要跑好几天,large 那部分 VAD 切分本身就要约30小时)。

##############################################################################
# Stage 1: 数据准备,直接把 --data_config bal 通过 --local_data_opts 一次性
# 传进去(而不是先跑全量、再单独补一刀)。注意:LibriLight small/medium/large
# 的 VAD 切分本身(local/data.sh 内部子阶段1)不受 --data_config 影响,还是
# 要完整跑一遍全量语料(large 那部分约30小时),--data_config bal 只在子阶段2
# 合并+抽样的时候起作用,省的是"先把17M+条全量池子落盘、再读出来重新抽样"
# 这一轮冗余I/O,不是省掉真正耗时的VAD切分本身。
#
# [已跑完,job 1799638。第一次提交 job 1799359 卡在 LibriLight 切分子任务
#  排不上队(local/data_librilight.sh 里硬编码 nj=32,要求单节点32核,当时
#  每个节点最多26核空闲),把 nj 改成16后重提。旧版VAD逻辑的结果跟老
#  checkout 精确一致:small 127,663 / medium 1,130,579 / large 10,917,974,
#  全量 speechlm 池子 12,176,216;D_Bal 抽样后 speechlm/textlm train 各30万。
#  用时:small切分13分钟、medium 43分钟、large 6小时23分,不用重跑]
# ./run.sh --stage 1 --stop_stage 1 --local_data_opts "--data_config bal"

##############################################################################
# Stage 2: wav dump(重新格式化各个 data 目录里的 wav.scp)。
# 环境已确认 resampy / h5py 都在共享 venv 里,不缺依赖。
# [已跑完,job 1799638,2026-09-18 16:44 结束,约4小时48分,不用重跑]
##############################################################################
# ./run.sh --stage 2 --stop_stage 2

##############################################################################
# Stage 3: 给 asr/tts/speechlm 三个任务打离散单元标签(hubert_base layer6,
# k=1000)。
#
# [改用论文作者发布的官方 kmeans,不自己训]:
#   模型: /export/fs06/mliang17/shared_kmeans/hubert_base_l6_k1000_official/km_1000.mdl
#   来源: https://huggingface.co/soumi-maiti/voxtlm-k1000 (md5 d4674452...,
#         只读保护,来源/验证记录在同目录 SOURCE.txt)
#   为什么: 所有 k1000 实验统一用同一套单元编号——TTS 测试用的 vocoder 只需
#   训一次、所有实验通用;跟论文作者对齐;还省掉了要~120GB内存、要占a100acct
#   大内存节点好几个小时的 kmeans 拟合那一步。
#   兼容性已验证: 格式同为 sklearn MiniBatchKMeans、1000类、768维;在76,202帧
#   hubert_base 第6层特征上平均最近中心距离 37.62,我们自己训的是37.06/37.11,
#   说明是同一个特征空间(换层的话会差好几倍)。
#   代价: 本实验的 token 跟老 checkout 自己训的 kmeans 不同,speechLM PPL 等
#   结果不能再跟老 checkout 的历史数字(24.93)直接比。
#   注意: 以后给这些实验做 TTS 评测的 vocoder,必须用这个官方模型打出的标签训。
#
# 用法: --km_dir 指向官方模型目录 + --learn_kmeans false,lm.sh 会跳过建池子、
# 提池子特征、拟合kmeans,直接用这个模型给三个任务打标签。km_dir 只在 Stage 3
# 用到,后面 Stage 4-7 不需要传。打标签只读这个文件,不会往共享目录里写东西。
#
# 不再需要把 train_cmd 临时切到 conf/slurm_kmeans.conf 大内存配置(那只是给
# learn_kmeans.py 用的)。cuda_cmd 保持默认的普通 gpu 分区即可。
#
# [第一次跑到这里(job 1799638)时还是自己训 kmeans 的写法,刚提完池子特征
#  就 scancel 了(连同它已经提交的 learn_kmeans job 1800320),改成官方模型的
#  写法从这里重提。遗留的 kmeans_pool 和 exp/kmeans 已经删掉]
#
# [第二次(job 1800321)asr 标签打完,speechlm/train 打标签时 16 个子任务里
#  10 个 CUDA OOM:旧版VAD切分不拆长片段,speechlm/train 里留下了一批超长
#  音频(最长 2199.8 秒,超过5分钟的135条),HuBERT 整段前向一次要 13.4GiB,
#  普通 gpu 的卡只有约10GiB。老 checkout 当年抽到的最长只有826秒,靠多次重试
#  才过;我们这次的长度重试多少次也过不了。]
#
# 解决:打标签前用 local/filter_long_utts.sh 把 speechlm/train 里 >120秒的
# 条目从清单里去掉(878条,占speechlm总时长4.2%;只改清单不删音频;原始清单
# 备份在 dump/audio_raw/speechlm/train/.orig_before_len_filter/)。
# 这不会改变最终训练集:BPE后语音至少23.2 token/秒(老checkout实测),>约65秒
# 的必然超过1500 token,Stage 6 之后的长度过滤本来就会把它们全扔掉(老checkout
# 里 >60秒的4538条一条没留下)。120秒留了约2倍余量,换官方kmeans后压缩率略有
# 变化也不影响。脚本可重复执行:已过滤过的再跑一次什么都不变;重跑 Stage 2
# 把清单恢复成30万条后,下次跑到这里会自动再过滤。
# asr、tts 和各 dev/test 都是短句,不用过滤。asr 已打完且有完成标记,会被跳过。
#
# 注意这是原地改 dump/audio_raw/speechlm/train,而这个目录被
# voxtlm_v2/lm1_k2000_Bal 软链接共享(它的 dump/audio_raw 指向这里,已查过
# 没有别的目录链过来)。对它没影响:它自己的 lm.sh 会再按 max_wav_duration
# (<120s,以后可能更低)过滤一遍、写到它自己的 dump/audio_raw_filtered,
# 只读不写这里;而且这里只动 speechlm/train。以后如果有新实验要链这个
# 目录、又需要 >120s 的条目,原始清单在 .orig_before_len_filter/ 里。
#
# PYTORCH_CUDA_ALLOC_CONF 缓解显存碎片化导致的偶发OOM(老checkout当年也加了)。
#
# [已跑完,job 1800548:speechlm 299,122 条、asr、tts 全部打完标签,标签在
#  dump/extracted 下,不用重跑]
##############################################################################
# local/filter_long_utts.sh dump/audio_raw/speechlm/train 120
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# KM_DIR=/export/fs06/mliang17/shared_kmeans/hubert_base_l6_k1000_official
# ./run.sh --stage 3 --stop_stage 3 --km_dir "${KM_DIR}" --learn_kmeans false

##############################################################################
# [job 1800548 的 Stage 4-6 跑完、Stage 7 第1个epoch也跑完了,但第2个epoch
#  开头崩了(CUDA index out of bounds)。原因:voxtlm_v2/lm1_k2000_Bal 的
#  data/ 当时是软链接到本目录 data/ 的,它的 Stage 4/5 在 05:01/05:53 覆盖了
#  这里的 data/train/bpe_text 和 data/en_token_list/bpe_unigram10000/(换成
#  了k2000的BPE),Stage 7 第2个epoch的 DataLoader 重新加载 bpe.model 后,
#  序列变长、超过 OPT 的2048位置上限。已把 k2000 的 data/ 换成独立拷贝,
#  两边不再共享。这里从 Stage 4 重跑,重新生成本实验自己的 bpe_text 和 BPE;
#  坏掉的 exp/lm_stats_en_bpe10000 和训练目录已改名为 *.broken_bpe_0919,
#  Stage 7 从头训。]
#
# Stage 4: 组装多任务文本;Stage 5: 用 Stage 4 产物训 BPE
# (sentencepiece unigram, nbpe=10000, 特殊token走data/nlsyms.txt保留不参与
# 合并)。两步都是纯CPU、本地跑,不用GPU。
##############################################################################
./run.sh --stage 4 --stop_stage 5

##############################################################################
# Stage 6: 统计信息收集(算 text_shape.bpe 等,给 Stage 7 的 batch sampler用)。
##############################################################################
./run.sh --stage 6 --stop_stage 6

# [README踩过的坑]:离散语音单元序列没有天然句子边界,极少数超长序列会让
# Stage 7 在 A100 上 OOM(self-attention显存 O(L^2),batch_bins只按token数
# 线性分桶)。过滤掉长度>1500的,备份原始文件以防误操作(脚本本身幂等)。
F=exp/lm_stats_en_bpe10000/train/text_shape.bpe
[ -e "$F.orig" ] || cp "$F" "$F.orig"
awk '{split($2,a,","); if (a[1]<=1500) print}' "$F.orig" > "$F"

##############################################################################
# Stage 7: 训练语言模型。用 OPT-350M 初始化(论文报告的主结果配置)。
#
# 这是全流程里唯一真正需要大显存、该用 A100 的地方——cuda_cmd 平时(Stage1-6)
# 都保持默认的普通GPU,只在这一步临时升级到 A100,跑完(或者被手动叫停后)
# 用 trap 改回默认,不会让"该用A100的地方一直占着普通GPU资源"或者反过来。
#
# HF_HOME 指向复用的 hf_cache/(已缓存 facebook/opt-350m),配合
# HF_HUB_OFFLINE=1 保证计算节点没有外网也能跑。这一步是真·训练,会跑很久
# (lm1 那次连续跑1天6小时才到22epoch,在13epoch过拟合)。到了合适的时候
# 要手动看 train.log 的 train/valid acc 曲线判断要不要提前叫停,不会自动停。
##############################################################################
# [已跑完,job 1800844/1800881:训到第19轮时手动停掉。valid_loss 在第13轮
#  见底(2.728),之后连涨5轮到2.750,而 train_acc 还从0.431涨到0.449,是典型
#  过拟合;valid.acc.best.pth 自07:27起就一直指向 13epoch.pth。最终用第13轮。
#  后续所有评测(Stage 8-10 + sLM21 + vocoder 客观分)都在 myrun_eval.sh 里,
#  一个脚本跑完,不要重跑这里的 Stage 7。]
trap 'sed -i "s#export cuda_cmd=\"slurm.pl --config conf/slurm_clsp_a100.conf\"#export cuda_cmd=\"slurm.pl\"#" cmd.sh' EXIT
sed -i 's#export cuda_cmd="slurm.pl"#export cuda_cmd="slurm.pl --config conf/slurm_clsp_a100.conf"#' cmd.sh
. ./cmd.sh
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1
./run.sh --stage 7 --stop_stage 7 --ngpu 1 --lm_config conf/tuning/train_transformer_opt350.yaml
