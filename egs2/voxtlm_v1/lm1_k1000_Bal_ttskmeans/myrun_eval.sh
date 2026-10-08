#!/usr/bin/env bash
#SBATCH --job-name=v1k1kttskm_eval
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_ttskmeans/log/myrun_eval.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_ttskmeans/log/myrun_eval.%j.err

set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_ttskmeans

. ./path.sh
. ./cmd.sh

# 一个脚本跑完 v1 k1000 ttskmeans (D_Bal, hubert_base 第6层 + 只用 tts 10% 子集
# 自训的 km_1000) 的全部评测。结构跟 lm1_k2000_Bal_ttskmeans/myrun_eval.sh 完全
# 对应(只是 k 从 2000 换成 1000),方便两边结果直接对比;跟 lm1_k1000_Bal(用官方
# km_1000)对比则能看出"kmeans 训法"本身的影响。
#
# 用的 checkpoint 是 valid.acc.best.pth。训练(myrun.sh 的 Stage 7)不会自动停,
# 要盯着 train.log 的 valid loss/acc 判断过拟合、手动停,valid.acc.best.pth 会
# 一直指向 valid acc 最高的那一轮。
#
# 四块内容顺序执行(注意顺序:sLM21 提到 ASR/TTS 解码之前):
#   1) Stage 8   : textlm / speechlm 的 PPL
#   2) sLM21     : sWUGGY(词汇) + sBLIMP(句法),文本轨 + 语音轨 + 打分
#   3) Stage 9-10: ASR 解码(WER/CER,会自动分出 test-clean / test-other)、
#                  TTS 解码(出 hyp.tok,只跑 100 句短子集)
#   4) vocoder   : hyp.tok 合成波形 + VERSA 客观分(CER/UTMOS 等)
#
# TTS 这块只评 local/tts_eval_subset/keys.txt 里那 100 条,跟 lm1_k1000_Bal /
# lm1_k2000_Bal / v2 lm1_k2000_Bal / lm1_k2000_Bal_ttskmeans 用的是同一份
# keys.txt,数字才可比。选法见那个目录的 README:以 token 真值长度 <200 为候选池、
# 种子42抽100条,避开"长句子自由生成时提前吐EOS收尾"这个已知问题。各 recipe 的
# test tts utt id 集合完全一致,而 text.tts 是纯文本提示、跟用哪套 kmeans 无关,
# 所以筛出来的 text.tts.subset100 跟别的 recipe 的那份逐字节相同(下面第3块之前
# 会自动生成,不用手动准备)。
#
# sLM21 只依赖 exp/lm_train_.../valid.acc.best.pth(见 local/slm21_common.sh),
# 跟 Stage 9/10 的解码产物没有任何关系,所以提前跑是安全的;好处是最贵的
# beam search(Stage 9/10 要几个小时)不会再挡在 sLM21 前面。
#
# GPU 策略:默认全部普通 gpu 分区,只有第4步 VERSA 打分那一下临时切 1 个 A100
# (所有指标模型同时加载峰值约11.5GB,普通卡10.9GB放不下),脚本内 trap 改回。
#
# 【提交前要确认的事】
#   a) cmd.sh 里 cuda_cmd 必须是默认的 "slurm.pl"(普通gpu)。训练那次 myrun.sh
#      会把它 sed 成 a100 配置,正常退出时 trap 会改回来,但被 scancel 的话不一定
#      来得及,所以提交前先 sed -n '83,85p' cmd.sh 看一眼。
#   b) 第4步用的 vocoder 是 hubert_voc_v1Balk1000_ttskmeans ——【目前还不存在】。
#      它必须用本实验 Stage 3 训出来的 km_1000.mdl 打的标签来训(参考
#      ParallelWaveGAN/egs/ljspeech/hubert_voc_v1Balk2000_ttskmeans 的做法:
#      复制那个 recipe、把 kmeans 指向本 recipe 的
#      exp/kmeans/hubert_base_6_1000clusters/km_1000.mdl、num_embs 改成 1000,
#      完整重训 40 万步)。特例:如果 myrun.sh 里 Stage 3 之后打印的 md5 显示
#      本实验的 km_1000.mdl 跟官方的逐字节一致,那么 hubert_voc1
#      (.../egs/ljspeech/hubert_voc1/exp/train_nodev_ljspeech_hifigan_hubert.v1/
#      checkpoint-400000steps.pkl)就能直接用,把下面的 VOCODER_CKPT 改成它即可。
#      前三块(PPL、sLM21、ASR/TTS 解码)不依赖 vocoder,可以先跑;要先跑的话把
#      第4块整段注释掉。

##############################################################################
# 1) Stage 8: textlm / speechlm 的 PPL
#
# 单次前向,很快,普通 gpu 够用。--inference_lm valid.acc.best.pth 覆盖 run.sh
# 默认的 valid.acc.ave.pth(多checkpoint平均),跟别的 recipe 的口径一致,
# 数字才可比。--nclusters 1000 要一路传下去,lm.sh 用它拼 pseudo_labels 的文件名。
#
# 结果位置:
#   exp/lm_train_.../perplexity_test_{textlm,unitlm}/
##############################################################################
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1
./run.sh --stage 8 --stop_stage 8 --ngpu 1 --nclusters 1000 \
    --lm_config conf/tuning/train_transformer_opt350.yaml \
    --inference_lm valid.acc.best.pth

##############################################################################
# 2) sLM21: sWUGGY + sBLIMP
#
# 脚本从 lm1_k1000_Bal 复制过来,离散化那部分指向本实验自己的东西:--km_dir 是
# Stage 3 自己训的 exp/kmeans/hubert_base_6_1000clusters,--nclusters 1000,
# 读 pseudo_labels_km1000.txt。必须用本实验自己的 kmeans,否则单元编号跟 LM
# 学到的对不上。
#
# 只跑 dev 不跑 test(论文 3.2 节报的就是 dev)。
# 文本轨会先按内容去重(20000/12566 条)再展开回每个文件名;语音轨没法去重,
# dev 的 80000+50400=130400 条音频全部要过特征提取+打标签,是最重的一步
# (lm1_k1000_Bal 那边这一步约 25 分钟)。
# sLM21 原始数据通过 downloads/sLM21 软链接复用,已解压的不会重复解压。
##############################################################################
bash local/eval_slm21_text.sh
bash local/eval_slm21_speech.sh
bash local/eval_slm21_score.sh

##############################################################################
# 3) Stage 9 (ASR 解码) + Stage 10 (TTS 解码)
#
# batch=1 的 beam search,是整个评测里最慢的一块(几个小时),普通 gpu 够用。
# 参数跟上面 Stage 8 那次完全一致,同一个 checkpoint。
#
# Stage 9 打分会同时产出合计和 test-clean / test-other 两个子集的 WER
# (这段逻辑已经加进共享的 TEMPLATE/lm1/lm.sh,所有 v1 recipe 都有)。
#
# Stage 10 产出的 hyp.tok 是下面第 4 块 vocoder 的输入。
#
# --lm_test_text_tts 指向 100 句子集,覆盖 run.sh 默认的全量 dump/raw/test/text.tts
# (9957 条)。lm.sh 用这个文件的 basename 给输出目录命名,所以结果落在
# decode_test_tts/text.tts.subset100/,跟全量结果各走各的、互不覆盖。
# Stage 9 走的是 text.asr,不受这个参数影响,仍然是全量 ASR 评测。
#
# text.tts.subset100 由 Stage 4 产出的 dump/raw/test/text.tts 按 keys.txt 过滤
# 出来(uttid 带 tts_ 前缀,所以先给 keys 加前缀)。已经存在就不重复生成。
#
# 结果位置:
#   exp/lm_train_.../decode_test_asr/decode_lm_asr/score_wer/result.txt
#                                          .../score_wer/{test_clean,test_other}/result.txt
#   exp/lm_train_.../decode_test_tts/text.tts.subset100/score_tts/hyp.tok
##############################################################################
if [ ! -s dump/raw/test/text.tts.subset100 ]; then
    sed 's/^/tts_/' local/tts_eval_subset/keys.txt > dump/raw/test/.tts_subset100_keys.txt
    utils/filter_scp.pl dump/raw/test/.tts_subset100_keys.txt dump/raw/test/text.tts \
        > dump/raw/test/text.tts.subset100
    rm -f dump/raw/test/.tts_subset100_keys.txt
fi
n_sub=$(wc -l < dump/raw/test/text.tts.subset100)
[ "${n_sub}" -eq 100 ] || { echo "text.tts.subset100 应该是 100 行,实际 ${n_sub}" >&2; exit 1; }

./run.sh --stage 9 --stop_stage 10 --ngpu 1 --nclusters 1000 \
    --lm_config conf/tuning/train_transformer_opt350.yaml \
    --inference_lm valid.acc.best.pth \
    --lm_test_text_tts dump/raw/test/text.tts.subset100

##############################################################################
# 4) TTS 客观评分:hyp.tok -> 波形 -> VERSA 打分
#
# vocoder 用 hubert_voc_v1Balk1000_ttskmeans(LJSpeech 训的
# DiscreteSymbolHiFiGAN,16kHz,hop=320 对应 50Hz 单元,num_embs=1000)——
# 【需要先按上面"提交前要确认的事 b)"准备好】。它训练用的 downloads/km_1000.mdl
# 必须跟本实验 Stage 3 的 kmeans md5 完全一致,单元编号含义才相同,LM 生成的序列
# 才能被正确还原成语音。
#
# 环境:这一步要 parallelwavegan 那个 conda 环境(import parallel_wavegan /
# versa),不能用 path.sh 的 espnet 环境,所以放在子shell里单独激活。
# 注意两个已经踩过的坑:
#   - 不要 unset CONDA_PREFIX / CONDA_DEFAULT_ENV,否则 conda activate 会在
#     读上一个环境的 deactivate.d 时拼出空路径直接报错,环境根本没切过去。
#   - slurm.pl 提交出去的作业脚本开头会执行 ". ./path.sh",在计算节点上把
#     espnet 环境又激活回来,所以 local/eval_tts_vocoder.sh 里每条提交出去的
#     命令都用 CONDA_REACTIVATE 前缀再激活一次 conda。
# VERSA 打分只用 1 个 A100(不切多份并行),避免占满 a100acct 配额。
##############################################################################
VOCODER_CKPT=/export/fs06/mliang17/phonologicaltokenizer_modify/ParallelWaveGAN/egs/ljspeech/hubert_voc_v1Balk1000_ttskmeans/exp/train_nodev_ljspeech_hifigan_hubert.v1/checkpoint-400000steps.pkl
[ -f "${VOCODER_CKPT}" ] || { echo "vocoder checkpoint 不存在: ${VOCODER_CKPT}(见脚本开头'提交前要确认的事 b)')" >&2; exit 1; }

(
    unset PYTHONPATH
    export PATH="${PWD}/utils:${PATH}"
    # shellcheck disable=SC1091
    source /opt/slurm_files/software/conda25/2025-08-21/etc/profile.d/conda.sh
    conda activate parallelwavegan
    bash local/eval_tts_vocoder.sh "${VOCODER_CKPT}" text.tts.subset100
)

##############################################################################
echo "======================================================================"
echo "全部评测跑完。结果位置:"
echo "  PPL      : exp/lm_train_transformer_opt350_en_bpe10000/perplexity_test_*/"
echo "  ASR WER  : exp/lm_train_transformer_opt350_en_bpe10000/decode_test_asr/decode_lm_asr/score_wer/{result.txt,test_clean/result.txt,test_other/result.txt}"
echo "  sLM21    : exp/slm21_eval/submission/ (准确率见本日志里 eval_slm21_score.sh 的输出)"
echo "  TTS 客观分: exp/lm_train_transformer_opt350_en_bpe10000/decode_test_tts/text.tts.subset100/vocoder_eval/*/versa_eval/avg_result.txt (100句子集)"
echo "======================================================================"
