#!/usr/bin/env bash
#SBATCH --job-name=v1k2000_eval
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal/log/myrun_eval.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal/log/myrun_eval.%j.err

set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal

. ./path.sh
. ./cmd.sh

# 一个脚本跑完 v1 k2000 (D_Bal, hubert_base 第6层 + 自训 km_2000) 的全部评测,
# 结构跟隔壁 lm1_k1000_Bal/myrun_eval.sh 完全对应,方便两边结果直接对比。
#
# 用的 checkpoint 是 valid.acc.best.pth = 18epoch.pth:valid_loss 第18轮见底
# (2.544),19/20轮回升到 2.551/2.558,valid_acc 从第13轮起就卡在 0.318~0.319
# 不动而 train_acc 还在涨(0.448 -> 0.470),已经过拟合,训练手动停在第21轮。
#
# 四块内容顺序执行:
#   1) Stage 8   : textlm / speechlm 的 PPL
#   2) Stage 9-10: ASR 解码(WER/CER,会自动分出 test-clean / test-other)、
#                  TTS 解码(出 hyp.tok)
#   3) sLM21     : sWUGGY(词汇) + sBLIMP(句法),文本轨 + 语音轨 + 打分
#   4) vocoder   : hyp.tok 合成波形 + VERSA 客观分(CER/UTMOS 等)
#
# GPU 策略:默认全部普通 gpu 分区,只有第4步 VERSA 打分那一下临时切 1 个 A100
# (所有指标模型同时加载峰值约11.5GB,普通卡10.9GB放不下),脚本内 trap 改回。
#
# 【提交前要确认的事】
#   a) cmd.sh 里 cuda_cmd 必须是默认的 "slurm.pl"(普通gpu)。提交前先
#      sed -n '83,85p' cmd.sh 看一眼,不对就手动改回 slurm.pl。
#   b) 第4步用的 k2000 vocoder(hubert_voc_v1Balk2000)已经训到 40 万步,
#      checkpoint-400000steps.pkl 已就位,不用再等。

##############################################################################
# 1-2) Stage 8 (PPL) + Stage 9 (ASR 解码) + Stage 10 (TTS 解码)
#
# 都是推理:Stage 8 单次前向,Stage 9/10 是 batch=1 的 beam search,普通 gpu
# 够用。--inference_lm valid.acc.best.pth 覆盖 run.sh 默认的 valid.acc.ave.pth
# (多checkpoint平均),跟 k1000 和 v2 k2000 的口径一致,数字才可比。
# --nclusters 2000 要一路传下去,lm.sh 用它拼 pseudo_labels 的文件名。
#
# Stage 9 打分会同时产出合计和 test-clean / test-other 两个子集的 WER
# (这段逻辑已经加进共享的 TEMPLATE/lm1/lm.sh,所有 v1 recipe 都有)。
#
# 结果位置:
#   exp/lm_train_.../perplexity_test_{textlm,unitlm}/
#   exp/lm_train_.../decode_test_asr/decode_lm_asr/score_wer/result.txt
#                                          .../score_wer/{test_clean,test_other}/result.txt
#   exp/lm_train_.../decode_test_tts/text.tts/score_tts/hyp.tok
##############################################################################
# [Stage 8/9 已跑完(job 1802026,取消前):
#    textLM PPL 23.78 / speechLM PPL 30.12
#    ASR WER 合计12.5%(test-clean 7.0% / test-other 18.0%)
#  Stage 10(TTS解码)被取消时只跑到 3898/9957,而且是在 maxlen=900 的实验性
#  配置下跑的一部分——已经证实 maxlen 改 900 对结果基本没区别(真正撞上限的
#  样本k1000/v2k2000里都只有10~12条,长句子CER高的根因是模型在自由生成时
#  提前吐EOS提前收尾,不是长度上限不够),maxlen 已经改回 300,跟 k1000/v2
#  保持一致。这里只重跑 Stage 10,Stage 8/9 结果不会变,不用重跑。]
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1
./run.sh --stage 10 --stop_stage 10 --ngpu 1 --nclusters 2000 \
    --lm_config conf/tuning/train_transformer_opt350.yaml \
    --inference_lm valid.acc.best.pth

##############################################################################
# 3) sLM21: sWUGGY + sBLIMP
#
# 脚本从 lm1_k1000_Bal 复制过来,只改了离散化那部分:--km_dir 指向本实验
# Stage 3 自己训的 exp/kmeans/hubert_base_6_2000clusters,--nclusters 2000,
# 读 pseudo_labels_km2000.txt。必须用本实验自己的 kmeans,否则单元编号跟 LM
# 学到的对不上。
#
# 只跑 dev 不跑 test(论文 3.2 节报的就是 dev)。
# 文本轨会先按内容去重(20000/12566 条)再展开回每个文件名;语音轨没法去重,
# dev 的 80000+50400=130400 条音频全部要过特征提取+打标签,是最重的一步
# (k1000 那边这一步约 25 分钟)。
# sLM21 原始数据通过 downloads/sLM21 软链接复用,已解压的不会重复解压。
##############################################################################
# [已单独跑完,job 1802253(myrun_slm21.sh,约40分钟):
#    sWUGGY 文本轨 83.06% / 语音轨 59.86%
#    sBLIMP 文本轨 68.73% / 语音轨 54.17%
#  不在这里重跑。]
# bash local/eval_slm21_text.sh
# bash local/eval_slm21_speech.sh
# bash local/eval_slm21_score.sh

##############################################################################
# 4) TTS 客观评分:hyp.tok -> 波形 -> VERSA 打分
#
# vocoder 用 hubert_voc_v1Balk2000(LJSpeech 训的 DiscreteSymbolHiFiGAN,
# 16kHz,hop=320 对应 50Hz 单元,num_embs=2000)。它训练用的
# downloads/km_2000.mdl 跟本实验 Stage 3 的 kmeans md5 完全一致
# (20df48056de7e156672a4e060d6ae43f),所以单元编号含义相同,LM 生成的序列
# 才能被正确还原成语音。
#
# 环境:这一步要 parallelwavegan 那个 conda 环境(import parallel_wavegan /
# versa),不能用 path.sh 的 espnet 环境,所以放在子shell里单独激活。
# 注意两个已经踩过的坑(k1000 那边都验证过了):
#   - 不要 unset CONDA_PREFIX / CONDA_DEFAULT_ENV,否则 conda activate 会在
#     读上一个环境的 deactivate.d 时拼出空路径直接报错,环境根本没切过去。
#   - slurm.pl 提交出去的作业脚本开头会执行 ". ./path.sh",在计算节点上把
#     espnet 环境又激活回来,所以 local/eval_tts_vocoder.sh 里每条提交出去的
#     命令都用 CONDA_REACTIVATE 前缀再激活一次 conda。
# VERSA 打分只用 1 个 A100(不切多份并行),避免占满 a100acct 配额。
##############################################################################
VOCODER_CKPT=/export/fs06/mliang17/phonologicaltokenizer_modify/ParallelWaveGAN/egs/ljspeech/hubert_voc_v1Balk2000/exp/train_nodev_ljspeech_hifigan_hubert.v1/checkpoint-400000steps.pkl

(
    unset PYTHONPATH
    export PATH="${PWD}/utils:${PATH}"
    # shellcheck disable=SC1091
    source /opt/slurm_files/software/conda25/2025-08-21/etc/profile.d/conda.sh
    conda activate parallelwavegan
    bash local/eval_tts_vocoder.sh "${VOCODER_CKPT}"
)

##############################################################################
echo "======================================================================"
echo "全部评测跑完。结果位置:"
echo "  PPL      : exp/lm_train_transformer_opt350_en_bpe10000/perplexity_test_*/"
echo "  ASR WER  : exp/lm_train_transformer_opt350_en_bpe10000/decode_test_asr/decode_lm_asr/score_wer/{result.txt,test_clean/result.txt,test_other/result.txt}"
echo "  sLM21    : exp/slm21_eval/submission/ (准确率见本日志里 eval_slm21_score.sh 的输出)"
echo "  TTS 客观分: exp/lm_train_transformer_opt350_en_bpe10000/decode_test_tts/text.tts/vocoder_eval/*/versa_eval/avg_result.txt"
echo "======================================================================"
