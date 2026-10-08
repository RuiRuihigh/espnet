#!/usr/bin/env bash
#SBATCH --job-name=v1k2kttskm1.3B_eval
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans_opt1.3B/log/myrun_eval.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans_opt1.3B/log/myrun_eval.%j.err

set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans_opt1.3B

. ./path.sh
. ./cmd.sh

# 跟隔壁 lm1_k2000_Bal_ttskmeans/myrun_eval.sh 结构完全一致,唯一区别是模型换成
# OPT-1.3B(--lm_config 指向 conf/tuning/train_transformer_opt1.3b.yaml),用来
# 跟 OPT-350M 那份直接对比初始化模型规模的影响。
#
# 用的 checkpoint 是 valid.acc.best.pth = 8epoch.pth:valid_loss 第7轮见底
# (2.520),之后连涨4轮到第11轮(2.525->2.532->2.550->2.562);valid_acc 第8轮
# 见顶(0.317),之后连续3轮下滑(0.316->0.315->0.314),而 train_acc 还在一路涨
# (0.450->0.475),已经过拟合,训练手动停在第12轮。比隔壁 OPT-350M 那份(在
# 第13~17轮才过拟合)明显更早——同样的 D_Bal 小数据集,参数量更大的模型更快
# 吃满、更快过拟合,符合预期。
#
# 提交前把 checkpoint 说明换成实际训练结果(看 train.log 的 valid loss/acc
# 曲线判断过拟合点,手动选 valid.acc.best.pth)。
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
# keys.txt(text.tts 是纯文本提示,只跟测试集有关,不受模型规模影响)。
#
# sLM21 只依赖训好的 checkpoint,跟 Stage 9/10 的解码产物没有关系,所以提前跑
# 是安全的;好处是最贵的 beam search(Stage 9/10 要好几个小时)不会挡在前面。
#
# GPU 策略:默认全部普通 gpu 分区,只有第4步 VERSA 打分那一下临时切 1 个 A100,
# 脚本内 trap 改回。
#
# 【提交前要确认的事】
#   a) cmd.sh 里 cuda_cmd 必须是默认的 "slurm.pl"(普通gpu)。训练那次 myrun.sh
#      会把它 sed 成 a100 配置,正常退出时 trap 会改回来,但被 scancel 的话不一定
#      来得及,所以提交前先 sed -n '83,85p' cmd.sh 看一眼。
#   b) 第4步用的 vocoder(hubert_voc_v1Balk2000_ttskmeans)可以直接复用,不用
#      像换 kmeans 那样另训 —— 它只跟"单元编号的含义"(即用的哪个 kmeans)
#      绑定,而这个实验的 kmeans 是软链接自 lm1_k2000_Bal_ttskmeans、逐字节
#      相同的模型,跟 LM 是 350M 还是 1.3B 无关。

##############################################################################
# 1) Stage 8: textlm / speechlm 的 PPL
##############################################################################
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1
./run.sh --stage 8 --stop_stage 8 --ngpu 1 --nclusters 2000 \
    --lm_config conf/tuning/train_transformer_opt1.3b.yaml \
    --inference_lm valid.acc.best.pth

##############################################################################
# 2) sLM21: sWUGGY + sBLIMP
#
# --km_dir 指向本 recipe 的 exp/kmeans/hubert_base_6_2000clusters,跟打标签
# 用的模型一致。只跑 dev,语音轨 dev 的 80000+50400=130400 条音频全部要过
# 特征提取+打标签,是最重的一步。
##############################################################################
bash local/eval_slm21_text.sh
bash local/eval_slm21_speech.sh
bash local/eval_slm21_score.sh

##############################################################################
# 3) Stage 9 (ASR 解码,全量) + Stage 10 (TTS 解码,只跑100句子集)
#
# Stage 9 打分会同时产出合计和 test-clean / test-other 两个子集的 WER
# (这段逻辑在共享的 TEMPLATE/lm1/lm.sh 里,所有 v1 recipe 都有)。
##############################################################################
./run.sh --stage 9 --stop_stage 10 --ngpu 1 --nclusters 2000 \
    --lm_config conf/tuning/train_transformer_opt1.3b.yaml \
    --inference_lm valid.acc.best.pth \
    --lm_test_text_tts dump/raw/test/text.tts.subset100

##############################################################################
# 4) TTS 客观评分:hyp.tok -> 波形 -> VERSA 打分
##############################################################################
VOCODER_CKPT=/export/fs06/mliang17/phonologicaltokenizer_modify/ParallelWaveGAN/egs/ljspeech/hubert_voc_v1Balk2000_ttskmeans/exp/train_nodev_ljspeech_hifigan_hubert.v1/checkpoint-400000steps.pkl

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
echo "  PPL      : exp/lm_train_transformer_opt1.3b_en_bpe10000/perplexity_test_*/"
echo "  ASR WER  : exp/lm_train_transformer_opt1.3b_en_bpe10000/decode_test_asr/decode_lm_asr/score_wer/{result.txt,test_clean/result.txt,test_other/result.txt}"
echo "  sLM21    : exp/slm21_eval/submission/ (准确率见本日志里 eval_slm21_score.sh 的输出)"
echo "  TTS 客观分: exp/lm_train_transformer_opt1.3b_en_bpe10000/decode_test_tts/text.tts.subset100/vocoder_eval/*/versa_eval/avg_result.txt (100句子集)"
echo "======================================================================"
