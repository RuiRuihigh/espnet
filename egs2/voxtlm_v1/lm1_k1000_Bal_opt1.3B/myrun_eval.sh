#!/usr/bin/env bash
#SBATCH --job-name=v1k1kBal1.3B_eval
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_opt1.3B/log/myrun_eval.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_opt1.3B/log/myrun_eval.%j.err

set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_opt1.3B

. ./path.sh
. ./cmd.sh

# 跟 lm1_k2000_Bal_ttskmeans_opt1.3B/myrun_eval.sh 结构完全一致(而不是照搬
# lm1_k1000_Bal 自己那份已经过时的 myrun_eval.sh——那份里 Stage8-10/sLM21 是
# 注释掉的历史记录,vocoder 那步用的是全量 9957 条,是当时还没搞清楚"束搜索
# 在长句子上会跑飞"这个问题之前跑的,不适合直接照抄)。这次是全新的、还没跑过
# 评测的 checkpoint,四块全部要跑,TTS 只评 100 句子集(local/tts_eval_subset/
# keys.txt,text.tts 是纯文本提示,跟模型规模无关,跟其它 recipe 用的是同一份)。
#
# 用的 checkpoint 是手动指定的 6epoch.pth,不是 valid.acc.best.pth(目前指向
# 第7轮)。训练在第8轮时手动停掉:valid_loss 第6轮见底(2.691),第7、8轮只是
# 小幅回升(2.692->2.700),valid_acc 卡在 0.297 三轮没动,train_acc 还在稳步涨
# (0.427->0.435->0.443)——过拟合信号还不够清晰(不像隔壁 OPT-350M/OPT-1.3B
# ttskmeans 那种连涨多轮、valid_acc 连续下滑的明确情况),但出于保守选了回升
# 前的最后一个稳定点(第6轮)评测,而不是 valid_acc 并列最高但已经开始回升的
# 第7轮。
#
# sLM21 的 checkpoint 路径在 local/slm21_common.sh 里改成了 6epoch.pth(同样
# 的理由,见该文件内注释)。eval_tts_vocoder.sh 不涉及 checkpoint 选择(只读
# Stage 10 产出的 hyp.tok),不用改。
#
# 四块内容顺序执行(注意顺序:sLM21 提到 ASR/TTS 解码之前):
#   1) Stage 8   : textlm / speechlm 的 PPL
#   2) sLM21     : sWUGGY(词汇) + sBLIMP(句法),文本轨 + 语音轨 + 打分
#   3) Stage 9-10: ASR 解码(WER/CER,会自动分出 test-clean / test-other)、
#                  TTS 解码(出 hyp.tok,只跑 100 句短子集)
#   4) vocoder   : hyp.tok 合成波形 + VERSA 客观分(CER/UTMOS 等)
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
#   b) 第4步用的 vocoder(hubert_voc1)可以直接复用,不用另训 —— 它用的官方
#      km_1000.mdl 跟本 recipe Stage 3 打标签用的是同一个外部模型(md5
#      d46744528bf7f8dba295d560925fc1cf),单元编号含义相同,跟 LM 是 350M
#      还是 1.3B 无关。

##############################################################################
# 1) Stage 8: textlm / speechlm 的 PPL
##############################################################################
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1
./run.sh --stage 8 --stop_stage 8 --ngpu 1 \
    --lm_config conf/tuning/train_transformer_opt1.3b.yaml \
    --inference_lm 6epoch.pth

##############################################################################
# 2) sLM21: sWUGGY + sBLIMP
#
# --km_dir 指向论文作者发布的官方 km_1000(外部路径,不在本 recipe 里),跟
# 本 recipe Stage 3 打标签用的是同一个模型。只跑 dev,语音轨 dev 的
# 80000+50400=130400 条音频全部要过特征提取+打标签,是最重的一步(k1000_Bal
# 那边这一步约 25 分钟)。
##############################################################################
bash local/eval_slm21_text.sh
bash local/eval_slm21_speech.sh
bash local/eval_slm21_score.sh

##############################################################################
# 3) Stage 9 (ASR 解码,全量) + Stage 10 (TTS 解码,只跑100句子集)
#
# Stage 9 打分会同时产出合计和 test-clean / test-other 两个子集的 WER
# (这段逻辑在共享的 TEMPLATE/lm1/lm.sh 里,所有 v1 recipe 都有)。
# --lm_test_text_tts 指向 100 句子集,覆盖 run.sh 默认的全量
# dump/raw/test/text.tts;lm.sh 用这个文件的 basename 给输出目录命名,所以
# 结果落在 decode_test_tts/text.tts.subset100/,不影响全量结果(如果以后要跑)。
##############################################################################
./run.sh --stage 9 --stop_stage 10 --ngpu 1 \
    --lm_config conf/tuning/train_transformer_opt1.3b.yaml \
    --inference_lm 6epoch.pth \
    --lm_test_text_tts dump/raw/test/text.tts.subset100

##############################################################################
# 4) TTS 客观评分:hyp.tok -> 波形 -> VERSA 打分
##############################################################################
VOCODER_CKPT=/export/fs06/mliang17/phonologicaltokenizer_modify/ParallelWaveGAN/egs/ljspeech/hubert_voc1/exp/train_nodev_ljspeech_hifigan_hubert.v1/checkpoint-400000steps.pkl

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
