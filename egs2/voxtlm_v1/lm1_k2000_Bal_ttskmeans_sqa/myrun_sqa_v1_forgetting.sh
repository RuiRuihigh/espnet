#!/usr/bin/env bash
#SBATCH --job-name=sqa_v1_forget_ttskm
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans_sqa/log/myrun_sqa_v1_forgetting.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans_sqa/log/myrun_sqa_v1_forgetting.%j.err

# 用法: sbatch myrun_sqa_v1_forgetting.sh [13epoch.pth]
#
# 跟 voxtlm_v2/lm1_k2000_Bal_sqa/myrun_sqa_v1_forgetting.sh 结构完全一致,这份是
# ttskmeans(k2000, hubert_base L6, 只用 tts 训的 kmeans; OPT-350M)底座的版本。
# SQA finetune 前(17epoch, zero-shot) vs 后(13epoch, valid.loss.best)的完整遗忘
# 测试,一个脚本里顺序跑完原来 4 个预训练任务全部对照:
#   1) textLM / speechLM PPL(Stage 8)+ ASR WER(Stage 9,自动拆 test-clean/other)
#   2) sWUGGY(lexical) + sBLIMP(syntactic),文本轨 + 语音轨
#   3) TTS:三级回退解码(beam5 撞满maxlen -> 贪心 -> top-p 0.9 采样,仿
#      lm1_k2000_Bal_ttskmeans/myrun_tts_fallback.sh)+ vocoder + VERSA(CER/UTMOS),
#      100 句固定随机子集(local/tts_eval_subset/keys_random100.txt,
#      对应 lm1_k2000_Bal_ttskmeans/dump/raw/test/text.tts.rand100)
#
# 跟 v2 版本的关键差异(ttskmeans 这条线特有的):
#   - lm.sh 是本 recipe 自己的真实拷贝(跟 lm1_k2000_Bal_ttskmeans 的 lm.sh 逐字节
#     相同,diff 确认过),默认 nclusters=1000,所以每次调用都要显式传
#     --nclusters 2000,不能省(v2 的 lm.sh 走共享 TEMPLATE,不需要这个参数)。
#   - 本 recipe 的 cmd.sh 默认 cuda_cmd 是 "${SQA_CUDA_CMD:-slurm.pl}"(训练脚本
#     用 export SQA_CUDA_CMD=... 临时切A100),不是其它 recipe 那种直接写
#     'export cuda_cmd="slurm.pl"' 字面文本再 sed 的机制 —— local/eval_tts_vocoder.sh
#     里 VERSA 打分那一步已经相应改成用同样的环境变量机制切A100(不是 sed cmd.sh)。
#   - sLM21 语音轨用的是本 recipe自己的 hubert_base(L6)+ 自训 kmeans(只用 tts
#     子集训的,不是 v2 的 phonological_tokenizer),--km_dir 指向
#     exp/kmeans/hubert_base_6_2000clusters(软链接自 lm1_k2000_Bal_ttskmeans)。
#   - TTS vocoder 用 hubert_voc_v1Balk2000_ttskmeans(不是 v2 的 phon_voc1),
#     跟 lm1_k2000_Bal_ttskmeans 自己的 myrun_eval.sh/myrun_tts_fallback.sh 一致。
#
# 参照基线(finetune 之前,17epoch,SQA 训练前跑的原始结果,全部在
# lm1_k2000_Bal_ttskmeans 自己的 exp/ 下,本脚本只读,不会写):
#   PPL/ASR  : exp/lm_train_transformer_opt350_en_bpe10000/{perplexity_test_textlm,
#              perplexity_test_unitlm,decode_test_asr}/...
#   slm21    : 来自 log/myrun_eval.1803972.out 的历史打分数字(没有单独的
#              submission 文件可以直接复用,只能贴数字,见下面汇总部分)
#   TTS      : exp/lm_train_transformer_opt350_en_bpe10000/decode_test_tts/
#              text.tts.fb_final/vocoder_eval/checkpoint-400000steps/versa_eval/
#              avg_result.txt(17epoch 自己也跑过三级回退,同一个 rand100 子集)
#
# 本地写入范围(不碰 lm1_k2000_Bal_ttskmeans):
#   - PPL/ASR/slm21/TTS 的所有输出都在本目录 exp/sqa_v1_ft、exp/slm21_eval 下
#   - TTS 回退用到的中间 prompt 文件(fb_greedy/fb_topp)写在本地 dump/tts_fallback/,
#     不写进 ${SRC}/dump/raw/test/(那是 lm1_k2000_Bal_ttskmeans 自己的目录)
#   - sLM21 数据集: zip 本身、extracted/ 都软链接自 lm1_k2000_Bal_ttskmeans(已经
#     解压好的 130400 个 wav,跟 extract_slm21() 的"已存在就跳过"配合,不会触发
#     重新解压)
#   - 三个只读输入软链接自 lm1_k2000_Bal_ttskmeans: exp/kmeans(转换好的码本)、
#     downloads/sLM21(zip+extracted,上面已说);ckpt(HuBERT SSL权重)是本
#     recipe自己早就有的软链接,不需要新建。eval_tts_vocoder.sh 里 ground truth
#     (data/test/speech/tts/{wav.scp,text})走绝对路径直接读
#     lm1_k2000_Bal_ttskmeans,不需要额外软链接。
set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans_sqa
. ./path.sh
. ./cmd.sh
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1

SRC=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans
LM_EXP=exp/sqa_v1_ft
CKPT=${1:-13epoch.pth}

##############################################################################
# 1) textLM / speechLM PPL(Stage 8)+ ASR WER(Stage 9)
##############################################################################
./lm.sh --stage 8 --stop_stage 9 --skip_train true --nclusters 2000 \
    --lang en --token_type bpe --nbpe 10000 \
    --dumpdir ${SRC}/dump --train_set train --valid_set dev --test_sets test \
    --lm_exp "${LM_EXP}" --inference_lm "${CKPT}" \
    --gpu_inference true --inference_nj 8 \
    --lm_inference_asr_config conf/decode_lm_asr.yaml \
    --lm_test_text_asr ${SRC}/dump/raw/test/text.asr \
    --lm_test_text_textlm ${SRC}/dump/raw/test/text.textlm \
    --lm_test_text_speechlm ${SRC}/dump/raw/test/text.speechlm

##############################################################################
# 2) sWUGGY(lexical) + sBLIMP(syntactic),文本轨 + 语音轨 + 打分
#    local/slm21_common.sh 里 LM_CONFIG/LM_FILE 已经指向 ${LM_EXP}/13epoch.pth;
#    如果传了别的 CKPT,注意 local/slm21_common.sh 里的 LM_FILE 是写死的,
#    需要同步手改。
##############################################################################
bash local/eval_slm21_text.sh
bash local/eval_slm21_speech.sh
bash local/eval_slm21_score.sh > "${LM_EXP}/slm21_score.txt"
cat "${LM_EXP}/slm21_score.txt"

##############################################################################
# 3) TTS: 三级回退解码 + vocoder + VERSA(仿 lm1_k2000_Bal_ttskmeans/
#    myrun_tts_fallback.sh,checkpoint 换成 sqa finetune 的,中间文件全部落
#    本地不写回 SRC)
##############################################################################
VOCODER_CKPT=/export/fs06/mliang17/phonologicaltokenizer_modify/ParallelWaveGAN/egs/ljspeech/hubert_voc_v1Balk2000_ttskmeans/exp/train_nodev_ljspeech_hifigan_hubert.v1/checkpoint-400000steps.pkl
[ -f "${VOCODER_CKPT}" ] || { echo "vocoder checkpoint 不存在: ${VOCODER_CKPT}" >&2; exit 1; }

MAXLEN=1200
D=${LM_EXP}/decode_test_tts
FB=dump/tts_fallback
mkdir -p "${FB}"
LM_CFG=conf/tuning/finetune_sqa_v1_opt350.yaml
COMMON_TTS="--stage 10 --stop_stage 10 --ngpu 1 --nclusters 2000 --gpu_inference true --inference_nj 8 --lang en --token_type bpe --nbpe 10000 --dumpdir ${SRC}/dump --train_set train --valid_set dev --test_sets test --lm_config ${LM_CFG} --lm_exp ${LM_EXP} --inference_lm ${CKPT}"

# ---------------- 第1级: beam5, maxlen 1200, 100句固定随机子集 ----------------
n1=0; [ -s "${D}/text.tts.rand100/score_tts/hyp.tok" ] && n1=$(wc -l < "${D}/text.tts.rand100/score_tts/hyp.tok")
ml1=""; [ -f "${D}/text.tts.rand100/logdir/lm_inference.1.log" ] && ml1=$(grep -m1 -o "max output length: \[[0-9]*\]" "${D}/text.tts.rand100/logdir/lm_inference.1.log" || true)
if [ "${n1}" -eq 100 ] && [ "${ml1}" = "max output length: [${MAXLEN}]" ]; then
    echo "[L1] 复用已有的 beam5 结果 text.tts.rand100(${n1} 句, ${ml1})"
else
    echo "[L1] 没有可用的 beam5 结果(hyp.tok=${n1} 行, '${ml1}'),重新解码 100 句"
    ./lm.sh ${COMMON_TTS} --python python3 \
        --lm_inference_tts_config conf/decode_lm_tts_maxlen1200.yaml \
        --lm_test_text_tts ${SRC}/dump/raw/test/text.tts.rand100
fi

python3 local/count_capped_steps.py "${D}/text.tts.rand100" ${MAXLEN}
python3 local/fallback_select.py capped "${D}/text.tts.rand100" ${MAXLEN} > "${FB}/capped_L1.txt"
echo "[L1] 撞满 $(wc -l < "${FB}/capped_L1.txt") 句"

rm -rf "${D}/text.tts.fb_greedy" "${D}/text.tts.fb_topp" "${D}/text.tts.fb_final"
LEVELS="${D}/text.tts.rand100"

# ---------------- 第2级: 贪心(只重解第1级撞满的句子) ----------------
if [ -s "${FB}/capped_L1.txt" ]; then
    python3 local/fallback_select.py make_text ${SRC}/dump/raw/test/text.tts.rand100 "${FB}/capped_L1.txt" "${FB}/text.tts.fb_greedy"
    unset SAMPLE_MODE SAMPLE_TOP_P SAMPLE_TEMP SAMPLE_SEED SAMPLE_PER_UTT_SEED
    ./lm.sh ${COMMON_TTS} --python local/python_sampling.sh \
        --lm_inference_tts_config conf/decode_lm_tts_beam1_maxlen1200.yaml \
        --lm_test_text_tts "${FB}/text.tts.fb_greedy"
    python3 local/fallback_select.py capped "${D}/text.tts.fb_greedy" ${MAXLEN} > "${FB}/capped_L2.txt"
    echo "[L2] 贪心后仍撞满 $(wc -l < "${FB}/capped_L2.txt") 句"
    LEVELS="${LEVELS} ${D}/text.tts.fb_greedy"

    # ---------------- 第3级: top-p 0.9 采样(只重解第2级仍撞满的句子) ----------------
    if [ -s "${FB}/capped_L2.txt" ]; then
        python3 local/fallback_select.py make_text ${SRC}/dump/raw/test/text.tts.rand100 "${FB}/capped_L2.txt" "${FB}/text.tts.fb_topp"
        SAMPLE_MODE=topp SAMPLE_TOP_P=0.9 SAMPLE_TEMP=1.0 SAMPLE_SEED=0 SAMPLE_PER_UTT_SEED=1 \
        ./lm.sh ${COMMON_TTS} --python local/python_sampling.sh \
            --lm_inference_tts_config conf/decode_lm_tts_beam1_maxlen1200.yaml \
            --lm_test_text_tts "${FB}/text.tts.fb_topp"
        LEVELS="${LEVELS} ${D}/text.tts.fb_topp"
    else
        echo "[L3] 贪心已经解决了所有撞满的句子,不需要 top-p"
    fi
else
    echo "[L2/L3] 第1级没有撞满的句子,不需要回退"
fi

# ---------------- 拼接 + vocoder + VERSA ----------------
python3 local/fallback_select.py assemble "${D}/text.tts.fb_final" ${MAXLEN} ${LEVELS}
(
    unset PYTHONPATH
    export PATH="${PWD}/utils:${PATH}"
    # shellcheck disable=SC1091
    source /opt/slurm_files/software/conda25/2025-08-21/etc/profile.d/conda.sh
    conda activate parallelwavegan
    bash local/eval_tts_vocoder.sh "${VOCODER_CKPT}" text.tts.fb_final
)
python3 local/summarize_fallback.py

##############################################################################
# 汇总: finetune(${CKPT}) vs zero-shot(17epoch,finetune 之前)
##############################################################################
BASE=${SRC}/exp/lm_train_transformer_opt350_en_bpe10000
sumavg() { grep "Sum/Avg" "$1" 2>/dev/null || echo "(找不到 $1)"; }

echo "======================================================================"
echo "==== 汇总: finetune(${CKPT}) vs zero-shot(17epoch) ===="
echo "======================================================================"
echo "---- textLM PPL ----"
echo "finetune : $(cat ${LM_EXP}/perplexity_test_textlm/*/ppl)"
echo "zero-shot: $(cat ${BASE}/perplexity_test_textlm/*/ppl)"
echo "---- speechLM PPL ----"
echo "finetune : $(cat ${LM_EXP}/perplexity_test_unitlm/*/ppl)"
echo "zero-shot: $(cat ${BASE}/perplexity_test_unitlm/*/ppl)"
echo "---- ASR WER(合并) ----"
echo "finetune : $(sumavg ${LM_EXP}/decode_test_asr/decode_lm_asr/score_wer/result.txt)"
echo "zero-shot: $(sumavg ${BASE}/decode_test_asr/decode_lm_asr/score_wer/result.txt)"
for split in test_clean test_other; do
    echo "---- ASR WER (${split}) ----"
    echo "finetune : $(sumavg ${LM_EXP}/decode_test_asr/decode_lm_asr/score_wer/${split}/result.txt)"
    echo "zero-shot: $(sumavg ${BASE}/decode_test_asr/decode_lm_asr/score_wer/${split}/result.txt)"
done
echo "---- sWUGGY/sBLIMP: finetune(${CKPT}) ----"
cat "${LM_EXP}/slm21_score.txt"
echo "---- sWUGGY/sBLIMP: zero-shot(17epoch,来自 ${SRC}/log/myrun_eval.1803972.out) ----"
cat <<'EOF'
lexical (text): accuracy=82.52% (40000 pairs)
lexical (speech): accuracy=59.79% (40000 pairs)
syntactic (text): accuracy=68.97% (25200 pairs)
syntactic (speech): accuracy=54.60% (25200 pairs)
EOF
echo "---- TTS: finetune(${CKPT}), 三级回退最终结果 ----"
cat "${D}/text.tts.fb_final/vocoder_eval/checkpoint-400000steps/versa_eval/avg_result.txt"
echo "---- TTS: zero-shot(17epoch), 三级回退最终结果 ----"
cat "${BASE}/decode_test_tts/text.tts.fb_final/vocoder_eval/checkpoint-400000steps/versa_eval/avg_result.txt"
echo "(逐句细节见上面 summarize_fallback.py 打印的完整对比表,含每一级回退用了几句)"
