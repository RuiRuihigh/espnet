#!/usr/bin/env bash
#SBATCH --job-name=sqa_v1_forget_k1000
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_sqa/log/myrun_sqa_v1_forgetting.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_sqa/log/myrun_sqa_v1_forgetting.%j.err

# 用法: sbatch myrun_sqa_v1_forgetting.sh [15epoch.pth]
#
# 跟 lm1_k2000_Bal_ttskmeans_sqa/myrun_sqa_v1_forgetting.sh(以及 v2/lm1_k2000_Bal_sqa
# 的同名脚本)结构完全一致,这份是最早的 V1 参照底座(k1000, hubert_base L6, 论文
# 作者发布的官方 kmeans; OPT-350M)。SQA finetune 前(13epoch, zero-shot) vs 后
# (15epoch, valid.acc.best)的完整遗忘测试:
#   1) textLM / speechLM PPL(Stage 8)+ ASR WER(Stage 9,自动拆 test-clean/other)
#   2) sWUGGY(lexical) + sBLIMP(syntactic),文本轨 + 语音轨
#   3) TTS:三级回退解码(beam5 撞满maxlen -> 贪心 -> top-p 0.9 采样,仿
#      lm1_k1000_Bal 自己的 rand100/fb_final 结果)+ vocoder + VERSA(CER/UTMOS),
#      100 句固定随机子集(local/tts_eval_subset/keys_random100.txt,对应
#      lm1_k1000_Bal/dump/raw/test/text.tts.rand100)
#
# 跟 ttskmeans 那份的几个关键差异(这条线更简单,踩过的坑已经避开):
#   - lm.sh 是软链接到共享 TEMPLATE(../../TEMPLATE/lm1/lm.sh),跟
#     lm1_k1000_Bal 自己的 lm.sh 是同一个文件,不是像 ttskmeans 那样的改过的
#     真实拷贝。TEMPLATE 默认 nclusters=1000,跟 k1000 一致,--nclusters 严格说
#     不传也行,这里还是显式传上保持跟其它几份一致。
#   - sLM21 语音轨用论文作者发布的官方 kmeans(外部绝对路径
#     /export/fs06/mliang17/shared_kmeans/hubert_base_l6_k1000_official),
#     不是 recipe 自己训的,所以不需要 exp/kmeans 软链接(跟 lm1_k1000_Bal_
#     opt1.3B 那次的结论一致)。
#   - TTS vocoder 用 hubert_voc1(不是 ttskmeans 专属的
#     hubert_voc_v1Balk2000_ttskmeans)。
#   - 本 recipe local/ 原来只有 SQA 自己需要的几个脚本(eval_sqa.sh/
#     run_mc_loglik.sh/postprocess.py 等),没有 sLM21/vocoder/fallback 相关
#     脚本,也**漏了 postprocess.py 的搭档 postprocess_speech.py**(Stage 10
#     TTS 解码后处理要用,ttskmeans 那次就是因为漏了这个在 Stage 9 postprocess.py
#     那里才报错——这次复制的时候已经把 postprocess_speech.py 一起带上了)。
#   - 本 recipe 的 cmd.sh 默认 cuda_cmd 是 "${SQA_CUDA_CMD:-slurm.pl}"(训练脚本
#     用 export SQA_CUDA_CMD=... 临时切A100),不是直接 sed cmd.sh 文件里的字面
#     文本,local/eval_tts_vocoder.sh 里 VERSA 打分那一步已经改成用同样的环境
#     变量机制切A100(不是 sed),避免了 ttskmeans 那份一开始没注意到、sed 匹配
#     不到导致 A100 切换静默失效的问题。
#
# 参照基线(finetune 之前,13epoch,SQA 训练前跑的原始结果,全部在 lm1_k1000_Bal
# 自己的 exp/ 下,本脚本只读,不会写):
#   PPL/ASR  : exp/lm_train_transformer_opt350_en_bpe10000/{perplexity_test_textlm,
#              perplexity_test_unitlm,decode_test_asr}/...
#   slm21    : 来自 log/myrun_eval.1801797.out 的历史打分数字(没有单独的
#              submission 文件可以直接复用,只能贴数字,见下面汇总部分)
#   TTS      : exp/lm_train_transformer_opt350_en_bpe10000/decode_test_tts/
#              text.tts.fb_final/vocoder_eval/checkpoint-400000steps/versa_eval/
#              avg_result.txt(13epoch 自己也跑过三级回退,同一个 rand100 子集)
#
# 本地写入范围(不碰 lm1_k1000_Bal):
#   - PPL/ASR/slm21/TTS 的所有输出都在本目录 exp/sqa_v1_ft、exp/slm21_eval 下
#   - TTS 回退用到的中间 prompt 文件(fb_greedy/fb_topp)写在本地 dump/tts_fallback/,
#     不写进 ${SRC}/dump/raw/test/(那是 lm1_k1000_Bal 自己的目录)
#   - sLM21 数据集: zip 本身、extracted/ 都软链接自 lm1_k1000_Bal(已经解压好
#     的 130400 个 wav,跟 extract_slm21() 的"已存在就跳过"配合,不会触发重新
#     解压)
#   - ckpt(HuBERT SSL权重)是本 recipe自己早就有的软链接,不需要新建。
#     eval_tts_vocoder.sh 里 ground truth(data/test/speech/tts/{wav.scp,text})
#     走绝对路径直接读 lm1_k1000_Bal,不需要额外软链接。
set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_sqa
. ./path.sh
. ./cmd.sh
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1

SRC=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal
LM_EXP=exp/sqa_v1_ft
CKPT=${1:-15epoch.pth}

##############################################################################
# 1) textLM / speechLM PPL(Stage 8)+ ASR WER(Stage 9)
##############################################################################
./lm.sh --stage 8 --stop_stage 9 --skip_train true --nclusters 1000 \
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
#    local/slm21_common.sh 里 LM_CONFIG/LM_FILE 已经指向 ${LM_EXP}/15epoch.pth;
#    如果传了别的 CKPT,注意 local/slm21_common.sh 里的 LM_FILE 是写死的,
#    需要同步手改。
##############################################################################
bash local/eval_slm21_text.sh
bash local/eval_slm21_speech.sh
bash local/eval_slm21_score.sh > "${LM_EXP}/slm21_score.txt"
cat "${LM_EXP}/slm21_score.txt"

##############################################################################
# 3) TTS: 三级回退解码 + vocoder + VERSA(checkpoint 换成 sqa finetune 的,
#    中间文件全部落本地不写回 SRC)
##############################################################################
VOCODER_CKPT=/export/fs06/mliang17/phonologicaltokenizer_modify/ParallelWaveGAN/egs/ljspeech/hubert_voc1/exp/train_nodev_ljspeech_hifigan_hubert.v1/checkpoint-400000steps.pkl
[ -f "${VOCODER_CKPT}" ] || { echo "vocoder checkpoint 不存在: ${VOCODER_CKPT}" >&2; exit 1; }

MAXLEN=1200
D=${LM_EXP}/decode_test_tts
FB=dump/tts_fallback
mkdir -p "${FB}"
LM_CFG=conf/tuning/finetune_sqa_v1_opt350.yaml
COMMON_TTS="--stage 10 --stop_stage 10 --ngpu 1 --nclusters 1000 --gpu_inference true --inference_nj 8 --lang en --token_type bpe --nbpe 10000 --dumpdir ${SRC}/dump --train_set train --valid_set dev --test_sets test --lm_config ${LM_CFG} --lm_exp ${LM_EXP} --inference_lm ${CKPT}"

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
# 汇总: finetune(${CKPT}) vs zero-shot(13epoch,finetune 之前)
##############################################################################
BASE=${SRC}/exp/lm_train_transformer_opt350_en_bpe10000
sumavg() { grep "Sum/Avg" "$1" 2>/dev/null || echo "(找不到 $1)"; }

echo "======================================================================"
echo "==== 汇总: finetune(${CKPT}) vs zero-shot(13epoch) ===="
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
echo "---- sWUGGY/sBLIMP: zero-shot(13epoch,来自 ${SRC}/log/myrun_eval.1801797.out) ----"
cat <<'EOF'
lexical (text): accuracy=82.61% (40000 pairs)
lexical (speech): accuracy=61.43% (40000 pairs)
syntactic (text): accuracy=68.41% (25200 pairs)
syntactic (speech): accuracy=54.35% (25200 pairs)
EOF
echo "---- TTS: finetune(${CKPT}), 三级回退最终结果 ----"
cat "${D}/text.tts.fb_final/vocoder_eval/checkpoint-400000steps/versa_eval/avg_result.txt"
echo "---- TTS: zero-shot(13epoch), 三级回退最终结果 ----"
cat "${BASE}/decode_test_tts/text.tts.fb_final/vocoder_eval/checkpoint-400000steps/versa_eval/avg_result.txt"
echo "(逐句细节见上面 summarize_fallback.py 打印的完整对比表,含每一级回退用了几句)"
