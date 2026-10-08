#!/usr/bin/env bash
#SBATCH --job-name=3M_eval_full
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_3M/log/myrun_eval_full.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_3M/log/myrun_eval_full.%j.err

# 用法: sbatch myrun_eval_full.sh [valid.acc.best.pth]
#
# lm1_k2000_3M(D_3M: textlm/speechlm 数据量放大到 3M, asr/tts/dev 不变,OPT-350M
# 训练到 70epoch 附近开始平台,valid.acc.best.pth -> 70epoch.pth)的完整评测,
# 跟 lm1_k2000_Bal(D_Bal 同模型同超参,350M 基线)做对照,顺序跑完:
#   1) textLM / speechLM PPL(Stage 8)+ ASR WER(Stage 9,自动拆 test-clean/other)
#   2) sWUGGY(lexical) + sBLIMP(syntactic),文本轨 + 语音轨
#   3) TTS: 三级回退解码(beam5 撞满maxlen -> 贪心 -> top-p 0.9 采样)+ vocoder + VERSA,
#      100 句固定随机子集(dump/raw/test/text.tts.rand100,跟 opt1.3B/Bal 同一批句子)
#
# 本地写入范围(不写任何软链接指向的地方):
#   - dump/、exp/ 在本目录是真实目录,所有输出(PPL/ASR/TTS/slm21/回退中间文件)都在这里
#   - downloads/(sLM21、phonological_tokenizer、LibriTTS 等)、exp/kmeans、data/test
#     经软链接指向别的 recipe,全部只读:extract_slm21() 是"已存在就跳过"写法,
#     kmeans 只读加载,TTS gt 参照走 data/test 的只读路径
#   - 三级回退的中间 prompt 写到 exp/fallback/,不写进 dump/raw/test/
set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_3M
. ./path.sh
. ./cmd.sh
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1

LM_EXP=exp/lm_train_transformer_opt350_en_bpe10000
LM_CFG=conf/tuning/train_transformer_opt350.yaml
CKPT=${1:-valid.acc.best.pth}
BASE=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal/exp/lm_train_transformer_opt350_en_bpe10000
sumavg() { grep "Sum/Avg" "$1" 2>/dev/null || echo "(找不到 $1)"; }

##############################################################################
# 1) textLM / speechLM PPL(Stage 8)+ ASR WER(Stage 9)
##############################################################################
./lm.sh --stage 8 --stop_stage 9 --skip_train true \
    --lang en --token_type bpe --nbpe 10000 \
    --train_set train --valid_set dev --test_sets test \
    --lm_exp "${LM_EXP}" --inference_lm "${CKPT}" \
    --gpu_inference true --inference_nj 8 \
    --lm_inference_asr_config conf/decode_lm_asr.yaml \
    --lm_test_text_asr dump/raw/test/text.asr \
    --lm_test_text_textlm dump/raw/test/text.textlm \
    --lm_test_text_speechlm dump/raw/test/text.speechlm

##############################################################################
# 2) sWUGGY(lexical) + sBLIMP(syntactic),文本轨 + 语音轨 + 打分
#    local/slm21_common.sh 里 LM_CONFIG/LM_FILE 指向本目录 valid.acc.best.pth
##############################################################################
bash local/eval_slm21_text.sh
bash local/eval_slm21_speech.sh
bash local/eval_slm21_score.sh > "${LM_EXP}/slm21_score.txt"
cat "${LM_EXP}/slm21_score.txt"

##############################################################################
# 3) TTS: 三级回退解码 + vocoder + VERSA
##############################################################################
VOCODER_CKPT=/export/fs06/mliang17/phonologicaltokenizer_modify/ParallelWaveGAN/egs/ljspeech/phon_voc1/exp/train_nodev_ljspeech_hifigan_phon.v1/checkpoint-400000steps.pkl
[ -f "${VOCODER_CKPT}" ] || { echo "vocoder checkpoint 不存在: ${VOCODER_CKPT}" >&2; exit 1; }

MAXLEN=1200
D=${LM_EXP}/decode_test_tts
F=exp/fallback
mkdir -p "${F}"
COMMON="--stage 10 --stop_stage 10 --ngpu 1 --inference_nj 8 --lm_config ${LM_CFG} --inference_lm ${CKPT}"

n1=0; [ -s "${D}/text.tts.rand100/score_tts/hyp.tok" ] && n1=$(wc -l < "${D}/text.tts.rand100/score_tts/hyp.tok")
ml1=""; [ -f "${D}/text.tts.rand100/logdir/lm_inference.1.log" ] && ml1=$(grep -m1 -o "max output length: \[[0-9]*\]" "${D}/text.tts.rand100/logdir/lm_inference.1.log" || true)
if [ "${n1}" -eq 100 ] && [ "${ml1}" = "max output length: [${MAXLEN}]" ]; then
    echo "[L1] 复用已有的 beam5 结果 text.tts.rand100(${n1} 句, ${ml1})"
else
    echo "[L1] 没有可用的 beam5 结果(hyp.tok=${n1} 行, '${ml1}'),重新解码 100 句"
    ./run.sh ${COMMON} --python python3 \
        --lm_inference_tts_config conf/decode_lm_tts_maxlen1200.yaml \
        --lm_test_text_tts dump/raw/test/text.tts.rand100
fi
python3 local/count_capped_steps.py "${D}/text.tts.rand100" ${MAXLEN}
python3 local/fallback_select.py capped "${D}/text.tts.rand100" ${MAXLEN} > "${F}/capped_L1.txt"
echo "[L1] 撞满 $(wc -l < "${F}/capped_L1.txt") 句"

rm -rf "${D}/text.tts.fb_greedy" "${D}/text.tts.fb_topp" "${D}/text.tts.fb_final"
LEVELS="${D}/text.tts.rand100"

if [ -s "${F}/capped_L1.txt" ]; then
    python3 local/fallback_select.py make_text dump/raw/test/text.tts.rand100 "${F}/capped_L1.txt" ${F}/text.tts.fb_greedy
    unset SAMPLE_MODE SAMPLE_TOP_P SAMPLE_TEMP SAMPLE_SEED SAMPLE_PER_UTT_SEED
    ./run.sh ${COMMON} --python local/python_sampling.sh \
        --lm_inference_tts_config conf/decode_lm_tts_beam1_maxlen1200.yaml \
        --lm_test_text_tts ${F}/text.tts.fb_greedy
    python3 local/fallback_select.py capped "${D}/text.tts.fb_greedy" ${MAXLEN} > "${F}/capped_L2.txt"
    echo "[L2] 贪心后仍撞满 $(wc -l < "${F}/capped_L2.txt") 句"
    LEVELS="${LEVELS} ${D}/text.tts.fb_greedy"

    if [ -s "${F}/capped_L2.txt" ]; then
        python3 local/fallback_select.py make_text dump/raw/test/text.tts.rand100 "${F}/capped_L2.txt" ${F}/text.tts.fb_topp
        SAMPLE_MODE=topp SAMPLE_TOP_P=0.9 SAMPLE_TEMP=1.0 SAMPLE_SEED=0 SAMPLE_PER_UTT_SEED=1 \
        ./run.sh ${COMMON} --python local/python_sampling.sh \
            --lm_inference_tts_config conf/decode_lm_tts_beam1_maxlen1200.yaml \
            --lm_test_text_tts ${F}/text.tts.fb_topp
        LEVELS="${LEVELS} ${D}/text.tts.fb_topp"
    else
        echo "[L3] 贪心已经解决了所有撞满的句子,不需要 top-p"
    fi
else
    echo "[L2/L3] 第1级没有撞满的句子,不需要回退"
fi

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
# 汇总: 3M(${CKPT}) vs 350M 基线(lm1_k2000_Bal, 16epoch)
##############################################################################
echo "======================================================================"
echo "==== 汇总: lm1_k2000_3M(${CKPT}) vs lm1_k2000_Bal(16epoch) ===="
echo "======================================================================"
echo "---- textLM PPL ----"
echo "3M  : $(cat ${LM_EXP}/perplexity_test_textlm/*/ppl)"
echo "Bal : $(cat ${BASE}/perplexity_test_textlm/*/ppl)"
echo "---- speechLM PPL ----"
echo "3M  : $(cat ${LM_EXP}/perplexity_test_unitlm/*/ppl)"
echo "Bal : $(cat ${BASE}/perplexity_test_unitlm/*/ppl)"
echo "---- ASR WER(合并) ----"
echo "3M  : $(sumavg ${LM_EXP}/decode_test_asr/decode_lm_asr/score_wer/result.txt)"
echo "Bal : $(sumavg ${BASE}/decode_test_asr/decode_lm_asr/score_wer/result.txt)"
for split in test_clean test_other; do
    echo "---- ASR WER (${split}) ----"
    echo "3M  : $(sumavg ${LM_EXP}/decode_test_asr/decode_lm_asr/score_wer/${split}/result.txt)"
    echo "Bal : $(sumavg ${BASE}/decode_test_asr/decode_lm_asr/score_wer/${split}/result.txt)"
done
echo "---- sWUGGY/sBLIMP: 3M(${CKPT}) ----"
cat "${LM_EXP}/slm21_score.txt"
echo "---- sWUGGY/sBLIMP: Bal 350M(16epoch, 来自 log/myrun_slm21.1801318.out) ----"
cat <<'EOF'
lexical (text): accuracy=82.16% (40000 pairs)
lexical (speech): accuracy=66.98% (40000 pairs)
syntactic (text): accuracy=68.83% (25200 pairs)
syntactic (speech): accuracy=57.31% (25200 pairs)
EOF
echo "---- TTS: 3M(${CKPT}), 三级回退最终结果 ----"
cat "${D}/text.tts.fb_final/vocoder_eval/checkpoint-400000steps/versa_eval/avg_result.txt"
echo "---- TTS: Bal 350M(16epoch), 三级回退最终结果 ----"
cat "${BASE}/decode_test_tts/text.tts.fb_final/vocoder_eval/checkpoint-400000steps/versa_eval/avg_result.txt"
echo "(逐句细节见上面 summarize_fallback.py 打印的完整对比表)"
