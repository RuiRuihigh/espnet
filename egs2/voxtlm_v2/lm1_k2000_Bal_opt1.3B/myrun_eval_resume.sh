#!/usr/bin/env bash
#SBATCH --job-name=opt1.3b_eval_resume
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_opt1.3B/log/myrun_eval_resume.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_opt1.3B/log/myrun_eval_resume.%j.err

# 用法: sbatch myrun_eval_resume.sh [valid.acc.best.pth]
#
# 训练过拟合(8/9epoch 之后 valid acc 掉,7epoch 是最优点,valid.acc.best.pth 已经
# 指向 7epoch)后的首次全套评测,顺序跑完(不是 finetune,没有旧任务要对比遗忘,
# 直接跟 lm1_k2000_Bal 自己的 350M/16epoch 数字做首次全量对照):
#   1) textLM / speechLM PPL(Stage 8)+ ASR WER(Stage 9,自动拆 test-clean/other)
#   2) sWUGGY(lexical) + sBLIMP(syntactic),文本轨 + 语音轨
#   3) TTS: 三级回退解码(beam5 撞满maxlen -> 贪心 -> top-p 0.9 采样)+ vocoder + VERSA,
#      100 句固定随机子集 —— 没有跑全量语料(9957条)的老式 beam5 解码,那是 350M
#      早期用过、后来发现会被重复循环撞满 maxlen 污染数据的方法,已经被回退法取代。
#
# sLM21 数据集不需要重新下载/解压: downloads/ 是整目录软链接到 lm1_k2000_Bal,
# 那边已经解压好全部 130400 个 wav + 两个 gold.csv,extract_slm21() 是"文件已存在
# 就跳过"的写法,这里直接复用,不会有任何解压动作,也不会有写操作。
#
# 本地写入范围(不碰 lm1_k2000_Bal,也不碰任何软链接指向的原始数据):
#   - PPL/ASR/TTS 解码结果全部落在本地 exp/lm_train_transformer_opt1.3b_en_bpe10000/
#   - slm21 结果落在本地 exp/slm21_eval/(SLM21_WORK,local/slm21_common.sh 里定义)
#   - TTS 回退中间 prompt 文件(fb_greedy/fb_topp)落在本地 exp/fallback/,不写进
#     dump/raw/test/(dump/ 在本目录是整目录软链接到 lm1_k2000_Bal,直接写那里的
#     文件会写穿到原始数据,已经改成写本地路径)
#   - sLM21 的 downloads/、TTS 的 data/test、Stage8/9 的 dump/raw/test/* 全部只读
set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_opt1.3B
. ./path.sh
. ./cmd.sh
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1

LM_EXP=exp/lm_train_transformer_opt1.3b_en_bpe10000
CKPT=${1:-valid.acc.best.pth}
BASE=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal/exp/lm_train_transformer_opt350_en_bpe10000
sumavg() { grep "Sum/Avg" "$1" 2>/dev/null || echo "(找不到 $1)"; }


##############################################################################
# 续跑: Stage8/9(PPL/ASR)和 slm21 文本轨已经跑完,文本轨 submission 文件都在;
# 语音轨失败是因为 exp/kmeans 软链接漏建(已经补上),从语音轨重新开始,不重跑文本轨
##############################################################################
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
COMMON="--stage 10 --stop_stage 10 --ngpu 1 --inference_nj 8 --lm_config conf/tuning/train_transformer_opt1.3b.yaml --inference_lm ${CKPT}"

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
# 汇总: opt1.3b(${CKPT}) vs opt350(16epoch)
##############################################################################
echo "======================================================================"
echo "==== 汇总: opt1.3b(${CKPT}) vs opt350(16epoch) ===="
echo "======================================================================"
echo "---- textLM PPL ----"
echo "opt1.3b: $(cat ${LM_EXP}/perplexity_test_textlm/*/ppl)"
echo "opt350 : $(cat ${BASE}/perplexity_test_textlm/*/ppl)"
echo "---- speechLM PPL ----"
echo "opt1.3b: $(cat ${LM_EXP}/perplexity_test_unitlm/*/ppl)"
echo "opt350 : $(cat ${BASE}/perplexity_test_unitlm/*/ppl)"
echo "---- ASR WER(合并) ----"
echo "opt1.3b: $(sumavg ${LM_EXP}/decode_test_asr/decode_lm_asr/score_wer/result.txt)"
echo "opt350 : $(sumavg ${BASE}/decode_test_asr/decode_lm_asr/score_wer/result.txt)"
for split in test_clean test_other; do
    echo "---- ASR WER (${split}) ----"
    echo "opt1.3b: $(sumavg ${LM_EXP}/decode_test_asr/decode_lm_asr/score_wer/${split}/result.txt)"
    echo "opt350 : $(sumavg ${BASE}/decode_test_asr/decode_lm_asr/score_wer/${split}/result.txt)"
done
echo "---- sWUGGY/sBLIMP: opt1.3b ----"
cat "${LM_EXP}/slm21_score.txt"
echo "---- sWUGGY/sBLIMP: opt350(16epoch,来自 ${BASE}/../../log/myrun_slm21.1801318.out) ----"
cat <<'EOF'
lexical (text): accuracy=82.16% (40000 pairs)
lexical (speech): accuracy=66.98% (40000 pairs)
syntactic (text): accuracy=68.83% (25200 pairs)
syntactic (speech): accuracy=57.31% (25200 pairs)
EOF
echo "---- TTS: opt1.3b, 三级回退最终结果 ----"
cat "${D}/text.tts.fb_final/vocoder_eval/checkpoint-400000steps/versa_eval/avg_result.txt"
echo "---- TTS: opt350(16epoch), 三级回退最终结果 ----"
cat "${BASE}/decode_test_tts/text.tts.fb_final/vocoder_eval/checkpoint-400000steps/versa_eval/avg_result.txt"
echo "(逐句细节见上面 summarize_fallback.py 打印的完整对比表,含每一级回退用了几句)"
