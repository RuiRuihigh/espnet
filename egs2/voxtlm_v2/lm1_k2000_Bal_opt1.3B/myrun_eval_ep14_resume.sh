#!/usr/bin/env bash
#SBATCH --job-name=opt1.3b_eval_ep14_resume
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_opt1.3B/log/myrun_eval_ep14_resume.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_opt1.3B/log/myrun_eval_ep14_resume.%j.err

# myrun_eval_ep14.sh(job 1815704)在sLM21语音轨的lexical_dev_speech打分卡住了:
# 16个并行分片里2个因为 g2p_en 导入时 nltk.download() 撞上
# "PermissionError: /home/mliang17"挂了(/home/mliang17 本身是正常可写的,
# 确认过,应该是那两个分片跑到的计算节点NFS挂载临时抽风,偶发性问题,不是真的
# 权限坏了)。Stage 8/9(PPL/ASR,跑了近30小时)和sLM21文本轨都已经正常跑完,
# 不能整个重跑——这里只续跑没完成的部分:
#   1) sLM21语音轨打分(lexical_dev_speech/syntactic_dev_speech)
#      跳过 eval_slm21_speech.sh 开头的 format_wav_scp.sh/perform_kmeans.sh
#      (那两步已经成功,pseudo_labels_km2000.txt都在),只重跑失败的打分循环
#   2) eval_slm21_score.sh
#   3) TTS三级回退 + vocoder + VERSA(原来就没跑到这一步)
#   4) 最终汇总
set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_opt1.3B
. ./path.sh
. ./cmd.sh
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1

CKPT=14epoch.pth
LM_EXP=exp/lm_train_transformer_opt1.3b_en_bpe10000_ep14
BASE_EXP=exp/lm_train_transformer_opt1.3b_en_bpe10000
SLM21_WORK_EP=exp/slm21_eval_ep14
FB=exp/fallback_ep14
sumavg() { grep "Sum/Avg" "$1" 2>/dev/null || echo "(找不到 $1)"; }

export LM_EXP LM_CONFIG="${LM_EXP}/config.yaml" LM_FILE="${LM_EXP}/${CKPT}" SLM21_WORK="${SLM21_WORK_EP}"
# shellcheck disable=SC1091
. local/slm21_common.sh
use_regular_gpu_partition

for split in lexical_dev syntactic_dev; do
    task=${split%_*}
    pseudo_labels=${SLM21_WORK}/extracted_feats/phonological_tokenizer/layer21/${split}/pseudo_labels_km2000.txt
    prep_dir=${SLM21_WORK}/${split}_speech

    python3 local/prepare_slm21_speech_text.py \
        --pseudo-labels "${pseudo_labels}" \
        --out "${prep_dir}/text_speech.txt"

    run_lm_calc_perplexity_parallel "${prep_dir}/text_speech.txt" "${prep_dir}" 16

    awk '{print $1"\t"$1}' "${prep_dir}/text_speech.txt" > "${prep_dir}/filename_map.tsv"

    python3 local/slm21_score_from_ppl.py \
        --ppl-dir "${prep_dir}/ppl_out" \
        --filename-map "${prep_dir}/filename_map.tsv" \
        --out "${SLM21_SUBMISSION}/${task}/dev_speech.txt"
done
echo "sLM21 speech-track submission ready at ${SLM21_SUBMISSION}/{lexical,syntactic}/dev_speech.txt"

bash local/eval_slm21_score.sh > "${LM_EXP}/slm21_score.txt"
cat "${LM_EXP}/slm21_score.txt"
unset LM_EXP LM_CONFIG LM_FILE SLM21_WORK
LM_EXP=exp/lm_train_transformer_opt1.3b_en_bpe10000_ep14

##############################################################################
# 3) TTS: 三级回退解码 + vocoder + VERSA
##############################################################################
VOCODER_CKPT=/export/fs06/mliang17/phonologicaltokenizer_modify/ParallelWaveGAN/egs/ljspeech/phon_voc1/exp/train_nodev_ljspeech_hifigan_phon.v1/checkpoint-400000steps.pkl
[ -f "${VOCODER_CKPT}" ] || { echo "vocoder checkpoint 不存在: ${VOCODER_CKPT}" >&2; exit 1; }

MAXLEN=1200
D=${LM_EXP}/decode_test_tts
mkdir -p "${FB}"
COMMON="--stage 10 --stop_stage 10 --ngpu 1 --inference_nj 8 --lm_config conf/tuning/train_transformer_opt1.3b.yaml --lm_exp ${LM_EXP} --inference_lm ${CKPT}"

PREV_L1=${BASE_EXP}/decode_test_tts/text.tts.rand100_ep14
if [ -s "${PREV_L1}/vocoder_eval/checkpoint-400000steps/versa_eval/result.1.txt" ]; then
    echo "[L1] 复用 14epoch 已经跑完的 beam5 结果(同一 checkpoint、同一配置、同一批 100 句)"
    mkdir -p "${D}"
    cp -r "${PREV_L1}" "${D}/text.tts.rand100"
else
    echo "[L1] 没有可复用的 14epoch beam5 结果,重新解码 100 句"
    ./run.sh ${COMMON} --python python3 \
        --lm_inference_tts_config conf/decode_lm_tts_maxlen1200.yaml \
        --lm_test_text_tts dump/raw/test/text.tts.rand100
    (
        unset PYTHONPATH
        export PATH="${PWD}/utils:${PATH}"
        export LM_EXP
        # shellcheck disable=SC1091
        source /opt/slurm_files/software/conda25/2025-08-21/etc/profile.d/conda.sh
        conda activate parallelwavegan
        bash local/eval_tts_vocoder.sh "${VOCODER_CKPT}" text.tts.rand100
    )
fi
python3 local/count_capped_steps.py "${D}/text.tts.rand100" ${MAXLEN}
python3 local/fallback_select.py capped "${D}/text.tts.rand100" ${MAXLEN} > "${FB}/capped_L1.txt"
echo "[L1] 撞满 $(wc -l < "${FB}/capped_L1.txt") 句"

LEVELS="${D}/text.tts.rand100"
if [ -s "${FB}/capped_L1.txt" ]; then
    python3 local/fallback_select.py make_text dump/raw/test/text.tts.rand100 "${FB}/capped_L1.txt" "${FB}/text.tts.fb_greedy"
    unset SAMPLE_MODE SAMPLE_TOP_P SAMPLE_TEMP SAMPLE_SEED SAMPLE_PER_UTT_SEED
    ./run.sh ${COMMON} --python local/python_sampling.sh \
        --lm_inference_tts_config conf/decode_lm_tts_beam1_maxlen1200.yaml \
        --lm_test_text_tts "${FB}/text.tts.fb_greedy"
    python3 local/fallback_select.py capped "${D}/text.tts.fb_greedy" ${MAXLEN} > "${FB}/capped_L2.txt"
    echo "[L2] 贪心后仍撞满 $(wc -l < "${FB}/capped_L2.txt") 句"
    LEVELS="${LEVELS} ${D}/text.tts.fb_greedy"

    if [ -s "${FB}/capped_L2.txt" ]; then
        python3 local/fallback_select.py make_text dump/raw/test/text.tts.rand100 "${FB}/capped_L2.txt" "${FB}/text.tts.fb_topp"
        SAMPLE_MODE=topp SAMPLE_TOP_P=0.9 SAMPLE_TEMP=1.0 SAMPLE_SEED=0 SAMPLE_PER_UTT_SEED=1 \
        ./run.sh ${COMMON} --python local/python_sampling.sh \
            --lm_inference_tts_config conf/decode_lm_tts_beam1_maxlen1200.yaml \
            --lm_test_text_tts "${FB}/text.tts.fb_topp"
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
    export LM_EXP
    # shellcheck disable=SC1091
    source /opt/slurm_files/software/conda25/2025-08-21/etc/profile.d/conda.sh
    conda activate parallelwavegan
    bash local/eval_tts_vocoder.sh "${VOCODER_CKPT}" text.tts.fb_final
)
python3 local/summarize_fallback.py "${LM_EXP}"

##############################################################################
# 汇总: 14epoch vs 7epoch(valid.acc.best 那份)
##############################################################################
echo "======================================================================"
echo "==== 汇总: 14epoch vs 7epoch ===="
echo "======================================================================"
echo "---- textLM PPL ----"
echo "14ep: $(cat ${LM_EXP}/perplexity_test_textlm/*/ppl)"
echo " 7ep: $(cat ${BASE_EXP}/perplexity_test_textlm/*/ppl)"
echo "---- speechLM PPL ----"
echo "14ep: $(cat ${LM_EXP}/perplexity_test_unitlm/*/ppl)"
echo " 7ep: $(cat ${BASE_EXP}/perplexity_test_unitlm/*/ppl)"
echo "---- ASR WER(合并) ----"
echo "14ep: $(sumavg ${LM_EXP}/decode_test_asr/decode_lm_asr/score_wer/result.txt)"
echo " 7ep: $(sumavg ${BASE_EXP}/decode_test_asr/decode_lm_asr/score_wer/result.txt)"
for split in test_clean test_other; do
    echo "---- ASR WER (${split}) ----"
    echo "14ep: $(sumavg ${LM_EXP}/decode_test_asr/decode_lm_asr/score_wer/${split}/result.txt)"
    echo " 7ep: $(sumavg ${BASE_EXP}/decode_test_asr/decode_lm_asr/score_wer/${split}/result.txt)"
done
echo "---- sWUGGY/sBLIMP ----"
echo "14ep:"; cat "${LM_EXP}/slm21_score.txt"
echo " 7ep:"; cat "${BASE_EXP}/slm21_score.txt"
echo "---- TTS 三级回退 ----"
echo "14ep:"; cat "${D}/text.tts.fb_final/vocoder_eval/checkpoint-400000steps/versa_eval/avg_result.txt"
echo " 7ep:"; cat "${BASE_EXP}/decode_test_tts/text.tts.fb_final/vocoder_eval/checkpoint-400000steps/versa_eval/avg_result.txt"
