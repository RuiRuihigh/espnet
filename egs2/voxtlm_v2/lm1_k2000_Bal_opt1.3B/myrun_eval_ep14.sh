#!/usr/bin/env bash
#SBATCH --job-name=opt1.3b_eval_ep14
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_opt1.3B/log/myrun_eval_ep14.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_opt1.3B/log/myrun_eval_ep14.%j.err

# 14epoch 的完整评测(PPL/ASR、sLM21、TTS三级回退+vocoder+VERSA),跟 7epoch
# (exp/lm_train_transformer_opt1.3b_en_bpe10000,valid.acc.best 那份)对照。
#
# 隔离: 14epoch 的所有输出都写到独立的 exp/lm_train_transformer_opt1.3b_en_bpe10000_ep14/
# (config.yaml 和 14epoch.pth 是指向 7epoch 那边的只读软链接),7epoch 的结果一个字节都
# 不动。sLM21 的 wav 和 hubert 特征是跟模型无关的输入,整目录复制到 exp/slm21_eval_ep14/
# 而不是共用,submission 文件才不会互相覆盖。
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

##############################################################################
# 0) 复制跟模型无关的 sLM21 输入(只复制一次,已存在就跳过)
##############################################################################
mkdir -p "${SLM21_WORK_EP}"
for d in audio_datadir extracted_feats; do
    if [ ! -d "${SLM21_WORK_EP}/${d}" ]; then
        cp -r "exp/slm21_eval/${d}" "${SLM21_WORK_EP}/${d}"
    fi
done

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
# 2) sWUGGY(lexical) + sBLIMP(syntactic)
##############################################################################
export LM_EXP LM_CONFIG="${LM_EXP}/config.yaml" LM_FILE="${LM_EXP}/${CKPT}" SLM21_WORK="${SLM21_WORK_EP}"
bash local/eval_slm21_text.sh
bash local/eval_slm21_speech.sh
bash local/eval_slm21_score.sh > "${LM_EXP}/slm21_score.txt"
cat "${LM_EXP}/slm21_score.txt"
unset LM_EXP LM_CONFIG LM_FILE SLM21_WORK

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
