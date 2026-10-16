#!/usr/bin/env bash
# SALMon (https://github.com/slp-rl/salmon) acoustic-consistency / alignment
# evaluation for this recipe's speechLM. Pure likelihood comparison of
# (positive, negative) wav pairs -- no generation, no vocoder.
#
# Same pipeline as ../../voxtlm_v2/lm1_k2000_Bal/local/eval_salmon.sh; only the
# tokenizer differs (this recipe: wavlm_large layer 21 + the km model below,
# i.e. exactly what local/eval_slm21_speech.sh and training data used):
#   1) check SALMON_ROOT = v2/lm1_k2000_Bal's downloads/SALMon (HF
#      slprl/SALMon parquet + the 3200 unpacked wavs), read in place by
#      absolute path. Nothing is downloaded or written there.
#   2) build 2 datadirs from it (local/prepare_salmon.py --read-only: fails
#      instead of writing if a wav were missing)
#   3) perform_kmeans.sh stage 3 only (--skip_stages "1 2"): wavlm_large L21
#      features + KM_DIR labels
#   4) unit2cjk + "<generatespeech>" prefix, no dedup
#      (local/prepare_slm21_speech_text.py), lm_calc_perplexity on all 3200
#      utterances (run_lm_calc_perplexity_parallel from local/slm21_common.sh;
#      same LM_CONFIG / LM_FILE = valid.acc.best.pth as sLM21)
#   5) per-part accuracy, official rule: whole-sequence loglik(pos) > loglik(neg)
#
# Writes only under exp/salmon_eval (new real dir). cmd.sh is not edited:
# cuda_cmd is overridden in this shell to the regular gpu partition.
# Read-only inputs: KM_DIR, ckpt/ (wavlm_large.pt; s3prl only
# touches its .lock file there, same as the sLM21 speech track already does).
#
# Usage: local/eval_salmon.sh [--stage N] [--stop_stage N] [--nj N]  (from the recipe root)

set -e
set -u
set -o pipefail

stage=1
stop_stage=5
nj=8

# shellcheck disable=SC1091
. ./path.sh
# shellcheck disable=SC1091
. ./cmd.sh
# shellcheck disable=SC1091
. utils/parse_options.sh
# shellcheck disable=SC1091
. local/slm21_common.sh

cuda_cmd="slurm.pl --config conf/slurm.conf"

# Tokenizer: must match this recipe's Stage 3 / local/eval_slm21_speech.sh.
KM_DIR=exp/kmeans/wavlm_large_21_2000clusters
NCLUSTERS=2000
FEATURE_TYPE=wavlm_large
LAYER=21
FEATURE_CONF='{type=s3prl,conf={s3prl_conf={upstream=wavlm_large},download_dir=ckpt,multilayer_feature=False,layer=21}}'

# Read-only, absolute path on purpose: no symlink is added under downloads/,
# since some downloads/ dirs here are shared with other recipes via symlink
# (e.g. lm1_k1000_Bal_opt1.3B/downloads -> lm1_k1000_Bal/downloads).
SALMON_ROOT=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal/downloads/SALMon
SALMON_WORK=exp/salmon_eval
DATADIR=${SALMON_WORK}/audio_datadir
FEATDIR=${SALMON_WORK}/extracted_feats
LM_DIR=${SALMON_WORK}/lm

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: check ${SALMON_ROOT}"
    if [ ! -f "${SALMON_ROOT}/hf/.complete" ]; then
        echo "${SALMON_ROOT}/hf/.complete not found."
        echo "(run local/eval_salmon.sh --stop_stage 2 in voxtlm_v2/lm1_k2000_Bal first)"
        exit 1
    fi
    n=$(find -L "${SALMON_ROOT}/wav" -name "*.wav" | wc -l)
    [ "${n}" -eq 3200 ] || { echo "expected 3200 wavs under ${SALMON_ROOT}/wav, found ${n}"; exit 1; }
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: datadirs"
    python3 local/prepare_salmon.py --read-only \
        --hf-dir "${SALMON_ROOT}/hf" \
        --wav-dir "${SALMON_ROOT}/wav" \
        --data-dir "${DATADIR}"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: ${FEATURE_TYPE} L${LAYER} + km${NCLUSTERS} labels"
    scripts/feats/perform_kmeans.sh \
        --stage 1 --stop-stage 3 \
        --skip_stages "1 2" \
        --train_set salmon_consistency \
        --dev_set salmon_alignment \
        --other_sets "" \
        --datadir "${DATADIR}" \
        --featdir "${FEATDIR}" \
        --audio_format wav \
        --feature_type "${FEATURE_TYPE}" \
        --layer "${LAYER}" \
        --feature_conf "${FEATURE_CONF}" \
        --km_dir "${KM_DIR}" \
        --nclusters "${NCLUSTERS}" \
        --storage_save_mode true \
        --use_gpu true \
        --nj "${nj}" \
        --cpu_cmd "${train_cmd}" \
        --cuda_cmd "${cuda_cmd}"
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: speechLM log-likelihood"
    mkdir -p "${LM_DIR}"
    for group in salmon_consistency salmon_alignment; do
        cat "${FEATDIR}/${FEATURE_TYPE}/layer${LAYER}/${group}/pseudo_labels_km${NCLUSTERS}.txt"
    done | sort > "${LM_DIR}/pseudo_labels_km${NCLUSTERS}.txt"
    n=$(<"${LM_DIR}/pseudo_labels_km${NCLUSTERS}.txt" wc -l)
    [ "${n}" -eq 3200 ] || { echo "expected 3200 labeled utterances (1600 pairs), got ${n}"; exit 1; }

    python3 local/prepare_slm21_speech_text.py \
        --pseudo-labels "${LM_DIR}/pseudo_labels_km${NCLUSTERS}.txt" \
        --out "${LM_DIR}/text_speech.txt"

    run_lm_calc_perplexity_parallel "${LM_DIR}/text_speech.txt" "${LM_DIR}" "${nj}"
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: accuracy"
    python3 local/salmon_pair_accuracy.py \
        --ppl-dir "${LM_DIR}/ppl_out" \
        --out "${SALMON_WORK}/results.tsv"
    echo "results: ${SALMON_WORK}/results.tsv"
fi
