#!/usr/bin/env bash
# SALMon (https://github.com/slp-rl/salmon) acoustic-consistency / alignment
# evaluation for this recipe's speechLM. Pure likelihood comparison of
# (positive, negative) wav pairs -- no generation, no vocoder.
#
# Pipeline mirrors local/eval_slm21_speech.sh step for step, so the speech side
# is tokenized exactly like training data and like the sLM21 speech track:
#   1) downloads/SALMon/hf (slprl/SALMon parquet) -> wavs + 2 datadirs
#      (local/prepare_salmon.py; data is already 16 kHz mono, so no
#      format_wav_scp.sh step)
#   2) perform_kmeans.sh stage 3 only, with the released Phonological Tokenizer
#      (downloads/phonological_tokenizer/ssl.pth, layer 21) and
#      exp/kmeans/phonological_tokenizer_2000clusters -- both read-only here
#   3) unit2cjk + "<generatespeech>" prefix, no dedup
#      (local/prepare_slm21_speech_text.py, reused as-is)
#   4) lm_calc_perplexity on all 3200 utterances (1600 pairs) in one sharded run
#      (run_lm_calc_perplexity_parallel from local/slm21_common.sh; same
#      LM_CONFIG / LM_FILE = valid.acc.best.pth as sLM21)
#   5) per-part accuracy, official rule: whole-sequence loglik(pos) > loglik(neg)
#
# Everything is written under downloads/SALMon and exp/salmon_eval only (both
# new real directories in this recipe); no symlinked/shared dir is written to.
#
# cmd.sh is NOT edited (unlike use_regular_gpu_partition in slm21_common.sh):
# cuda_cmd is just overridden in this shell to the regular gpu partition, so
# whatever cmd.sh currently says is left untouched for other jobs.
#
# Usage: local/eval_salmon.sh [--stage N] [--stop_stage N]   (from the recipe root)
#   stage 1: download (login node only -- compute nodes have no internet)
#   stage 2: prepare wavs/datadirs
#   stage 3: tokenize (GPU jobs)
#   stage 4: LM scoring (GPU jobs)
#   stage 5: accuracy

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

SALMON_HF=downloads/SALMon/hf
SALMON_WORK=exp/salmon_eval
DATADIR=${SALMON_WORK}/audio_datadir
FEATDIR=${SALMON_WORK}/extracted_feats
LM_DIR=${SALMON_WORK}/lm

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: download slprl/SALMon -> ${SALMON_HF}"
    if [ -f "${SALMON_HF}/.complete" ]; then
        echo "already downloaded, skipping"
    else
        mkdir -p "${SALMON_HF}"
        python3 -c "from huggingface_hub import snapshot_download; snapshot_download('slprl/SALMon', repo_type='dataset', local_dir='${SALMON_HF}')"
        touch "${SALMON_HF}/.complete"
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: parquet -> wavs + datadirs"
    python3 local/prepare_salmon.py \
        --hf-dir "${SALMON_HF}" \
        --wav-dir downloads/SALMon/wav \
        --data-dir "${DATADIR}"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Phonological Tokenizer + km2000 labels"
    phonological_tokenizer_conf="{type=phonological_tokenizer,conf={ssl_model_path=downloads/phonological_tokenizer/ssl.pth,layer=21}}"
    scripts/feats/perform_kmeans.sh \
        --stage 1 --stop-stage 3 \
        --skip_stages "1 2" \
        --train_set salmon_consistency \
        --dev_set salmon_alignment \
        --other_sets "" \
        --datadir "${DATADIR}" \
        --featdir "${FEATDIR}" \
        --audio_format wav \
        --feature_type phonological_tokenizer \
        --layer 21 \
        --feature_conf "${phonological_tokenizer_conf}" \
        --km_dir exp/kmeans/phonological_tokenizer_2000clusters \
        --nclusters 2000 \
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
        cat "${FEATDIR}/phonological_tokenizer/layer21/${group}/pseudo_labels_km2000.txt"
    done | sort > "${LM_DIR}/pseudo_labels_km2000.txt"
    n=$(<"${LM_DIR}/pseudo_labels_km2000.txt" wc -l)
    [ "${n}" -eq 3200 ] || { echo "expected 3200 labeled utterances (1600 pairs), got ${n}"; exit 1; }

    python3 local/prepare_slm21_speech_text.py \
        --pseudo-labels "${LM_DIR}/pseudo_labels_km2000.txt" \
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
