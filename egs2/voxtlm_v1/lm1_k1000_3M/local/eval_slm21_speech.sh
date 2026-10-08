#!/usr/bin/env bash
# sWUGGY(lexical)+ sBLIMP(syntactic) discrete-unit (speechLM) track evaluation.
#
# Only dev, not test -- same reasoning as local/eval_slm21_text.sh (the
# paper reports dev-set numbers). Shares the same gold.csv as the text
# track, but this time actually processes audio: extract wavs -> build a
# kaldi-style wav.scp -> format_wav_scp.sh (resample to 16k, also produces
# utt2num_samples) -> perform_kmeans.sh labels with the already-trained
# official km_1000.mdl (--skip_train_kmeans true, no retraining; this is the
# SAME model this recipe's Stage 3 labeled its training data with, so the unit
# ids here mean exactly what the LM was trained on,
# reuses the existing model) -> each pseudo-label sequence is mapped to CJK
# pseudo-characters the same way local/prepare_lm_data.py's unit2cjk does
# (local/prepare_slm21_speech_text.py's mapping has been checked
# character-for-character against the original, including that neither
# deduplicates repeated consecutive units -- training data doesn't either,
# it's one symbol per frame concatenated as-is).
#
# Key difference from the text track: audio can't be deduplicated by content
# the way text can (the same word said by a different voice is acoustically
# distinct), so all of dev's 80,000+50,400=130,400 recordings actually get
# run through feature extraction + labeling. Restricting to dev instead of
# dev+test already skips the much bigger chunk (test would have been
# 446,000 more), but this is still the heaviest of the two tracks.
# Scoring reuses the same run_lm_calc_perplexity_parallel helper as the text
# track (see local/slm21_common.sh), at nj=16 to match the nj already used
# for format_wav_scp.sh/perform_kmeans.sh, since this data is larger
# (undeduplicated) than the text track's. Kept at 16 rather than higher --
# the regular gpu partition only has a handful of nodes, and this project
# already keeps many other jobs queued on it.
#
# Usage: local/eval_slm21_speech.sh (run from the recipe root, egs2/voxtlm_v1/lm1)

set -e
set -u
set -o pipefail

# shellcheck disable=SC1091
. ./path.sh
# shellcheck disable=SC1091
. ./cmd.sh
# shellcheck disable=SC1091
. local/slm21_common.sh
use_regular_gpu_partition

mkdir -p downloads/sLM21/extracted
extract_slm21 downloads/sLM21/extracted "lexical/dev/*.wav" "syntactic/dev/*.wav"

SLM21_AUDIO_ROOT=${SLM21_WORK}/audio_datadir
declare -A slm21_wav_src=(
    [lexical_dev]=downloads/sLM21/extracted/lexical/dev
    [syntactic_dev]=downloads/sLM21/extracted/syntactic/dev
)
for split in "${!slm21_wav_src[@]}"; do
    _raw_dir=${SLM21_AUDIO_ROOT}/raw/${split}
    _dir=${SLM21_AUDIO_ROOT}/${split}
    mkdir -p "${_raw_dir}" "${_dir}"
    # Build the raw wav.scp/utt2spk in a *separate* staging dir, not ${_dir}
    # itself: format_wav_scp.sh's <out-datadir> argument is where it writes
    # its own reformatted wav.scp, so if in-wav.scp and out-datadir were the
    # same path it would overwrite/clobber its own input while still
    # reading it (this is exactly what broke the first run here -- parallel
    # shards racing to read a wav.scp format_wav_scp.sh had already started
    # overwriting). Every other use of format_wav_scp.sh in
    # this recipe already keeps the source datadir and the formatted output
    # datadir as two different directories; this now does too.
    #
    # wav.scp/utt2spk use the extracted filename as uttid directly -- i.e.
    # gold.csv's own "filename" column -- so scores land on the right
    # submission line with no separate id<->filename map needed (unlike the
    # text track, which has to dedup and then expand back out).
    find "${slm21_wav_src[${split}]}" -name "*.wav" | while read -r wav; do
        uttid=$(basename "${wav}" .wav)
        echo "${uttid} ${wav}"
    done | sort > "${_raw_dir}/wav.scp"
    awk '{print $1, $1}' "${_raw_dir}/wav.scp" > "${_raw_dir}/utt2spk"

    scripts/audio/format_wav_scp.sh --nj 16 --cmd "${train_cmd}" \
        --audio-format flac --fs 16k \
        "${_raw_dir}/wav.scp" "${_dir}"
    echo "raw" > "${_dir}/feats_type"
    echo "flac" > "${_dir}/audio_format"
    # format_wav_scp.sh only (re)writes wav.scp/utt2num_samples into
    # ${_dir} -- it doesn't manage utt2spk/spk2utt, so those have to be
    # copied/generated separately from the staging dir where utt2spk was
    # built. perform_kmeans.sh needs both alongside wav.scp in the same
    # datadir: utt2spk for stage 1 (utils/subset_data_dir.sh) and spk2utt
    # for stage 3 (apply_map.pl on spk2utt when generating pseudo-labels).
    cp "${_raw_dir}/utt2spk" "${_dir}/utt2spk"
    utils/utt2spk_to_spk2utt.pl "${_dir}/utt2spk" > "${_dir}/spk2utt"
done

# Only two datasets now, so train_set/dev_set take one each (those labels
# don't mean "training"/"validation" here -- just reusing
# perform_kmeans.sh's own multi-dataset interface to cover both splits in
# one call instead of extracting features twice); other_sets stays empty so
# it doesn't try to process a third, nonexistent dataset.
scripts/feats/perform_kmeans.sh \
    --stage 1 --stop-stage 3 \
    --skip_train_kmeans true \
    --train_set lexical_dev \
    --dev_set syntactic_dev \
    --other_sets "" \
    --datadir "${SLM21_AUDIO_ROOT}" \
    --featdir "${SLM21_WORK}/extracted_feats" \
    --audio_format flac \
    --feature_type hubert_base \
    --layer 6 \
    --feature_conf '{type=s3prl,conf={s3prl_conf={upstream=hubert_base},download_dir=ckpt,multilayer_feature=False,layer=6}}' \
    --km_dir /export/fs06/mliang17/shared_kmeans/hubert_base_l6_k1000_official \
    --nclusters 1000 \
    --storage_save_mode true \
    --use_gpu true \
    --nj 16 \
    --cpu_cmd "${train_cmd}" \
    --cuda_cmd "${cuda_cmd}"

for split in lexical_dev syntactic_dev; do
    task=${split%_*}
    pseudo_labels=${SLM21_WORK}/extracted_feats/hubert_base/layer6/${split}/pseudo_labels_km1000.txt
    prep_dir=${SLM21_WORK}/${split}_speech

    python3 local/prepare_slm21_speech_text.py \
        --pseudo-labels "${pseudo_labels}" \
        --out "${prep_dir}/text_speech.txt"

    run_lm_calc_perplexity_parallel "${prep_dir}/text_speech.txt" "${prep_dir}" 16

    # uttid already equals filename (audio can't be deduplicated), so an
    # identity map is enough -- reuse the text track's expansion script as-is.
    awk '{print $1"\t"$1}' "${prep_dir}/text_speech.txt" > "${prep_dir}/filename_map.tsv"

    python3 local/slm21_score_from_ppl.py \
        --ppl-dir "${prep_dir}/ppl_out" \
        --filename-map "${prep_dir}/filename_map.tsv" \
        --out "${SLM21_SUBMISSION}/${task}/dev_speech.txt"
done

echo "sLM21 speech-track submission ready at ${SLM21_SUBMISSION}/{lexical,syntactic}/dev_speech.txt"
