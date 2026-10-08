#!/usr/bin/env bash
# Drop utterances longer than a duration threshold from a Kaldi-style data
# dir, in place, based on its utt2num_samples.
#
# Why: the old LibriLight VAD segmentation (local/librilight/cut_by_vad.py)
# never splits a long uninterrupted VAD segment, so speechlm/train keeps a
# tail of very long utterances (up to ~37 min). HuBERT feature extraction in
# Stage 3 does a full-sequence forward pass per utterance, and these OOM on
# ~10GB GPUs. They never make it into LM training anyway: BPE'd speech runs at
# >= 23.2 tokens/s (measured on the old checkout, audio >30s), so anything
# over ~65s exceeds the 1500-token cap applied after Stage 6. Dropping them
# before Stage 3 leaves the final training set unchanged as long as max_sec
# is comfortably above 65s.
#
# Safe to rerun: filtering an already-filtered dir changes nothing, and if
# Stage 2 regenerates the dir it simply gets filtered again. The original
# files are backed up once, to <data-dir>/.orig_before_len_filter/.
#
# Must be run from the recipe root (uses utils/).

set -euo pipefail

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <data-dir> <max-sec> [<sample-rate>]" >&2
    echo "e.g.: $0 dump/audio_raw/speechlm/train 120" >&2
    exit 1
fi

dir=$1
max_sec=$2
fs=${3:-16000}

for f in wav.scp utt2spk spk2utt utt2num_samples; do
    if [ ! -f "${dir}/${f}" ]; then
        echo "$0: ${dir}/${f} not found" >&2
        exit 1
    fi
done

backup=${dir}/.orig_before_len_filter
if [ ! -d "${backup}" ]; then
    mkdir -p "${backup}"
    cp "${dir}"/{wav.scp,utt2spk,spk2utt,utt2num_samples} "${backup}/"
fi

n_before=$(wc -l < "${dir}/wav.scp")
awk -v max=$((max_sec * fs)) '$2 <= max {print $1}' "${dir}/utt2num_samples" \
    > "${dir}/.keep_utts"

for f in wav.scp utt2spk utt2num_samples; do
    utils/filter_scp.pl "${dir}/.keep_utts" "${dir}/${f}" > "${dir}/${f}.tmp"
    mv "${dir}/${f}.tmp" "${dir}/${f}"
done
utils/utt2spk_to_spk2utt.pl "${dir}/utt2spk" > "${dir}/spk2utt"

utils/validate_data_dir.sh --no-feats --no-text "${dir}"

n_after=$(wc -l < "${dir}/wav.scp")
echo "$0: ${dir}: kept ${n_after}/${n_before} utts (<= ${max_sec}s), dropped $((n_before - n_after))"
