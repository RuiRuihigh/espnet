#!/usr/bin/env bash
# Downloads the pretrained Phonological Tokenizer released by Sony
# (https://huggingface.co/Sony/Phonological-Tokenizer, arXiv:2601.19781):
# a WavLM-large checkpoint fine-tuned with differentiable k-means, and the
# fixed 2000-entry codebook (centroids) used to discretize its layer-21
# hidden states.
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

hf_repo_url="https://huggingface.co/Sony/Phonological-Tokenizer/resolve/main"
download_dir="downloads/phonological_tokenizer"

log "$0 $*"
. utils/parse_options.sh

mkdir -p "${download_dir}"

if [ -f "${download_dir}/.complete" ]; then
    log "${download_dir}/.complete already exists. Skip downloading."
    exit 0
fi

for f in ssl.pth centroids.npy; do
    if [ ! -f "${download_dir}/${f}" ]; then
        log "Downloading ${f} to ${download_dir}/${f}"
        wget --continue -O "${download_dir}/${f}.part" "${hf_repo_url}/${f}"
        mv "${download_dir}/${f}.part" "${download_dir}/${f}"
    else
        log "${download_dir}/${f} already exists. Skip downloading."
    fi
done

touch "${download_dir}/.complete"
log "Successfully downloaded Phonological Tokenizer assets to ${download_dir}"
