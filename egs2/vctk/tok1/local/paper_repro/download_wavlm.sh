#!/usr/bin/env bash
set -euo pipefail
test "$#" -eq 1 || { echo "Usage: $0 /absolute/experiment/root" >&2; exit 2; }
source "$1/env.sh"
mkdir -p "$PAPER_BASE/assets"
cd "$PAPER_BASE/assets"
expected=6fb4b3c3e6aa567f0a997b30855859cb81528ee8078802af439f7b2da0bf100f
if [ ! -f wavlm_large.pt ]; then
    curl --fail --location --retry 3 --connect-timeout 30 --continue-at - \
        --output wavlm_large.pt.partial \
        https://huggingface.co/s3prl/converted_ckpts/resolve/7bc66aee7c4a8c287923bf853afb702a5f8bf08e/wavlm_large.pt
    printf '%s  %s\n' "$expected" wavlm_large.pt.partial | sha256sum -c -
    mv wavlm_large.pt.partial wavlm_large.pt
fi
printf '%s  %s\n' "$expected" wavlm_large.pt | sha256sum -c -
