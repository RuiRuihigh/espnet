#!/usr/bin/env bash

# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db_root=$1

# check arguments
if [ $# != 1 ]; then
    echo "Usage: $0 <db_root>"
    exit 1
fi

set -euo pipefail

cwd=$(pwd)
if [ ! -e "${db_root}/VCTK-Corpus" ]; then
    # NOTE: the original www.udialogue.org mirror (legacy wav48/ layout) is
    # unreachable, so this pulls the official Edinburgh DataShare 0.92
    # release instead. Its layout (wav48_silence_trimmed/, flac, mic1/mic2)
    # differs from the legacy one; local/vctk/data_prep.sh below is written
    # against this 0.92 layout.
    mkdir -p "${db_root}"
    cd "${db_root}" || exit 1;
    wget -O VCTK-Corpus-0.92.zip \
        https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip
    rm -rf VCTK-Corpus-0.92_extracted
    unzip -q VCTK-Corpus-0.92.zip -d VCTK-Corpus-0.92_extracted
    rm ./VCTK-Corpus-0.92.zip
    # The archive may or may not wrap its contents in a top-level folder;
    # locate whichever directory actually holds wav48_silence_trimmed and
    # rename that to VCTK-Corpus so the rest of the pipeline doesn't care
    # which mirror the data came from.
    src_dir=$(dirname "$(find VCTK-Corpus-0.92_extracted -type d -name wav48_silence_trimmed)")
    mv "${src_dir}" VCTK-Corpus
    rm -rf VCTK-Corpus-0.92_extracted
    cd "${cwd}" || exit 1;
    echo "Successfully downloaded data."
else
    echo "Already exists. Skipped."
fi

if [ ! -e "${db_root}/VCTK-Corpus/lab" ]; then
    cd "${db_root}" || exit 1;
    git clone https://github.com/kan-bayashi/VCTKCorpusFullContextLabel.git
    cp -r VCTKCorpusFullContextLabel/lab ./VCTK-Corpus
    cd "${cwd}" || exit 1;
    echo "Successfully downloaded label data."
else
    echo "Already exists. Skipped."
fi
