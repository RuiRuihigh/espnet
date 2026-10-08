#!/usr/bin/env bash
# Final sWUGGY/sBLIMP accuracy numbers, computed from whichever submission
# files local/eval_slm21_text.sh / local/eval_slm21_speech.sh have produced
# so far (text: dev.txt, speech: dev_speech.txt). Safe to run at any point --
# it just skips a track/task combo whose submission file doesn't exist yet,
# so running it after only the text track has finished still reports the
# text-track numbers.
#
# Usage: local/eval_slm21_score.sh (run from the recipe root, egs2/voxtlm_v1/lm1)

set -e
set -u
set -o pipefail

# shellcheck disable=SC1091
. local/slm21_common.sh

for task in lexical syntactic; do
    gold_csv=downloads/sLM21/extracted/${task}/dev/gold.csv
    if [ ! -f "${gold_csv}" ]; then
        echo "skipping ${task}: ${gold_csv} not found (run local/eval_slm21_text.sh or local/eval_slm21_speech.sh first)"
        continue
    fi
    for track_label in "text:dev.txt" "speech:dev_speech.txt"; do
        track=${track_label%%:*}
        submission=${SLM21_SUBMISSION}/${task}/${track_label#*:}
        if [ ! -f "${submission}" ]; then
            echo "skipping ${task} (${track}): ${submission} not found yet"
            continue
        fi
        python3 local/slm21_pair_accuracy.py \
            --gold-csv "${gold_csv}" \
            --submission "${submission}" \
            --task "${task} (${track})"
    done
done
