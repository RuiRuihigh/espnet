#!/usr/bin/env bash
# sWUGGY(lexical)+ sBLIMP(syntactic) text-track evaluation.
#
# Only dev, not test -- per the paper (arXiv:2309.07937) section 3.2: "we
# use dev sets from sWUGGY and sBLIMP". test is 4x+ bigger than dev and the
# paper doesn't report on it, so this skips it entirely (both the extraction
# and the scoring).
#
# Data comes from downloads/sLM21/sLM21.dataset.zip (already downloaded,
# MD5-verified elsewhere). This only extracts gold.csv (a few tens of MB of
# text metadata) -- not the audio, which the text track doesn't need.
#
# Scoring reuses espnet2.bin.lm_calc_perplexity (the same tool Stage 8
# uses) rather than any new model-loading code. sLM21's submission format
# wants each item's raw total log-likelihood, not perplexity; recovered via
# ppl=exp(nll/ntokens) => nll=ntokens*ln(ppl), score=-nll -- see
# local/slm21_score_from_ppl.py's own docstring for the full derivation.
# Verified end to end on 5 real items: "abandoned" (a real word) scored
# PPL=29.5, far below its non-word pair "abando"'s 332.9 -- the expected
# direction.
#
# gold.csv has many rows that are the same word/sentence recorded by a
# different voice; text scoring doesn't care about voice, so
# local/prepare_slm21_text.py scores each unique text only once (20000/12566
# unique texts, not the raw 80000/50400 rows) and local/slm21_score_from_ppl.py
# expands that one score back out to every filename that shares it.
#
# Usage: local/eval_slm21_text.sh (run from the recipe root, egs2/voxtlm_v1/lm1)

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
extract_slm21 downloads/sLM21/extracted "lexical/dev/gold.csv" "syntactic/dev/gold.csv"

for task in lexical syntactic; do
    prep_dir=${SLM21_WORK}/${task}_dev

    python3 local/prepare_slm21_text.py \
        --gold-csv "downloads/sLM21/extracted/${task}/dev/gold.csv" \
        --task "${task}" \
        --out-dir "${prep_dir}"

    run_lm_calc_perplexity_parallel "${prep_dir}/text_dedup.txt" "${prep_dir}" 8

    python3 local/slm21_score_from_ppl.py \
        --ppl-dir "${prep_dir}/ppl_out" \
        --filename-map "${prep_dir}/filename_map.tsv" \
        --out "${SLM21_SUBMISSION}/${task}/dev.txt"
done

echo "sLM21 text-track submission ready at ${SLM21_SUBMISSION}/{lexical,syntactic}/dev.txt"
