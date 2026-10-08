#!/usr/bin/env bash
#SBATCH --job-name=k2000_slm21
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal/log/myrun_slm21.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal/log/myrun_slm21.%j.err

set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal

. ./path.sh
. ./cmd.sh

##############################################################################
# sWUGGY(lexical) + sBLIMP(syntactic) evaluation, mirroring
# ../../voxtlm_v1/lm1/myrun.sh's steps 17-19 exactly: text track, then
# speech track (local/eval_slm21_speech.sh has been adapted to use this
# recipe's Phonological Tokenizer codebook instead of v1's hubert_base_6_
# 1000clusters -- see that file's own header comment), then final accuracy
# scoring. Independent of Stage 7-10 (only needs the trained LM checkpoint,
# already have it), so this runs as its own job rather than waiting in line
# behind Stage 8-10 in myrun.sh.
##############################################################################
bash local/eval_slm21_text.sh
bash local/eval_slm21_speech.sh
bash local/eval_slm21_score.sh
