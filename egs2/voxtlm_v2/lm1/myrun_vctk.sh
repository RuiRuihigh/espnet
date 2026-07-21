#!/usr/bin/env bash
#SBATCH --job-name=vctk_download
#SBATCH --partition=RM-shared
#SBATCH --account=cis210027p
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --time=3-00:00:00
#SBATCH --output=/ocean/projects/cis210027p/mliang4/dailytalk_tts/espnet/egs2/voxtlm_v2/lm1/log/myrunvctk.%j.out
#SBATCH --error=/ocean/projects/cis210027p/mliang4/dailytalk_tts/espnet/egs2/voxtlm_v2/lm1/log/myrunvctk.%j.err

# Resumes/continues the LibriLight download + VAD-cut + data-prep
# (local/data_librilight.sh is idempotent per-part: small is already done,
# medium's tar is fully downloaded but not yet extracted, large hasn't
# started -- this will pick up from wherever it left off).
set -e
set -u
set -o pipefail

cd /ocean/projects/cis210027p/mliang4/dailytalk_tts/espnet/egs2/voxtlm_v2/lm1

. ./path.sh
. ./cmd.sh

./local/data_vctk.sh data/vctk

