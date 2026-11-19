#!/usr/bin/env bash
export PAPER_ROOT=@ROOT@
export PAPER_BASE=@BASE@
export PAPER_MAIN=$PAPER_BASE/software/envs/phonological
export PAPER_AUDIO=$PAPER_BASE/software/envs/audio
export PATH="$PAPER_BASE/software/envs/paper-tools/bin:$PAPER_MAIN/bin:$PATH"
export PYTHONPATH="$PAPER_ROOT/espnet${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=$PAPER_BASE/cache/huggingface
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONIOENCODING=UTF-8
export GIT_PAGER=cat

# Resource-only override for the CPU clustering job (also sourced by official path.sh).
# Do not cap a 16-core allocation at one math-library thread.
if [ "${SLURM_JOB_NAME:-}" = paper_kmeans ]; then
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
    export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
fi
