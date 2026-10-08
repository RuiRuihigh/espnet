#!/usr/bin/env bash
# Sharded Part II log-likelihood scoring (local/score_mc_loglik.py) on the gpu partition.
# Usage: local/run_mc_loglik.sh --lm_exp exp/sqa_v1_ft --ckpt 10epoch.pth \
#            --data dump/v1/raw/dev_p2 --out_dir exp/sqa_v1_ft/mc_loglik/dev_p2/10epoch [--variants A] [--nj 8]
# Output: ${out_dir}/result.txt (+ logdir/scores.*.jsonl). Skips if result.txt already exists.
set -e
set -u
set -o pipefail

lm_exp=
ckpt=
data=
out_dir=
variants=A,B,L
with_rand=true
nj=8

. utils/parse_options.sh
. ./path.sh
. ./cmd.sh

if [ -s "${out_dir}/result.txt" ]; then
    echo "${out_dir}/result.txt exists, skip"
    exit 0
fi
mkdir -p "${out_dir}/logdir"
split_scps=""
for n in $(seq ${nj}); do split_scps+=" ${out_dir}/logdir/keys.${n}.scp"; done
# shellcheck disable=SC2086
utils/split_scp.pl "${data}/text.sqa" ${split_scps}

${cuda_cmd} --gpu 1 JOB=1:${nj} "${out_dir}"/logdir/score.JOB.log \
    python local/score_mc_loglik.py --lm_train_config "${lm_exp}"/config.yaml \
        --lm_file "${lm_exp}/${ckpt}" --test_dir "${data}" --keys "${out_dir}"/logdir/keys.JOB.scp \
        --variants "${variants}" --with_rand "${with_rand}" \
        --out "${out_dir}"/logdir/scores.JOB.jsonl

if grep -h boundary_mismatch "${out_dir}"/logdir/score.*.log | grep -vq "boundary_mismatch 0"; then
    echo "WARNING: tokenization boundary mismatch in ${out_dir}" >&2
fi
python local/score_mc_loglik.py summarize "${out_dir}/result.txt" "${out_dir}"/logdir/scores.*.jsonl
