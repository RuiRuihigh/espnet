#!/usr/bin/env bash
# Decode LibriSQA test prompts with espnet2.bin.lm_inference and score them.
# Same decoding path as lm.sh Stage 9 (ASR), just a different prompt file.
#
# Usage: local/eval_sqa.sh --part 2 --lm_exp exp/sqa_part2_ft --data dump/part2/raw/test \
#            --decode_config conf/decode_sqa_p2_letter.yaml [--inference_lm valid.acc.best.pth] [--nj 8]
# Output: ${lm_exp}/decode_sqa/<config name>/{text,score_sqa/result.txt}
set -e
set -u
set -o pipefail

part=2
lm_exp=
data=
decode_config=
inference_lm=valid.acc.best.pth
bpemodel=data/en_token_list/bpe_unigram10000/bpe.model
nj=8
tag=            # optional suffix for the output dir (e.g. diagnostic prompt sets)
python=python3

. utils/parse_options.sh
. ./path.sh
. ./cmd.sh

_dir="${lm_exp}/decode_sqa/$(basename ${inference_lm} .pth)_$(basename ${decode_config} .yaml)${tag:+_${tag}}"
_logdir="${_dir}/logdir"
mkdir -p "${_logdir}"

key_file="${data}/text.sqa"
_nj=$(( nj < $(<${key_file} wc -l) ? nj : $(<${key_file} wc -l) ))
split_scps=""
for n in $(seq "${_nj}"); do
    split_scps+=" ${_logdir}/keys.${n}.scp"
done
# shellcheck disable=SC2086
utils/split_scp.pl "${key_file}" ${split_scps}

echo "$(date) decoding ${key_file} -> ${_dir} (log: ${_logdir}/lm_inference.*.log)"
# shellcheck disable=SC2086
${cuda_cmd} --gpu 1 JOB=1:"${_nj}" "${_logdir}"/lm_inference.JOB.log \
    ${python} -m espnet2.bin.lm_inference \
        --batch_size 1 \
        --ngpu 1 \
        --data_path_and_name_and_type "${key_file},text,text" \
        --key_file "${_logdir}"/keys.JOB.scp \
        --output_dir "${_logdir}"/output.JOB \
        --token_type bpe \
        --bpemodel "${bpemodel}" \
        --lm_train_config "${lm_exp}"/config.yaml \
        --lm_file "${lm_exp}/${inference_lm}" \
        --config "${decode_config}" || { cat $(grep -l -i error "${_logdir}"/lm_inference.*.log) ; exit 1; }

for f in token token_int score text; do
    for i in $(seq "${_nj}"); do
        cat "${_logdir}/output.${i}/1best_recog/${f}"
    done | sort -k1 > "${_dir}/${f}"
done

mkdir -p "${_dir}/score_sqa"
${python} local/score_librisqa.py --part "${part}" --hyp "${_dir}/text" \
    --ref_dir "${data}" --out "${_dir}/score_sqa/result.txt"
