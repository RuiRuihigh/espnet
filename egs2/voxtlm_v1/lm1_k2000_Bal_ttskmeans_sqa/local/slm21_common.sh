#!/usr/bin/env bash
# Shared setup for local/eval_slm21_text.sh and local/eval_slm21_speech.sh.
# Meant to be `source`d, not run directly.

LM_CONFIG=exp/sqa_v1_ft/config.yaml
# 用 valid.loss.best.pth 指向的 13epoch(SQA finetune 选定的 checkpoint,
# 跟 myrun_sqa_test.sh ft 模式默认用的是同一个),不是 valid.acc.best(这份
# 训练只保留 valid loss 最好的 5 个 checkpoint,没有 acc.best)。
LM_FILE=exp/sqa_v1_ft/13epoch.pth
SLM21_WORK=exp/slm21_eval
SLM21_SUBMISSION=${SLM21_WORK}/submission

# cmd.sh's cuda_cmd defaults to the A100 config (conf/slurm_clsp_a100.conf),
# which Stage 7 training genuinely needs (80GB). Everything in this file is
# light GPU inference instead -- lm_calc_perplexity.py forward passes
# (batch_size=1, fp32, same as Stage 8) and hubert feature extraction on
# short clips (perform_kmeans.sh) -- so it doesn't need an A100, and A100
# account quota may not have room for it anyway. Same trick used for Stage
# 8-10 earlier: temporarily point cuda_cmd at the regular gpu partition
# instead, with a trap so it's switched back to the A100 config on exit
# (success, failure, or interrupt) no matter what -- so a later Stage 7
# resume never silently runs on too little GPU memory because this was left
# pointed at the wrong partition.
use_regular_gpu_partition() {
    # 老 checkout 的 cmd.sh 默认 cuda_cmd 指向 a100,所以那边这个函数要临时 sed
    # 成普通分区再改回去。本 recipe 的 cmd.sh 默认就是普通 gpu(只有 myrun.sh 的
    # Stage 7 训练那一段才临时切到 a100,且用 trap 改回),这里什么都不用做,
    # 保留同名函数只是为了跟老脚本保持一致、少改调用处。
    # shellcheck disable=SC1091
    . ./cmd.sh
}

# Extracts files matching glob patterns (e.g. "lexical/dev/*.wav") from
# sLM21.dataset.zip, without depending on a system `unzip` binary -- some
# compute nodes on this cluster don't have unzip in PATH (job 1798491 hit
# this: "unzip: command not found"), while python3's stdlib zipfile module
# is always available since this whole pipeline already requires python3.
# Mirrors `unzip -n`'s behavior of skipping files that already exist, so
# re-running after a partial extraction is still cheap and safe.
extract_slm21() {
    local out_dir=$1
    shift
    python3 - "${out_dir}" "$@" <<'PYEOF'
import fnmatch
import os
import sys
import zipfile

out_dir, patterns = sys.argv[1], sys.argv[2:]
with zipfile.ZipFile("downloads/sLM21/sLM21.dataset.zip") as zf:
    names = zf.namelist()
    matched = set()
    for pattern in patterns:
        matched.update(fnmatch.filter(names, pattern))
    for name in sorted(matched):
        target = os.path.join(out_dir, name)
        if os.path.exists(target):
            continue
        zf.extract(name, out_dir)
PYEOF
}

# Shards --data_path_and_name_and_type (via --key_file, same mechanism
# lm.sh's own Stage 9/10 decoding uses) into $3-way parallel GPU jobs
# instead of running lm_calc_perplexity.py's inherently-serial per-utterance
# loop as one single process.
run_lm_calc_perplexity_parallel() {
    local text_file=$1 out_dir=$2 nj=$3
    local logdir=${out_dir}/logdir
    mkdir -p "${logdir}"

    local nutt
    nutt=$(<"${text_file}" wc -l)
    [ "${nj}" -gt "${nutt}" ] && nj=${nutt}

    awk '{print $1}' "${text_file}" > "${logdir}/keys.scp"
    local split_scps=""
    for n in $(seq "${nj}"); do
        split_scps+=" ${logdir}/keys.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${logdir}/keys.scp" ${split_scps}

    ${cuda_cmd} --gpu 1 JOB=1:"${nj}" "${logdir}/lm_calc_perplexity.JOB.log" \
        python3 -m espnet2.bin.lm_calc_perplexity \
            --ngpu 1 \
            --data_path_and_name_and_type "${text_file},text,text" \
            --key_file "${logdir}/keys.JOB.scp" \
            --train_config "${LM_CONFIG}" \
            --model_file "${LM_FILE}" \
            --output_dir "${out_dir}/ppl_out.JOB"

    mkdir -p "${out_dir}/ppl_out"
    for f in utt2ppl utt2ntokens; do
        for n in $(seq "${nj}"); do
            cat "${out_dir}/ppl_out.${n}/${f}"
        done > "${out_dir}/ppl_out/${f}"
    done
}
