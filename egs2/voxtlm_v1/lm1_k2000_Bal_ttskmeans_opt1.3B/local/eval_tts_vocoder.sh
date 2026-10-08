#!/usr/bin/env bash
# TTS output -> audio -> CER/MOSNet-style objective scores.
#
# Stage 10 only gets us as far as hyp.tok: a discrete-unit sequence the LM
# generated from text, not audio. This script picks up from there:
#   1. hyp.tok -> per-utterance HDF5 "feats" files (local/tok_to_pwg_feats.py)
#   2. parallel-wavegan-decode: feats -> waveform, using the user-provided
#      DiscreteSymbolHiFiGANGenerator checkpoint (hubert_voc_v1Balk2000, trained on
#      LJSpeech, 16kHz, hop=320 matching this recipe's HuBERT-base/50Hz
#      tokens; its downloads/km_2000.mdl is byte-identical (md5
#      20df48056de7e156672a4e060d6ae43f) to the km_2000.mdl this recipe
#      labeled its training data with, so the unit ids mean the same thing
#      on both sides -- that match is what makes this vocoder usable here) -- there is no MOSNet script here; scoring is VERSA (already
#      set up in that same ParallelWaveGAN checkout for reconstruction
#      eval), which reports whisper_wer (-> CER/WER) and pseudo_mos/utmos
#      (a modern MOSNet-family reference-free naturalness predictor) in one
#      pass, plus mcd_f0/discrete_speech/speaker as a bonus.
#   3. VERSA scoring against this recipe's own tts test set (data/test/
#      speech/tts/{wav.scp,text}), not hubert_voc_v1Balk2000's LJSpeech eval set.
#
# Must run with the `parallelwavegan` conda env active (steps 1-2 import
# parallel_wavegan; step 3 imports versa, confirmed installed there too) --
# NOT this recipe's own ./path.sh, which would activate the espnet env
# instead and shadow it. ./cmd.sh IS sourced (for cuda_cmd/train_cmd), since
# those are just shell variables (slurm.pl ...), independent of which conda
# env happens to be active when the submitted job actually runs.
#
# Usage: local/eval_tts_vocoder.sh <checkpoint.pkl> [tts_subdir]
#   tts_subdir defaults to "text.tts" (the full 9957-utt corpus test set, matching
#   Stage 10's default ${lm_test_text_tts} basename). Pass a different value to
#   score a different Stage-10 output side by side without touching the
#   full-corpus one -- e.g. "text.tts.subset100" for the 100-utterance short
#   subset (local/tts_eval_subset/), which lands in its own
#   decode_test_tts/text.tts.subset100/ directory because lm.sh names that
#   directory after ${lm_test_text_tts}'s basename.
#   (run from the recipe root, egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans)

set -e
set -u
set -o pipefail

# Deliberately not sourcing ./path.sh (it would activate the espnet conda
# env via tools/activate_python.sh, shadowing the parallelwavegan env this
# script needs) -- but that also means utils/ (slurm.pl, split_scp.pl) never
# makes it onto PATH, which ./path.sh normally handles with this one line.
export PATH="${PWD}/utils:${PATH}"
. ./cmd.sh

# slurm.pl's --export=ALL passes this shell's current environment as a
# baseline, but the submitted job's own shell startup still runs and can
# reset PATH -- confirmed the hard way: step 2's first attempt failed with
# "parallel-wavegan-decode: command not found" even though this script's own
# `conda activate parallelwavegan` (done by the caller, myrun_vocoder.sh)
# had clearly succeeded here. Re-activating inside every submitted command
# (via this prefix) is what actually makes the env stick in the new job,
# same reasoning ./path.sh/./cmd.sh get re-sourced at every level elsewhere
# in this recipe rather than trusted to survive a new process.
CONDA_REACTIVATE="source /opt/slurm_files/software/conda25/2025-08-21/etc/profile.d/conda.sh && conda activate parallelwavegan && "

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <checkpoint.pkl> [tts_subdir]" >&2
    exit 1
fi
checkpoint=$1
tts_subdir=${2:-text.tts}
[ -f "${checkpoint}" ] || { echo "checkpoint not found: ${checkpoint}" >&2; exit 1; }

PWG_ROOT=/export/fs06/mliang17/phonologicaltokenizer_modify/ParallelWaveGAN
VERSA_CACHE=${PWG_ROOT}/egs/ljspeech/hubert_voc_v1Balk2000/downloads/versa_cache
VERSA_CONFIG=conf/versa.yaml

hyp_tok=exp/lm_train_transformer_opt1.3b_en_bpe10000/decode_test_tts/${tts_subdir}/score_tts/hyp.tok
if [ ! -f "${hyp_tok}" ]; then
    echo "Error: ${hyp_tok} not found -- run Stage 10 first (./run.sh --stage 10 --stop_stage 10 ...)." >&2
    exit 1
fi

work=exp/lm_train_transformer_opt1.3b_en_bpe10000/decode_test_tts/${tts_subdir}/vocoder_eval/$(basename "${checkpoint}" .pkl)
feats_dir=${work}/feats
gen_dir=${work}/wav
eval_dir=${work}/versa_eval
mkdir -p "${feats_dir}" "${gen_dir}" "${eval_dir}"

echo "Step 1: hyp.tok -> HDF5 feats"
python3 local/tok_to_pwg_feats.py --hyp-tok "${hyp_tok}" --dumpdir "${feats_dir}" --nclusters 2000

echo "Step 2: parallel-wavegan-decode (regular gpu partition)"
${cuda_cmd} --gpu 1 "${gen_dir}/decode.log" \
    bash -c "${CONDA_REACTIVATE}parallel-wavegan-decode \
        --dumpdir '${feats_dir}' \
        --checkpoint '${checkpoint}' \
        --outdir '${gen_dir}' \
        --verbose 1"

n_gen=$(find "${gen_dir}" -maxdepth 1 -iname "*.wav" | wc -l)
echo "Generated ${n_gen} wavs at ${gen_dir}"
[ "${n_gen}" -eq 0 ] && { echo "Error: no wavs generated, check ${gen_dir}/decode.log" >&2; exit 1; }

# Same awk as hubert_voc_v1Balk2000/run.sh Stage 4: strip ".wav" and the decode script's
# own "_gen" suffix so keys match data/test/speech/tts/{wav.scp,text}.
find "${gen_dir}" -maxdepth 1 -iname "*.wav" | sort \
    | awk -F/ '{fn=$NF; sub(/\.wav$/, "", fn); sub(/_gen$/, "", fn); print fn, $0}' \
    > "${eval_dir}/pred.scp"

gt_wavscp=data/test/speech/tts/wav.scp
gt_text=data/test/speech/tts/text

echo "Step 3: VERSA scoring (whisper_wer -> CER/WER, pseudo_mos/utmos, mcd_f0, discrete_speech, speaker), on the A100"
# Loading every metric in conf/versa.yaml at once peaks at ~11.5GB reserved
# (hubert_voc_v1Balk2000's own conf/versa.yaml comment / run.sh Stage 4 comment) -- does
# not fit the regular partition's 10.9GB 1080 Ti nodes. Same sed+trap
# pattern used throughout this recipe for Stage 7/sWUGGY: temporarily point
# cuda_cmd at the A100 config, revert on exit no matter what.
trap 'sed -i "s#export cuda_cmd=\"slurm.pl --config conf/slurm_clsp_a100.conf\"#export cuda_cmd=\"slurm.pl\"#" cmd.sh' EXIT
sed -i 's#export cuda_cmd="slurm.pl"#export cuda_cmd="slurm.pl --config conf/slurm_clsp_a100.conf"#' cmd.sh
. ./cmd.sh

# [预热步骤已去掉] 原本这里会先在本进程里给 2 条音频完整打一遍分,目的是把
# 各指标模型的缓存拉好,避免多个并行打分任务同时首次写缓存时 FileExistsError。
# 现在打分只用 1 个任务、不存在并行竞争,这一步就没有意义了;而且控制脚本跑在
# cpu 节点上没有 GPU,实测在 CPU 上给 2 条音频算完全部指标要约 50 分钟,纯属
# 白等(k1000 第一次跑到这里就卡了半小时)。缓存本来就是 hubert_voc_v1Balk2000
# 已经下载好的,打分任务自己读即可。

# Single A100 job, not an 8-way array -- the A100 partition is a shared,
# scarce resource this account is already using concurrently for training
# (train.sh, exp/lm_train...), so scoring should not grab 8 GPUs at once.
versa_nj=1
cp "${eval_dir}/pred.scp" "${eval_dir}/pred.1"

${cuda_cmd} --gpu 1 "${eval_dir}/versa_eval.1.log" \
    bash -c "${CONDA_REACTIVATE}python3 -m versa.bin.scorer \
        --pred '${eval_dir}/pred.1' \
        --gt '${gt_wavscp}' \
        --text '${gt_text}' \
        --score_config '${VERSA_CONFIG}' \
        --cache_folder '${VERSA_CACHE}' \
        --output_file '${eval_dir}/result.1.txt' \
        --io soundfile \
        --scoring_mode utterance \
        --resume \
        --use_gpu"

python3 local/aggregate_versa_eval.py \
    --logdir "${eval_dir}" \
    --scoredir "${eval_dir}" \
    --nj "${versa_nj}"

echo "Done. See ${eval_dir}/avg_result.txt"
