#!/usr/bin/env bash
#SBATCH --job-name=ttskm_trace_beam
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans/log/myrun_trace_beam.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans/log/myrun_trace_beam.%j.err
set -e; set -u; set -o pipefail
cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans
. ./path.sh; . ./cmd.sh
export HF_HOME="${PWD}/hf_cache"; export HF_HUB_OFFLINE=1
# 束搜索(beam5,maxlen1200,和 rand100 那次完全一样)逐步追踪:贪心能完整说对、束搜索却跑飞的句子。
D=exp/lm_train_transformer_opt350_en_bpe10000/decode_test_tts
mkdir -p exp/beam_trace
${cuda_cmd} --gpu 1 exp/beam_trace/trace.log \
  python3 local/trace_beam.py \
    --keys 7902_96592_000035_000000 \
    --text dump/raw/test/text.tts.rpt_n0 --greedy_dir $D/text.tts.samp_greedy --beam_dir $D/text.tts.rpt_n0 \
    --out exp/beam_trace --maxlen 1200
