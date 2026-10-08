#!/usr/bin/env bash
#SBATCH --job-name=opt1.3b_greedy_short8
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_opt1.3B/log/myrun_tts_greedy_short8.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_opt1.3B/log/myrun_tts_greedy_short8.%j.err

# penalty 在 0.02~0.5 对这8句完全没用,1.0 直接让其中1句冲过头撞 maxlen
# (myrun_tts_penalty_sweep_short8.sh 的结论)。这8句在原来的beam5解码里
# 从没触发过fallback(fallback_log.tsv: level_used=beam5, still_capped=0,
# 因为触发条件只看"撞maxlen",这8句是提前停,不是撞上限)。这里直接测greedy
# (beam_size=1)本身是否能绕开束搜索把这些句子提前截断的问题。
set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_opt1.3B
. ./path.sh
. ./cmd.sh
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1

trap 'sed -i "s#export cuda_cmd=\"slurm.pl --config conf/slurm_clsp_a100.conf\"#export cuda_cmd=\"slurm.pl\"#" cmd.sh' EXIT
sed -i 's#export cuda_cmd="slurm.pl"#export cuda_cmd="slurm.pl --config conf/slurm_clsp_a100.conf"#' cmd.sh
. ./cmd.sh

MAXLEN=1200
LM_EXP=exp/lm_train_transformer_opt1.3b_en_bpe10000
SWEEP=exp/penalty_sweep
SRC_PROMPT="${SWEEP}/text.tts.short8"
[ -s "${SRC_PROMPT}" ] || { echo "缺 ${SRC_PROMPT}" >&2; exit 1; }

SHORT_KEYS="121_121726_000004_000003 1580_141083_000061_000001 1995_1837_000021_000002 237_126133_000002_000003 4294_14317_000018_000001 5442_41169_000022_000000 7127_75947_000065_000000 7902_96592_000029_000002"

echo "==== baseline: beam5, penalty=0 ===="
python3 local/penalty_sweep_report.py "${LM_EXP}/decode_test_tts/text.tts.rand100" ${MAXLEN} ${SHORT_KEYS}

echo
echo "==== greedy (beam_size=1, penalty=0) ===="
prompt="${SWEEP}/text.tts.short8_greedy"
cp "${SRC_PROMPT}" "${prompt}"

./run.sh --stage 10 --stop_stage 10 --ngpu 1 --gpu_inference true --inference_nj 1 \
    --lm_config conf/tuning/train_transformer_opt1.3b.yaml --inference_lm valid.acc.best.pth \
    --lm_inference_tts_config conf/decode_lm_tts_maxlen1200_greedy.yaml \
    --lm_test_text_tts "${prompt}"

python3 local/penalty_sweep_report.py "${LM_EXP}/decode_test_tts/text.tts.short8_greedy" ${MAXLEN} ${SHORT_KEYS}
