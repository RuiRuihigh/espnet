#!/usr/bin/env bash
#SBATCH --job-name=opt1.3b_topp_short8
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_opt1.3B/log/myrun_tts_topp_short8.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_opt1.3B/log/myrun_tts_topp_short8.%j.err

# greedy(beam_size=1,确定性)在这8句上系统性更差(6/8更短,1句撞maxlen)
# (myrun_tts_greedy_short8.sh 的结论) -- 排除了"换成确定性搜索就能解决"
# 这条路。这里换top-p 0.9采样(温度1.0,Gumbel-max技巧实现,见
# local/lm_inference_sampling.py),看引入随机性本身是否能绕开"模型对这些
# 句子的EOS概率本来就偏高"这个倾向。采样有随机性,跑3个种子取中位数更可靠。
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
SEEDS=${SEEDS:-"0 1 2"}

echo "==== baseline: beam5, penalty=0 ===="
python3 local/penalty_sweep_report.py "${LM_EXP}/decode_test_tts/text.tts.rand100" ${MAXLEN} ${SHORT_KEYS}

for s in ${SEEDS}; do
    echo
    echo "==== top-p 0.9, seed=${s} ===="
    export SAMPLE_MODE=topp SAMPLE_TOP_P=0.9 SAMPLE_TEMP=1.0 SAMPLE_SEED="${s}"
    prompt="${SWEEP}/text.tts.short8_topp_s${s}"
    cp "${SRC_PROMPT}" "${prompt}"

    ./run.sh --stage 10 --stop_stage 10 --ngpu 1 --gpu_inference true --inference_nj 1 \
        --python local/python_sampling.sh \
        --lm_config conf/tuning/train_transformer_opt1.3b.yaml --inference_lm valid.acc.best.pth \
        --lm_inference_tts_config conf/decode_lm_tts_beam1_maxlen1200.yaml \
        --lm_test_text_tts "${prompt}"

    python3 local/penalty_sweep_report.py "${LM_EXP}/decode_test_tts/text.tts.short8_topp_s${s}" ${MAXLEN} ${SHORT_KEYS}
done
unset SAMPLE_MODE SAMPLE_TOP_P SAMPLE_TEMP SAMPLE_SEED

echo
echo "==== 全部跑完,对照beam5/greedy/top-p(3种子)下这8句的步数变化 ===="
