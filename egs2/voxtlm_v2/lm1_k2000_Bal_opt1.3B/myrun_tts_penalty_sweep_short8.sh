#!/usr/bin/env bash
#SBATCH --job-name=opt1.3b_penalty_short8
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_opt1.3B/log/myrun_tts_penalty_sweep_short8.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_opt1.3B/log/myrun_tts_penalty_sweep_short8.%j.err

# 用法: sbatch myrun_tts_penalty_sweep_short8.sh
#       PENALTIES="0.1 0.2" sbatch myrun_tts_penalty_sweep_short8.sh
#
# 取代 myrun_tts_penalty_sweep.sh(已经手动停掉,那个是在全部100句上扫,
# 一个penalty就要跑一次8分片x100句的完整beam5解码,太慢)。这里只解码之前
# 识别出的8句明显提前停/漏字的句子,8句用1个分片足够,每个penalty几秒到
# 几十秒就能跑完,可以扫更多个值、更快迭代。
#
# exp/penalty_sweep/text.tts.short8(8句,从 dump/raw/test/text.tts.rand100
# 按key过滤出来,本地文件,不碰软链接的原始数据)已经建好。
set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_opt1.3B
. ./path.sh
. ./cmd.sh
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1

# 普通 gpu 分区这几天一直被其它任务占满,这8句的小解码一直在排队。A100现在
# 空着,临时切过去(trap 保证不管成功/失败/中断都会切回来)。
trap 'sed -i "s#export cuda_cmd=\"slurm.pl --config conf/slurm_clsp_a100.conf\"#export cuda_cmd=\"slurm.pl\"#" cmd.sh' EXIT
sed -i 's#export cuda_cmd="slurm.pl"#export cuda_cmd="slurm.pl --config conf/slurm_clsp_a100.conf"#' cmd.sh
. ./cmd.sh

MAXLEN=1200
LM_EXP=exp/lm_train_transformer_opt1.3b_en_bpe10000
SWEEP=exp/penalty_sweep
SRC_PROMPT="${SWEEP}/text.tts.short8"
[ -s "${SRC_PROMPT}" ] || { echo "缺 ${SRC_PROMPT}" >&2; exit 1; }
PENALTIES=${PENALTIES:-"0.02 0.05 0.1 0.15 0.2 0.3 0.5 1.0"}

SHORT_KEYS="121_121726_000004_000003 1580_141083_000061_000001 1995_1837_000021_000002 237_126133_000002_000003 4294_14317_000018_000001 5442_41169_000022_000000 7127_75947_000065_000000 7902_96592_000029_000002"

echo "==== baseline: penalty=0(已有的 text.tts.rand100 解码,这8句的子集) ===="
python3 local/penalty_sweep_report.py "${LM_EXP}/decode_test_tts/text.tts.rand100" ${MAXLEN} ${SHORT_KEYS}

for p in ${PENALTIES}; do
    echo
    echo "==== penalty=${p} ===="
    cfg="conf/decode_lm_tts_maxlen1200_p${p}.yaml"
    cat > "${cfg}" <<EOF
log_level: INFO
beam_size: 5
penalty: ${p}
maxlen: ${MAXLEN}
minlen: 1
EOF
    prompt="${SWEEP}/text.tts.short8_p${p}"
    cp "${SRC_PROMPT}" "${prompt}"

    ./run.sh --stage 10 --stop_stage 10 --ngpu 1 --gpu_inference true --inference_nj 1 \
        --lm_config conf/tuning/train_transformer_opt1.3b.yaml --inference_lm valid.acc.best.pth \
        --lm_inference_tts_config "${cfg}" \
        --lm_test_text_tts "${prompt}"

    python3 local/penalty_sweep_report.py "${LM_EXP}/decode_test_tts/text.tts.short8_p${p}" ${MAXLEN} ${SHORT_KEYS}
done

echo
echo "==== 全部跑完,对照每个 penalty 下这8句的步数变化、以及有没有新撞maxlen的 ===="
