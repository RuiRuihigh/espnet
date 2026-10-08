#!/usr/bin/env bash
#SBATCH --job-name=opt1.3b_penalty_sweep
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_opt1.3B/log/myrun_tts_penalty_sweep.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_opt1.3B/log/myrun_tts_penalty_sweep.%j.err

# 用法: sbatch myrun_tts_penalty_sweep.sh
#       PENALTIES="0.05 0.1" sbatch myrun_tts_penalty_sweep.sh   # 只扫指定的值
#
# 只做 LM 解码(Stage 10),不跑 vocoder/VERSA -- 便宜、快,用来先看 penalty 对
# "撞maxlen" vs "提前停漏字" 这两种相反失败模式的影响,再决定要不要对某个
# penalty 值跑完整的 vocoder+VERSA 确认实际 CER/WER 有没有改善。
#
# 背景(见跟用户的分析讨论): penalty=0 时 1.3B 在 rand100(100句固定子集)上
# 有 8 句明显提前停、生成长度不够导致漏字,beam5 的回退机制(只检测撞maxlen,
# 不检测提前停)完全没救到这 8 句。这里扫几个小的正 penalty 值,观察:
#   1) 这 8 句的生成步数有没有明显变长(修复提前停的信号)
#   2) 全部100句里撞maxlen的句子数有没有从当前的1句明显增多(补偿过头的信号,
#      说明把"漏字"问题换成了"陷入重复循环出不来"的问题)
# 每个 penalty 值的 100 句 prompt 用同一份内容、不同文件名(避免互相覆盖输出目录,
# lm.sh 按 --lm_test_text_tts 的 basename 给 Stage10 输出目录命名)。
set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_opt1.3B
. ./path.sh
. ./cmd.sh
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1

MAXLEN=1200
LM_EXP=exp/lm_train_transformer_opt1.3b_en_bpe10000
SWEEP=exp/penalty_sweep
mkdir -p "${SWEEP}"
PENALTIES=${PENALTIES:-"0.05 0.1 0.2"}

# 之前识别出的8句明显提前停/漏字的句子(不带 tts_ 前缀,跟 fallback_select.py/
# penalty_sweep_report.py 的约定一致)
SHORT_KEYS="121_121726_000004_000003 1580_141083_000061_000001 1995_1837_000021_000002 237_126133_000002_000003 4294_14317_000018_000001 5442_41169_000022_000000 7127_75947_000065_000000 7902_96592_000029_000002"

echo "==== baseline: penalty=0(已有的 text.tts.rand100 解码,不重跑) ===="
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
    prompt="${SWEEP}/text.tts.rand100_p${p}"
    cp dump/raw/test/text.tts.rand100 "${prompt}"

    ./run.sh --stage 10 --stop_stage 10 --ngpu 1 --gpu_inference true --inference_nj 8 \
        --lm_config conf/tuning/train_transformer_opt1.3b.yaml --inference_lm valid.acc.best.pth \
        --lm_inference_tts_config "${cfg}" \
        --lm_test_text_tts "${prompt}"

    python3 local/penalty_sweep_report.py "${LM_EXP}/decode_test_tts/text.tts.rand100_p${p}" ${MAXLEN} ${SHORT_KEYS}
done

echo
echo "==== 全部跑完,对照上面每个 penalty 的 median/max/capped 和8句的生成步数变化 ===="
echo "(选出一个"8句明显变长但capped数没有明显增多"的 penalty 后,"
echo " 再单独对那个值跑 local/eval_tts_vocoder.sh 确认实际 CER/WER 改善)"
