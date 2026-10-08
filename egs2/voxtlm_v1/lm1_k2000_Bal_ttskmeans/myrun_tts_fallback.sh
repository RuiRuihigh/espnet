#!/usr/bin/env bash
#SBATCH --job-name=v1k2000ttskm_fallback
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans/log/myrun_tts_fallback.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans/log/myrun_tts_fallback.%j.err

set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans

. ./path.sh
. ./cmd.sh

# 原始随机 100 句(local/tts_eval_subset/keys_random100.txt,种子 1234,可复现)上的
# 两级回退解码:
#
#   第1级  束搜索 beam5, maxlen 1200      —— 正常句子的准确率最好
#   第2级  贪心(束宽1)                   —— 只重解第1级撞满上限的句子
#   第3级  top-p 0.9 采样(温度 1.0)      —— 只重解第2级仍撞满上限的句子
#
# "撞满" = 生成的 BPE 步数 >= maxlen-1(即被强制截断,不需要参考文本就能判断,见
# local/fallback_select.py 的说明)。之前的实验表明:束搜索会因为开头几步的早期剪枝把
# 正确路径挤出束、掉进重复循环、一直不吐 EOS 而撞满;贪心/采样不做这种淘汰,基本不会
# 跑飞,但对正常句子的准确率不如束搜索,所以只当兜底。
#
# 采样用 SAMPLE_PER_UTT_SEED=1:每句话的随机种子只由 (SAMPLE_SEED, 该句提示词) 决定,
# 不依赖分片和运行顺序,所以重跑结果一致(GPU 浮点非确定性除外,没有专门验证)。
#
# 不动任何已有结果:
#   - 第1级复用 exp/.../decode_test_tts/text.tts.rand100/(同一批句子、同一套 beam5
#     maxlen1200 参数,已核对过 n=0 的重跑逐 token 一致);不存在或不完整才会重跑。
#   - 第2/3级、最终拼接结果分别落在 text.tts.fb_greedy / fb_topp / fb_final,
#     每次运行开头会先清掉这三个目录(它们只由本脚本产生),避免 VERSA --resume 读到旧结果。
# 最终的 100 句 hyp.tok 再统一走 vocoder + VERSA(A100 只用 1 张)。
# vocoder: hubert_voc_v1Balk2000_ttskmeans。rand100 上有 1 句撞满,预期贪心即可解决。

export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1

VOCODER_CKPT=/export/fs06/mliang17/phonologicaltokenizer_modify/ParallelWaveGAN/egs/ljspeech/hubert_voc_v1Balk2000_ttskmeans/exp/train_nodev_ljspeech_hifigan_hubert.v1/checkpoint-400000steps.pkl
[ -f "${VOCODER_CKPT}" ] || { echo "vocoder checkpoint 不存在: ${VOCODER_CKPT}" >&2; exit 1; }

MAXLEN=1200
D=exp/lm_train_transformer_opt350_en_bpe10000/decode_test_tts
F=exp/fallback
mkdir -p "${F}"
COMMON="--stage 10 --stop_stage 10 --ngpu 1 --nclusters 2000 --inference_nj 8 --lm_config conf/tuning/train_transformer_opt350.yaml --inference_lm valid.acc.best.pth"

# ---------------- 第1级:beam5 ----------------
n1=0; [ -s "${D}/text.tts.rand100/score_tts/hyp.tok" ] && n1=$(wc -l < "${D}/text.tts.rand100/score_tts/hyp.tok")
ml1=""; [ -f "${D}/text.tts.rand100/logdir/lm_inference.1.log" ] && ml1=$(grep -m1 -o "max output length: \[[0-9]*\]" "${D}/text.tts.rand100/logdir/lm_inference.1.log" || true)
if [ "${n1}" -eq 100 ] && [ "${ml1}" = "max output length: [${MAXLEN}]" ]; then
    echo "[L1] 复用已有的 beam5 结果 text.tts.rand100(${n1} 句, ${ml1})"
else
    echo "[L1] 没有可用的 beam5 结果(hyp.tok=${n1} 行, '${ml1}'),重新解码 100 句"
    ./run.sh ${COMMON} --python python3 \
        --lm_inference_tts_config conf/decode_lm_tts_maxlen1200.yaml \
        --lm_test_text_tts dump/raw/test/text.tts.rand100
fi
python3 local/count_capped_steps.py "${D}/text.tts.rand100" ${MAXLEN}
python3 local/fallback_select.py capped "${D}/text.tts.rand100" ${MAXLEN} > "${F}/capped_L1.txt"
echo "[L1] 撞满 $(wc -l < "${F}/capped_L1.txt") 句"

rm -rf "${D}/text.tts.fb_greedy" "${D}/text.tts.fb_topp" "${D}/text.tts.fb_final"
LEVELS="${D}/text.tts.rand100"

# ---------------- 第2级:贪心 ----------------
if [ -s "${F}/capped_L1.txt" ]; then
    python3 local/fallback_select.py make_text dump/raw/test/text.tts.rand100 "${F}/capped_L1.txt" dump/raw/test/text.tts.fb_greedy
    unset SAMPLE_MODE SAMPLE_TOP_P SAMPLE_TEMP SAMPLE_SEED SAMPLE_PER_UTT_SEED
    ./run.sh ${COMMON} --python local/python_sampling.sh \
        --lm_inference_tts_config conf/decode_lm_tts_beam1_maxlen1200.yaml \
        --lm_test_text_tts dump/raw/test/text.tts.fb_greedy
    python3 local/fallback_select.py capped "${D}/text.tts.fb_greedy" ${MAXLEN} > "${F}/capped_L2.txt"
    echo "[L2] 贪心后仍撞满 $(wc -l < "${F}/capped_L2.txt") 句"
    LEVELS="${LEVELS} ${D}/text.tts.fb_greedy"

    # ---------------- 第3级:top-p 0.9 ----------------
    if [ -s "${F}/capped_L2.txt" ]; then
        python3 local/fallback_select.py make_text dump/raw/test/text.tts.rand100 "${F}/capped_L2.txt" dump/raw/test/text.tts.fb_topp
        SAMPLE_MODE=topp SAMPLE_TOP_P=0.9 SAMPLE_TEMP=1.0 SAMPLE_SEED=0 SAMPLE_PER_UTT_SEED=1 \
        ./run.sh ${COMMON} --python local/python_sampling.sh \
            --lm_inference_tts_config conf/decode_lm_tts_beam1_maxlen1200.yaml \
            --lm_test_text_tts dump/raw/test/text.tts.fb_topp
        LEVELS="${LEVELS} ${D}/text.tts.fb_topp"
    else
        echo "[L3] 贪心已经解决了所有撞满的句子,不需要 top-p"
    fi
else
    echo "[L2/L3] 第1级没有撞满的句子,不需要回退"
fi

# ---------------- 拼接 + vocoder + VERSA ----------------
python3 local/fallback_select.py assemble "${D}/text.tts.fb_final" ${MAXLEN} ${LEVELS}
(
    unset PYTHONPATH
    export PATH="${PWD}/utils:${PATH}"
    # shellcheck disable=SC1091
    source /opt/slurm_files/software/conda25/2025-08-21/etc/profile.d/conda.sh
    conda activate parallelwavegan
    bash local/eval_tts_vocoder.sh "${VOCODER_CKPT}" text.tts.fb_final
)
python3 local/summarize_fallback.py
echo "======================================================================"
echo "两级回退跑完:${D}/text.tts.fb_final/(fallback_log.tsv 记录每句用了哪一级)"
echo "======================================================================"
