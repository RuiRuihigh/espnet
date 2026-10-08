#!/usr/bin/env bash
#SBATCH --job-name=v1k2000ttskm_rand100
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans/log/myrun_tts_rand100.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans/log/myrun_tts_rand100.%j.err

set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans

. ./path.sh
. ./cmd.sh

# 随机100句(不限长度) + 解码 maxlen 放开到 1200 的 TTS 测算。四个 recipe
# (v1 k1000_Bal / v1 k2000_Bal / v1 k2000_Bal_ttskmeans / v2 k2000_Bal)用同一份
# keys、同一个 maxlen,数字才可比。选法和 maxlen 的依据见
# local/tts_eval_subset/README_random100.md。
#
# 背景:之前全量测试集里 TTS CER 高,几乎都是被 conf/decode_lm_tts.yaml 里
# maxlen=300(BPE 步数上限)强制截断造成的,不是模型的问题;没被截断的句子里
# k1000 和 v2 的 CER 一样(都是1.7%)。这次把 maxlen 提到能盖住最长测试句
# (各 recipe 完整复现真值最多要 788~977 步)看放开之后的真实表现。
#
# 不动任何已有结果:
#   - 提示词文件是 dump/raw/test/text.tts.rand100(新文件),原来的
#     text.tts / text.tts.subset100 不碰;
#   - lm.sh 用输入文件名给输出目录命名,所以产物全部落在
#     exp/.../decode_test_tts/text.tts.rand100/,跟已有的 text.tts/ 和
#     text.tts.subset100/ 是平行目录;
#   - 解码配置是新文件 conf/decode_lm_tts_maxlen1200.yaml(只有 maxlen 不同),
#     原来的 conf/decode_lm_tts.yaml 不改。
# vocoder 是 hubert_voc_v1Balk2000_ttskmeans(tts 10% 子集训的 km_2000 的标签训的)。

##############################################################################
# Stage 10: TTS 解码,只对 100 句。--lm_inference_tts_config 覆盖 run.sh 里
# 默认的 conf/decode_lm_tts.yaml(run.sh 把 "$@" 放最后,后传的同名参数生效)。
# 用 valid.acc.best.pth,跟之前的评测口径一致。
##############################################################################
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1
./run.sh --stage 10 --stop_stage 10 --ngpu 1 --nclusters 2000 \
    --lm_config conf/tuning/train_transformer_opt350.yaml \
    --inference_lm valid.acc.best.pth \
    --lm_inference_tts_config conf/decode_lm_tts_maxlen1200.yaml \
    --lm_test_text_tts dump/raw/test/text.tts.rand100

# 只打印:100 句里有几条仍然撞满上限(理论上应为 0,>0 说明 1200 还不够)
python3 local/count_capped_steps.py \
    exp/lm_train_transformer_opt350_en_bpe10000/decode_test_tts/text.tts.rand100 1200

##############################################################################
# vocoder 合成 + VERSA 打分(A100 只用 1 张,脚本内 trap 切换/还原)。
##############################################################################
VOCODER_CKPT=/export/fs06/mliang17/phonologicaltokenizer_modify/ParallelWaveGAN/egs/ljspeech/hubert_voc_v1Balk2000_ttskmeans/exp/train_nodev_ljspeech_hifigan_hubert.v1/checkpoint-400000steps.pkl
[ -f "${VOCODER_CKPT}" ] || { echo "vocoder checkpoint 不存在: ${VOCODER_CKPT}" >&2; exit 1; }

(
    unset PYTHONPATH
    export PATH="${PWD}/utils:${PATH}"
    # shellcheck disable=SC1091
    source /opt/slurm_files/software/conda25/2025-08-21/etc/profile.d/conda.sh
    conda activate parallelwavegan
    bash local/eval_tts_vocoder.sh "${VOCODER_CKPT}" text.tts.rand100
)

echo "======================================================================"
echo "随机100句 + maxlen=1200 跑完。结果:"
echo "  exp/lm_train_transformer_opt350_en_bpe10000/decode_test_tts/text.tts.rand100/vocoder_eval/*/versa_eval/avg_result.txt"
echo "======================================================================"
