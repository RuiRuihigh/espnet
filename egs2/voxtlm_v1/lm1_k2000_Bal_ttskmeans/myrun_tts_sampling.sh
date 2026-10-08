#!/usr/bin/env bash
#SBATCH --job-name=ttskm_sampling
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans/log/myrun_tts_sampling.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans/log/myrun_tts_sampling.%j.err

set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans

. ./path.sh
. ./cmd.sh

# 实验:TTS 解码里有些句子"跑飞"(掉进重复循环、不吐 EOS、撞满 maxlen=1200)。
# 禁止重复 n-gram 那轮实验证明"拆掉循环后内容能恢复",但方法本身有副作用。
# 这里换一个思路:用不同的解码方式,看跑飞是不是束搜索这种确定性搜索造成的。
#   greedy      束宽 1 = 贪心(每步取最大概率)。同样是确定性的,用来区分
#               "是束搜索的问题"还是"模型的分布本身就会循环"。
#   top-p 0.9   温度 1.0,每步只在累计概率达到 0.9 的最小 token 集合里随机抽
#               (核采样)。3 个随机种子,因为采样有随机性,单次结果不可靠。
# 采样是通过 local/lm_inference_sampling.py 实现的(不改共享 espnet2 源码),用
# Gumbel-max 技巧让束宽 1 的束搜索等价于从截断分布里抽样,所以除解码方式外,
# 其余(maxlen=1200、文件格式、后面的 vocoder/VERSA)跟之前完全一致。
# 句子:跟 norepeat 那轮同一批(跑飞句 + 20 条最长的正常句作对照)。
#
# 输出各落在 exp/.../decode_test_tts/text.tts.samp_<配置>/,不碰已有结果。
# 每组解码用 --inference_nj 8;VERSA 只用 1 张 A100,四组顺序跑。

export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1

VOCODER_CKPT=/export/fs06/mliang17/phonologicaltokenizer_modify/ParallelWaveGAN/egs/ljspeech/hubert_voc_v1Balk2000_ttskmeans/exp/train_nodev_ljspeech_hifigan_hubert.v1/checkpoint-400000steps.pkl
[ -f "${VOCODER_CKPT}" ] || { echo "vocoder checkpoint 不存在: ${VOCODER_CKPT}" >&2; exit 1; }

# 配置名 : SAMPLE_MODE : SAMPLE_SEED
for cfg in greedy:: topp09_s0:topp:0 topp09_s1:topp:1 topp09_s2:topp:2; do
    IFS=: read -r name mode seed <<< "${cfg}"
    echo "################## ${name}  (SAMPLE_MODE='${mode}' SAMPLE_SEED='${seed}') ##################"
    export SAMPLE_MODE="${mode}" SAMPLE_TOP_P=0.9 SAMPLE_TEMP=1.0 SAMPLE_SEED="${seed:-0}"
    ./run.sh --stage 10 --stop_stage 10 --ngpu 1 --nclusters 2000 \
        --inference_nj 8 \
        --python local/python_sampling.sh \
        --lm_config conf/tuning/train_transformer_opt350.yaml \
        --inference_lm valid.acc.best.pth \
        --lm_inference_tts_config conf/decode_lm_tts_beam1_maxlen1200.yaml \
        --lm_test_text_tts dump/raw/test/text.tts.samp_${name}

    python3 local/count_capped_steps.py \
        exp/lm_train_transformer_opt350_en_bpe10000/decode_test_tts/text.tts.samp_${name} 1200

    (
        unset PYTHONPATH
        export PATH="${PWD}/utils:${PATH}"
        # shellcheck disable=SC1091
        source /opt/slurm_files/software/conda25/2025-08-21/etc/profile.d/conda.sh
        conda activate parallelwavegan
        bash local/eval_tts_vocoder.sh "${VOCODER_CKPT}" text.tts.samp_${name}
    )
done
unset SAMPLE_MODE SAMPLE_TOP_P SAMPLE_TEMP SAMPLE_SEED

echo "======================================================================"
echo "贪心 / top-p 采样实验跑完。用 local/summarize_sampling.py 汇总对比。"
echo "======================================================================"
