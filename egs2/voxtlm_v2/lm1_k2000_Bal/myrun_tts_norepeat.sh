#!/usr/bin/env bash
#SBATCH --job-name=v2_norepeat
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal/log/myrun_tts_norepeat.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal/log/myrun_tts_norepeat.%j.err

set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal

. ./path.sh
. ./cmd.sh

# 实验:TTS 解码里有些句子"跑飞"——束搜索掉进 "A B A B ..." 这样的两 token 循环,
# 一直不吐 EOS,直到撞满 maxlen=1200(v2 k2000 有 6/100,ttskmeans 有 1/100)。
# 这里验证"确定性束搜索被循环锁死"这个解释:在束搜索里加一个"禁止重复 n-gram"
# 的打分项(local/lm_inference_norepeat.py,不改共享的 espnet2 源码,通过 lm.sh 的
# --python 变量 + local/python_norepeat.sh 接进去),n 从环境变量 NO_REPEAT_NGRAM 读。
#
# 三组,同一批句子(跑飞句 + 20 条最长的正常句作对照,看惩罚会不会伤正常句):
#   n=0  基线:不加禁令,等价于原来的解码。用来核对这个包装脚本没有改变行为
#        (跑飞句应该原样跑飞、对照句的结果应该跟之前的 rand100 一致)。
#   n=8  真实语音 BPE 序列里(9957 条测试句,两种 tokenizer)一句都没有重复的
#        8-gram,所以这个禁令碰不到任何像真实语音的东西,而 2-token 循环
#        第 9 步左右就会违反它。是主设置。
#   n=6  更激进一点(真值里约 1.3% 的 v2 句子含重复 6-gram),看结果是否稳健。
#
# 输出各落在 exp/.../decode_test_tts/text.tts.rpt_n<N>/,不碰任何已有结果。
# 解码 maxlen=1200(conf/decode_lm_tts_maxlen1200.yaml),跟 rand100 那次完全一致。
# 每组解码用 --inference_nj 4(句子只有 20 多条,不用 8 路);VERSA 只用 1 张 A100,
# 三组顺序跑,所以任意时刻本 recipe 最多占 1 张 A100。

export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1

VOCODER_CKPT=/export/fs06/mliang17/phonologicaltokenizer_modify/ParallelWaveGAN/egs/ljspeech/phon_voc1/exp/train_nodev_ljspeech_hifigan_phon.v1/checkpoint-400000steps.pkl
[ -f "${VOCODER_CKPT}" ] || { echo "vocoder checkpoint 不存在: ${VOCODER_CKPT}" >&2; exit 1; }

for n in 0 6 8; do
    echo "################## NO_REPEAT_NGRAM=${n} ##################"
    export NO_REPEAT_NGRAM=${n}
    ./run.sh --stage 10 --stop_stage 10 --ngpu 1 \
        --inference_nj 4 \
        --python local/python_norepeat.sh \
        --lm_config conf/tuning/train_transformer_opt350.yaml \
        --inference_lm valid.acc.best.pth \
        --lm_inference_tts_config conf/decode_lm_tts_maxlen1200.yaml \
        --lm_test_text_tts dump/raw/test/text.tts.rpt_n${n}

    # 只打印:有几条仍然撞满 1200 步
    python3 local/count_capped_steps.py \
        exp/lm_train_transformer_opt350_en_bpe10000/decode_test_tts/text.tts.rpt_n${n} 1200

    (
        unset PYTHONPATH
        export PATH="${PWD}/utils:${PATH}"
        # shellcheck disable=SC1091
        source /opt/slurm_files/software/conda25/2025-08-21/etc/profile.d/conda.sh
        conda activate parallelwavegan
        bash local/eval_tts_vocoder.sh "${VOCODER_CKPT}" text.tts.rpt_n${n}
    )
done
unset NO_REPEAT_NGRAM

echo "======================================================================"
echo "禁止重复 n-gram 实验跑完。用 local/summarize_norepeat.py 汇总对比。"
echo "======================================================================"
