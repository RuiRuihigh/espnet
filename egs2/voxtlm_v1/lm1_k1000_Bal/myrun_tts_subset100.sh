#!/usr/bin/env bash
#SBATCH --job-name=k1000_tts100
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal/log/myrun_tts_subset100.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal/log/myrun_tts_subset100.%j.err

set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal

. ./path.sh
. ./cmd.sh

# 只对 local/tts_eval_subset/keys.txt 里那 100 条(真值单元长度<200,跨三个
# recipe 完全一致)重新做 TTS 解码 + vocoder 合成 + VERSA 打分,用来跟
# lm1_k2000_Bal / v2 lm1_k2000_Bal 在同一批句子上比 TTS 客观分,不受"长句子
# 自由生成时提前收尾"这个已知问题的干扰,也避开了论文报告口径(100句、2说话人)
# 跟我们全量9957条口径不一致导致数字对不上的问题。
#
# 关键点:输出目录完全独立,不会碰到全量结果。
#   - dump/raw/test/text.tts.subset100 是从 dump/raw/test/text.tts 里用
#     local/tts_eval_subset/keys.txt(加上 tts_ 前缀)过滤出来的100行,
#     跟全量的 dump/raw/test/text.tts 是两个不同文件。
#   - lm.sh Stage 9/10 用 $(basename ${lm_test_text_tts}) 来命名输出目录,
#     所以这次的产物全部落在 exp/.../decode_test_tts/text.tts.subset100/,
#     跟全量结果所在的 exp/.../decode_test_tts/text.tts/ 是两个平行目录,
#     互不覆盖。全量那边的 hyp.tok(9957条)、已经打完的 9647/9957 条 VERSA
#     结果原封不动留在原处。
#   - local/eval_tts_vocoder.sh 加了第二个可选参数 tts_subdir(默认
#     "text.tts"),这里传 "text.tts.subset100",vocoder_eval 的产物同样落在
#     独立子目录 decode_test_tts/text.tts.subset100/vocoder_eval/ 下。

##############################################################################
# Stage 10: 只对 100 条重新解码。--lm_test_text_tts 覆盖掉 run.sh 里默认的
# dump/raw/test/text.tts(全量),run.sh 把 "$@" 放在最后,后传的参数会覆盖
# 前面的同名参数(lm.sh 内部 parse_options.sh 是 last-wins)。
# 100条比9957条快得多,预计几分钟到几十分钟量级,不用等一整晚。
##############################################################################
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1
./run.sh --stage 10 --stop_stage 10 --ngpu 1 \
    --lm_config conf/tuning/train_transformer_opt350.yaml \
    --inference_lm valid.acc.best.pth \
    --lm_test_text_tts dump/raw/test/text.tts.subset100

##############################################################################
# vocoder 合成 + VERSA 打分,跟 myrun_eval.sh 里的第4步是同一个脚本,只是
# 多传了第二个参数指定用 text.tts.subset100 这个子目录,不影响全量那边。
##############################################################################
VOCODER_CKPT=/export/fs06/mliang17/phonologicaltokenizer_modify/ParallelWaveGAN/egs/ljspeech/hubert_voc1/exp/train_nodev_ljspeech_hifigan_hubert.v1/checkpoint-400000steps.pkl

(
    unset PYTHONPATH
    export PATH="${PWD}/utils:${PATH}"
    # shellcheck disable=SC1091
    source /opt/slurm_files/software/conda25/2025-08-21/etc/profile.d/conda.sh
    conda activate parallelwavegan
    bash local/eval_tts_vocoder.sh "${VOCODER_CKPT}" text.tts.subset100
)

echo "======================================================================"
echo "100句子集跑完。结果:"
echo "  exp/lm_train_transformer_opt350_en_bpe10000/decode_test_tts/text.tts.subset100/vocoder_eval/*/versa_eval/avg_result.txt"
echo "======================================================================"
