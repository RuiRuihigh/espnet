#!/usr/bin/env bash
#SBATCH --job-name=sqa_diag
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_sqa/log/myrun_sqa_diag.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_sqa/log/myrun_sqa_diag.%j.err

# Part II 23epoch(test 准确率 34.9%)的两个诊断,只换测试 prompt,不重训:
#   transcript: <startoftext> 标准转写 question..options<generatetext>
#               语音换成 LibriSpeech 原文 -> 看瓶颈在语音理解还是 LM 本身。
#               注意这个格式训练时没见过(文本上下文),低分不一定说明 LM 不行。
#   nospeech  : <startofspeech><startoftext> question..options<generatetext>
#               训练格式不变、语音段为空 -> 看模型到底有没有用到语音。
# 两个并行跑,结果在 exp/sqa_part2_ft/decode_sqa/23epoch_decode_sqa_p2_letter_{transcript,nospeech}/score_sqa/
set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_sqa
. ./path.sh
. ./cmd.sh
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1

pids=()
for t in transcript nospeech; do
    local/eval_sqa.sh --part 2 --lm_exp exp/sqa_part2_ft --data dump/part2/raw/test_${t} \
        --decode_config conf/decode_sqa_p2_letter.yaml --inference_lm 23epoch.pth --nj 8 --tag ${t} &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

for t in transcript nospeech; do
    echo "==== ${t}"; cat exp/sqa_part2_ft/decode_sqa/23epoch_decode_sqa_p2_letter_${t}/score_sqa/result.txt
done
