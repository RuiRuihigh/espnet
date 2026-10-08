#!/usr/bin/env bash
#SBATCH --job-name=sqa_mc
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_sqa/log/myrun_sqa_mc.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_sqa/log/myrun_sqa_mc.%j.err

# Part II 不训练的打分:按续写概率选答案(lm-eval-harness 做法,GPTQ 等论文的
# zero-shot 选择题都是这么测的),替代"贪心生成 the answer is option x"。
# 变体(见 local/score_mc_loglik.py):
#   A cloze : prompt 里不列选项,续写=选项原文(harness 原版)
#   B listed: prompt 同训练(列选项),续写=选项原文
#   L letter: prompt 同训练,续写="the answer is option x"
# 每种都报 sum / norm(按字符数归一,harness 的 acc_norm)/ calib(减去语音段为空时的分数)
# / nospeech(只看无语音)。两个模型:23epoch(finetune 后)和原始 13epoch(zero-shot)。
# 每个模型切 8 片在普通 gpu 上跑。
set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_sqa
. ./path.sh
. ./cmd.sh
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1

NJ=8
pids=()
for m in "exp/sqa_part2_ft 23epoch.pth" "exp/zeroshot_13epoch 13epoch.pth"; do
    set -- $m
    lm_exp=$1; ckpt=$2
    d=${lm_exp}/mc_loglik/$(basename ${ckpt} .pth)
    mkdir -p ${d}/logdir
    split_scps=""
    for n in $(seq ${NJ}); do split_scps+=" ${d}/logdir/keys.${n}.scp"; done
    utils/split_scp.pl dump/part2/raw/test/text.sqa ${split_scps}
    (
        ${cuda_cmd} --gpu 1 JOB=1:${NJ} ${d}/logdir/score.JOB.log \
            python local/score_mc_loglik.py --lm_train_config ${lm_exp}/config.yaml \
                --lm_file ${lm_exp}/${ckpt} --keys ${d}/logdir/keys.JOB.scp \
                --out ${d}/logdir/scores.JOB.jsonl
        python local/score_mc_loglik.py summarize ${d}/result.txt ${d}/logdir/scores.*.jsonl
    ) &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

for d in exp/sqa_part2_ft/mc_loglik/23epoch exp/zeroshot_13epoch/mc_loglik/13epoch; do
    echo "==== ${d}"; grep -h boundary_mismatch ${d}/logdir/score.*.log | sort | uniq -c; cat ${d}/result.txt
done
