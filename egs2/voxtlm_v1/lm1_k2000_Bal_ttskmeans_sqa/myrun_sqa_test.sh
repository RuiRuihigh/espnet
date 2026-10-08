#!/usr/bin/env bash
#SBATCH --job-name=k2000_sqa_test
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4000M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans_sqa/log/myrun_sqa_test.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans_sqa/log/myrun_sqa_test.%j.err

# 用法: sbatch myrun_sqa_test.sh ft [ckpt]   # finetune 后,默认 valid.loss.best.pth 指向的 epoch
#       sbatch myrun_sqa_test.sh zs           # zero-shot: lm1_k2000_Bal_ttskmeans 的 17epoch.pth
#
# 测法与 lm1_k1000_Bal_sqa 完全一致(可直接对比):
#   Part II: test_p2 全部 2620 题,按续写概率给 4 个选项原文打分(local/run_mc_loglik.sh)
#            主指标 A-calib,另报 A-sum / A-norm / A-calib_rand,以及 B、L 变体
#   Part I : test_p1_sub500(与 k1000 相同的 500 条),beam5 + 长度奖励 penalty 0.3
#            (penalty 沿用 k1000 在 dev 上选定的值,不重新调),
#            BLEU-4 / ROUGE-L / BERTScore(rescaled P/R/F1 + raw F1, GPU)/ 重复率,附"答案=转写"基线
# 已有结果会跳过,可重复提交续跑。结果汇总: ${LM_EXP}/sqa_results_<ckpt>.txt
set -e
set -u
set -o pipefail

MODEL=${1:?usage: myrun_sqa_test.sh <ft|zs> [ckpt]}

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans_sqa
. ./path.sh
. ./cmd.sh
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1

SRC=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k2000_Bal_ttskmeans/exp/lm_train_transformer_opt350_en_bpe10000
D=dump/v1/raw
if [ "${MODEL}" = zs ]; then
    # 只放软链接,结果写在本目录,不碰 lm1_k2000_Bal_ttskmeans/exp。
    # config.yaml 里 bpemodel 是相对路径 data/en_token_list/...,在本目录解析到同一份 BPE(md5 相同)。
    LM_EXP=exp/zeroshot_17epoch
    if [ ! -e ${LM_EXP}/17epoch.pth ]; then
        mkdir -p ${LM_EXP}
        ln -s ${SRC}/17epoch.pth ${LM_EXP}/17epoch.pth
        ln -s ${SRC}/config.yaml ${LM_EXP}/config.yaml
    fi
    CKPT=17epoch.pth
elif [ "${MODEL}" = ft ]; then
    LM_EXP=exp/sqa_v1_ft
    CKPT=${2:-$(readlink ${LM_EXP}/valid.loss.best.pth)}   # 用实际 epoch 名,结果目录里能看出是哪个
else
    echo "unknown model '${MODEL}' (ft|zs)" >&2; exit 1
fi
TAG=$(basename ${CKPT} .pth)
[ -e ${LM_EXP}/${CKPT} ] || { echo "missing ${LM_EXP}/${CKPT}" >&2; exit 1; }
echo "model=${MODEL} lm_exp=${LM_EXP} ckpt=${CKPT}"

##############################################################################
# 1) Part II 打分 + Part I 解码(并行)
##############################################################################
local/run_mc_loglik.sh --lm_exp ${LM_EXP} --ckpt ${CKPT} --data ${D}/test_p2 \
    --out_dir ${LM_EXP}/mc_loglik/test_p2/${TAG} --variants A,B,L &
p2=$!
P1=${LM_EXP}/decode_sqa/${TAG}_decode_sqa_p1_lp0.3_sub500
if [ ! -s ${P1}/score_sqa/result.txt ]; then
    local/eval_sqa.sh --part 1 --lm_exp ${LM_EXP} --data ${D}/test_p1_sub500 \
        --decode_config conf/decode_sqa_p1_lp0.3.yaml --inference_lm ${CKPT} --nj 8 --tag sub500 &
    p1=$!
    wait ${p1}
fi
wait ${p2}

##############################################################################
# 2) Part I: BERTScore(GPU, 含转写基线)+ 重复率
##############################################################################
if [ ! -s ${P1}/score_sqa/bertscore.txt ]; then
    ${cuda_cmd} --gpu 1 ${P1}/score_sqa/bertscore.log \
        env PYTHONPATH=${PWD}/tools/pylib HF_HOME=${PWD}/tools/hf_bertscore \
        python local/score_bertscore.py --hyp ${P1}/text --ref_dir ${D}/test_p1_sub500 \
            --out ${P1}/score_sqa/bertscore.txt --device cuda --with_transcript
fi
REP=$(python3 -c "
import sys
n=r=0
for l in open(sys.argv[1]):
    w=l.rstrip().split('<generatetext>')[-1].split(); ng=[tuple(w[i:i+4]) for i in range(len(w)-3)]
    n+=1; r+=len(ng)!=len(set(ng))
print(f'{r/n*100:.1f}')" ${P1}/text)

##############################################################################
# 3) 汇总
##############################################################################
val() { awk -F'\t' -v k="$2" '$1==k {print $2}' "$1"; }
R=${P1}/score_sqa/result.txt; B=${P1}/score_sqa/bertscore.txt
{
    echo "==== ${MODEL} (${LM_EXP}/${CKPT})"
    echo "---- Part II test (2620 题; random 25 / always-c 32.14)"
    cat ${LM_EXP}/mc_loglik/test_p2/${TAG}/result.txt
    echo "---- Part I test (500 题, beam5, penalty 0.3; BERTScore rescaled)"
    printf "system\tavg_words\tBLEU4\tROUGE-L\tBERTScore_P\tBERTScore_R\tBERTScore_F1\tBERTScore_F1_raw\trepeat%%\n"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" ${MODEL}_${TAG} "$(val $R avg_hyp_words)" "$(val $R bleu4)" "$(val $R rougeL_f1)" \
        "$(val $B model_bertscore_rescaled_P)" "$(val $B model_bertscore_rescaled_R)" "$(val $B model_bertscore_rescaled_F1)" \
        "$(val $B model_bertscore_raw_F1)" "${REP}"
    printf "transcript_as_answer\t-\t%s\t%s\t%s\t%s\t%s\t%s\t-\n" "$(val $R baseline_transcript_bleu4)" "$(val $R baseline_transcript_rougeL_f1)" \
        "$(val $B baseline_transcript_bertscore_rescaled_P)" "$(val $B baseline_transcript_bertscore_rescaled_R)" \
        "$(val $B baseline_transcript_bertscore_rescaled_F1)" "$(val $B baseline_transcript_bertscore_raw_F1)"
} > ${LM_EXP}/sqa_results_${TAG}.txt
cat ${LM_EXP}/sqa_results_${TAG}.txt
