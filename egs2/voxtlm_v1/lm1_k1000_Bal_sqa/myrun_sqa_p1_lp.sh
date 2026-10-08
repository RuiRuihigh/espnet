#!/usr/bin/env bash
#SBATCH --job-name=sqa_p1_lp
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_sqa/log/myrun_sqa_p1_lp.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_sqa/log/myrun_sqa_p1_lp.%j.err

# 用法: sbatch myrun_sqa_p1_lp.sh            # 全部三步
#       STAGE=2 sbatch myrun_sqa_p1_lp.sh    # 从第2步开始(已选好 penalty 时)
#       PENALTIES="0.1 0.2 0.3 0.4" STOP_STAGE=1 sbatch myrun_sqa_p1_lp.sh   # 只在 dev 上补扫这些 penalty
# dev 上已有结果(result.txt / bertscore.txt)的 penalty 不会重跑;dev_sweep.tsv 汇总所有已跑过的 penalty。
#
# Part I 重新解码: 加长度奖励 + BERTScore,V1(15epoch)和 zero-shot(原始 13epoch)都测。
# 背景: V1 不加奖励时答案平均 8.1 词(参考 15.4),BLEU/ROUGE 被短答案拖低,但人工看答案大多是对的。
# penalty = lm_inference 的 length bonus: beam search 里每多生成一个 token 分数 +penalty。
#
#   1) 在 dev(dev_p1_sub200: dev_p1 随机200条, seed 0, 覆盖全部17个 dev 说话人)上
#      用 V1 扫 penalty 0 / 0.5 / 1.0 / 1.5,按 BERTScore F1(rescaled)选最好的 -> 只在 dev 上选,不看 test
#   2) test(test_p1_sub200: 从之前 V1 无奖励已解码的 1018 条里随机抽 200, seed 0):
#      V1 用选中的 penalty;zero-shot 13epoch 用 penalty 0 和选中的 penalty 各一次
#      (V1 penalty 0 直接复用之前 1018 条的解码结果,只按这 200 条重新打分,不重跑)
#   3) 全部 test 结果算 BLEU / ROUGE-L / BERTScore(raw + rescaled),附"答案=转写"基线,汇总成表
#
# BERTScore: roberta-large(layer 17, bert-score 0.3.13 默认),装在 tools/pylib(--no-deps,
# 不动共享 venv),模型缓存在 tools/hf_bertscore(本目录,不写共享 hf_cache)。在本控制任务的 CPU 上算。
set -e
set -u
set -o pipefail

STAGE=${STAGE:-1}
STOP_STAGE=${STOP_STAGE:-3}
PENALTIES=${PENALTIES:-"0.0 0.5 1.0 1.5"}
cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_sqa
. ./path.sh
. ./cmd.sh
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1

V1=exp/sqa_v1_ft
ZS=exp/zeroshot_13epoch
D=dump/v1/raw
DEV=${D}/dev_p1_sub200
TEST=${D}/test_p1_sub200
OUT=${V1}/p1_lp
mkdir -p ${OUT}

bertscore() {  # $1=decode text  $2=ref dir  $3=out  [$4=--with_transcript]
    PYTHONPATH=${PWD}/tools/pylib HF_HOME=${PWD}/tools/hf_bertscore \
        python local/score_bertscore.py --hyp "$1" --ref_dir "$2" --out "$3" --device cpu ${4:-} > /dev/null
}
cfg_name() { [ "$1" = 0.0 ] && echo decode_sqa_p1 || echo decode_sqa_p1_lp$1; }
val() { awk -F'\t' -v k="$2" '$1==k {print $2}' "$1"; }

##############################################################################
# 1) dev: 扫 penalty,按 BERTScore F1(rescaled) 选
##############################################################################
if [ ${STAGE} -le 1 ]; then
    pids=()
    for lp in ${PENALTIES}; do
        d=${V1}/decode_sqa/15epoch_$(cfg_name ${lp})_dev200
        [ -s ${d}/score_sqa/result.txt ] && continue   # 已解码过
        local/eval_sqa.sh --part 1 --lm_exp ${V1} --data ${DEV} --decode_config conf/$(cfg_name ${lp}).yaml \
            --inference_lm 15epoch.pth --nj 4 --tag dev200 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
    # 汇总所有已跑过的 penalty(包括之前跑的)
    rows=""
    for d in ${V1}/decode_sqa/15epoch_decode_sqa_p1*_dev200; do
        [ -s ${d}/score_sqa/result.txt ] || continue
        [ -s ${d}/score_sqa/bertscore.txt ] || bertscore ${d}/text ${DEV} ${d}/score_sqa/bertscore.txt
        lp=$(basename ${d} | sed -E 's/^15epoch_decode_sqa_p1(_lp([0-9.]+))?_dev200$/\2/'); lp=${lp:-0.0}
        rows+=$(printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" ${lp} "$(val ${d}/score_sqa/result.txt avg_hyp_words)" \
            "$(val ${d}/score_sqa/result.txt bleu4)" "$(val ${d}/score_sqa/result.txt rougeL_f1)" \
            "$(val ${d}/score_sqa/bertscore.txt model_bertscore_rescaled_P)" \
            "$(val ${d}/score_sqa/bertscore.txt model_bertscore_rescaled_R)" \
            "$(val ${d}/score_sqa/bertscore.txt model_bertscore_rescaled_F1)" \
            "$(val ${d}/score_sqa/bertscore.txt model_bertscore_raw_F1)")$'\n'
    done
    { printf "penalty\tavg_words\tBLEU4\tROUGE-L\tBERTScore_P_rescaled\tBERTScore_R_rescaled\tBERTScore_F1_rescaled\tBERTScore_F1_raw\n"
      printf "%s" "${rows}" | sort -g -k1,1; } > ${OUT}/dev_sweep.tsv
    awk -F'\t' 'NR>1 {if ($7>m) {m=$7; b=$1}} END {print b}' ${OUT}/dev_sweep.tsv > ${OUT}/chosen_penalty
    echo "==== dev sweep (chosen penalty by BERTScore F1 rescaled: $(cat ${OUT}/chosen_penalty))"; column -t -s$'\t' ${OUT}/dev_sweep.tsv
fi
[ ${STOP_STAGE} -le 1 ] && exit 0
LP=$(cat ${OUT}/chosen_penalty)

##############################################################################
# 2) test: V1(选中的 penalty) + zero-shot(0 和选中的 penalty)
##############################################################################
if [ ${STAGE} -le 2 ] && [ ${STOP_STAGE} -ge 2 ]; then
    pids=()
    if [ "${LP}" != 0.0 ]; then   # penalty 0 的 V1 结果已有,不重跑
        local/eval_sqa.sh --part 1 --lm_exp ${V1} --data ${TEST} --decode_config conf/$(cfg_name ${LP}).yaml \
            --inference_lm 15epoch.pth --nj 8 --tag sub200 &
        pids+=($!)
    fi
    for lp in $(printf "0.0\n%s\n" ${LP} | sort -u); do
        local/eval_sqa.sh --part 1 --lm_exp ${ZS} --data ${TEST} --decode_config conf/$(cfg_name ${lp}).yaml \
            --inference_lm 13epoch.pth --nj 8 --tag sub200 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
fi

##############################################################################
# 3) test 汇总: BLEU / ROUGE-L / BERTScore + 转写基线(同 1018 题)
##############################################################################
V1_P0=${V1}/decode_sqa/15epoch_decode_sqa_p1/score_sqa_partial/text   # 之前 V1 无奖励的 1018 条;打分只取 TEST 里的 200 条
{
    printf "system\tpenalty\tavg_words\tBLEU4\tROUGE-L\tBERTScore_F1_rescaled\tBERTScore_F1_raw\n"
    rows=("V1_15epoch 0.0 ${V1_P0}" "zeroshot_13epoch 0.0 ${ZS}/decode_sqa/13epoch_decode_sqa_p1_sub200/text")
    if [ "${LP}" != 0.0 ]; then
        rows+=("V1_15epoch ${LP} ${V1}/decode_sqa/15epoch_$(cfg_name ${LP})_sub200/text"
               "zeroshot_13epoch ${LP} ${ZS}/decode_sqa/13epoch_$(cfg_name ${LP})_sub200/text")
    fi
    for row in "${rows[@]}"; do
        set -- ${row}
        [ -s "$3" ] || continue
        s=${OUT}/test_$1_lp$2
        mkdir -p ${s}
        python local/score_librisqa.py --part 1 --hyp "$3" --ref_dir ${TEST} --out ${s}/result.txt > /dev/null
        bertscore "$3" ${TEST} ${s}/bertscore.txt
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" $1 $2 "$(val ${s}/result.txt avg_hyp_words)" \
            "$(val ${s}/result.txt bleu4)" "$(val ${s}/result.txt rougeL_f1)" \
            "$(val ${s}/bertscore.txt model_bertscore_rescaled_F1)" "$(val ${s}/bertscore.txt model_bertscore_raw_F1)"
    done
    # 转写基线
    s=${OUT}/test_baseline_transcript; mkdir -p ${s}
    bertscore ${V1_P0} ${TEST} ${s}/bertscore.txt --with_transcript
    b=${OUT}/test_V1_15epoch_lp0.0/result.txt
    printf "transcript_as_answer\t-\t-\t%s\t%s\t%s\t%s\n" "$(val ${b} baseline_transcript_bleu4)" "$(val ${b} baseline_transcript_rougeL_f1)" \
        "$(val ${s}/bertscore.txt baseline_transcript_bertscore_rescaled_F1)" "$(val ${s}/bertscore.txt baseline_transcript_bertscore_raw_F1)"
} > ${OUT}/test_summary.tsv
echo "==== test Part I, 200 题"; column -t -s$'\t' ${OUT}/test_summary.tsv
