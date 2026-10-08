#!/usr/bin/env bash
#SBATCH --job-name=sqa_p1_full
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4000M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_sqa/log/myrun_sqa_p1_full.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_sqa/log/myrun_sqa_p1_full.%j.err

# 用法: sbatch myrun_sqa_p1_full.sh
# Part I 全量 test(test_p1, 2620 题)。长度奖励 penalty=0.3(dev_p1_sub200 上选定:
# BERTScore F1 46.66,与 0.4 的 46.71 打平,但 0.3 精确率更高 49.8 vs 45.5、重复答案少一半 16 vs 29/200)。
#   1) 解码(全部并行):
#      V1 15epoch  penalty 0.3 全量
#      V1 15epoch  penalty 0   只补剩下的 1602 条(另 1018 条之前已解码),再合并成全量
#      zero-shot 13epoch  penalty 0 / 0.3 全量
#   2) 打分: BLEU-4 / ROUGE-L / BERTScore(raw+rescaled, P/R/F1)+ 重复率,附"答案=转写"基线
#      BERTScore 这次放 GPU 上算(2620 条在 CPU 上每个系统约 1 小时)
# 各步已有结果会跳过,可重复提交续跑。
set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal_sqa
. ./path.sh
. ./cmd.sh
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1

V1=exp/sqa_v1_ft
ZS=exp/zeroshot_13epoch
D=dump/v1/raw
TEST=${D}/test_p1
OUT=${V1}/p1_full
mkdir -p ${OUT}
val() { awk -F'\t' -v k="$2" '$1==k {print $2}' "$1"; }
decode() {  # $1=lm_exp $2=ckpt $3=config $4=data $5=tag ; skip if already scored
    local d=$1/decode_sqa/$(basename $2 .pth)_$(basename $3 .yaml)_$5
    [ -s ${d}/score_sqa/result.txt ] && return 0
    local/eval_sqa.sh --part 1 --lm_exp $1 --data $4 --decode_config $3 --inference_lm $2 --nj 8 --tag $5
}

##############################################################################
# 1) 解码
##############################################################################
pids=()
decode ${V1} 15epoch.pth conf/decode_sqa_p1_lp0.3.yaml ${TEST} full & pids+=($!)
decode ${V1} 15epoch.pth conf/decode_sqa_p1.yaml ${D}/test_p1_rest1602 rest1602 & pids+=($!)
decode ${ZS} 13epoch.pth conf/decode_sqa_p1.yaml ${TEST} full & pids+=($!)
decode ${ZS} 13epoch.pth conf/decode_sqa_p1_lp0.3.yaml ${TEST} full & pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done

# V1 penalty 0: 之前的 1018 条 + 这次补的 1602 条 -> 全量
V1_P0=${OUT}/V1_lp0.0_merged.text
cat ${V1}/decode_sqa/15epoch_decode_sqa_p1/score_sqa_partial/text \
    ${V1}/decode_sqa/15epoch_decode_sqa_p1_rest1602/text | sort -k1,1 > ${V1_P0}
[ "$(wc -l < ${V1_P0})" = 2620 ] || { echo "V1 penalty0 merged has $(wc -l < ${V1_P0}) lines, expected 2620" >&2; exit 1; }

##############################################################################
# 2) 打分
##############################################################################
rows=("V1_15epoch 0.0 ${V1_P0}"
      "V1_15epoch 0.3 ${V1}/decode_sqa/15epoch_decode_sqa_p1_lp0.3_full/text"
      "zeroshot_13epoch 0.0 ${ZS}/decode_sqa/13epoch_decode_sqa_p1_full/text"
      "zeroshot_13epoch 0.3 ${ZS}/decode_sqa/13epoch_decode_sqa_p1_lp0.3_full/text")
pids=()
for row in "${rows[@]}"; do
    set -- ${row}
    s=${OUT}/$1_lp$2; mkdir -p ${s}
    python local/score_librisqa.py --part 1 --hyp "$3" --ref_dir ${TEST} --out ${s}/result.txt > /dev/null
    extra=""; [ "$1_$2" = V1_15epoch_0.0 ] && extra=--with_transcript   # 转写基线只算一次
    if [ ! -s ${s}/bertscore.txt ]; then
        ${cuda_cmd} --gpu 1 ${s}/bertscore.log \
            env PYTHONPATH=${PWD}/tools/pylib HF_HOME=${PWD}/tools/hf_bertscore \
            python local/score_bertscore.py --hyp "$3" --ref_dir ${TEST} --out ${s}/bertscore.txt --device cuda ${extra} &
        pids+=($!)
    fi
done
for p in "${pids[@]}"; do wait "$p"; done

rep_rate() {  # % answers containing a repeated 4-gram
    python3 -c "
import sys
n=r=0
for l in open(sys.argv[1]):
    w=l.rstrip().split('<generatetext>')[-1].split(); ng=[tuple(w[i:i+4]) for i in range(len(w)-3)]
    n+=1; r+=len(ng)!=len(set(ng))
print(f'{r/n*100:.1f}')" "$1"
}
{
    printf "system\tpenalty\tavg_words\tBLEU4\tROUGE-L\tBERTScore_P\tBERTScore_R\tBERTScore_F1\tBERTScore_F1_raw\trepeat%%\n"
    for row in "${rows[@]}"; do
        set -- ${row}
        s=${OUT}/$1_lp$2
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" $1 $2 "$(val ${s}/result.txt avg_hyp_words)" \
            "$(val ${s}/result.txt bleu4)" "$(val ${s}/result.txt rougeL_f1)" \
            "$(val ${s}/bertscore.txt model_bertscore_rescaled_P)" "$(val ${s}/bertscore.txt model_bertscore_rescaled_R)" \
            "$(val ${s}/bertscore.txt model_bertscore_rescaled_F1)" "$(val ${s}/bertscore.txt model_bertscore_raw_F1)" \
            "$(rep_rate "$3")"
    done
    s=${OUT}/V1_15epoch_lp0.0
    printf "transcript_as_answer\t-\t-\t%s\t%s\t%s\t%s\t%s\t%s\t-\n" \
        "$(val ${s}/result.txt baseline_transcript_bleu4)" "$(val ${s}/result.txt baseline_transcript_rougeL_f1)" \
        "$(val ${s}/bertscore.txt baseline_transcript_bertscore_rescaled_P)" "$(val ${s}/bertscore.txt baseline_transcript_bertscore_rescaled_R)" \
        "$(val ${s}/bertscore.txt baseline_transcript_bertscore_rescaled_F1)" "$(val ${s}/bertscore.txt baseline_transcript_bertscore_raw_F1)"
} > ${OUT}/test_summary.tsv
echo "==== Part I test (2620 题), BERTScore = rescaled"; column -t -s$'\t' ${OUT}/test_summary.tsv
