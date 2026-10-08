#!/usr/bin/env bash
#SBATCH --job-name=sqa_p1_500lp03
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4000M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_opt1.3B_sqa/log/myrun_sqa_p1_sub500_lp03.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_opt1.3B_sqa/log/myrun_sqa_p1_sub500_lp03.%j.err

# 用法: sbatch myrun_sqa_p1_sub500_lp03.sh
# 对齐 lm1_k2000_Bal_sqa/myrun_sqa_p1_sub500_lp03.sh (350M版的真正Part I测试,
# 用的是这个脚本,不是 myrun_sqa_v1_test.sh/eval.sh 里泛用的 test_p1 全量2620+
# decode_sqa_p1.yaml——那两个用的是没有length bonus、没有BERTScore的简化版,
# 不是实际报告用的数字)。OPT-1.3B版,搬过来只换了checkpoint/路径。
#
# test_p1_sub500(dump/v1/raw/test_p1_sub500,500题)直接从 lm1_k2000_Bal_sqa
# 复制过来,不是重新抽样——两边的 test_p1 全量2620条uid已核对完全一致(同一份
# LibriSQA test-clean,prepare_librisqa.py跟模型大小无关,确定性生成),所以
# 500题子集内容也必然一致;抽样算法本身涉及"覆盖全部40个说话人"的分层逻辑,
# 没有必要重新反推一遍去冒不一致的风险。
#
# BERTScore依赖(tools/pylib、tools/hf_bertscore,roberta-large)跟350M版一样
# 软链接自v1——跟k1000/k2000/模型大小无关的共享大资源。
#
# penalty=0.3 沿用v1/350M在dev上选出来的值,没有为opt1.3B重新扫描,跟350M版
# 同样的免责声明:如果opt1.3B的重复生成问题跟350M不一样严重,这个值可能不是
# 本recipe的最优解,先用它看量级。
#
#   0) Part II(全量 test_p2,2620题),finetune(valid.acc.best.pth=12epoch,
#      早停选出)+ zero-shot(7epoch.pth)都跑,两个都要对照(350M版自己的脚本
#      只跑了finetune,zero-shot那边之前单独跑过没搬进来;这里两个都在本脚本
#      里跑,不依赖外部已有结果)
#   1) Part I 解码(全部并行,penalty 0.3,500题子集):
#      finetune(valid.acc.best.pth)  penalty 0.3
#      zero-shot(7epoch.pth,SQA finetune之前、过拟合前最后一个最优点)  penalty 0.3
#   2) 打分: BLEU-4 / ROUGE-L / BERTScore(raw+rescaled, P/R/F1)+ 重复率,附"答案=转写"基线
# 各步已有结果会跳过,可重复提交续跑。
set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_opt1.3B_sqa
. ./path.sh
. ./cmd.sh
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1

FT=exp/sqa_v1_ft
FT_CKPT=valid.acc.best.pth   # 早停在14epoch,best_model_criterion=valid acc 选出12epoch
ZS=exp/zeroshot_7epoch
ZS_CKPT=7epoch.pth
D=dump/v1/raw
TEST=${D}/test_p1_sub500
OUT=${FT}/p1_sub500_lp03
mkdir -p ${OUT}
val() { awk -F'\t' -v k="$2" '$1==k {print $2}' "$1"; }
decode() {  # $1=lm_exp $2=ckpt $3=config $4=data $5=tag ; skip if already scored
    local d=$1/decode_sqa/$(basename $2 .pth)_$(basename $3 .yaml)_$5
    [ -s ${d}/score_sqa/result.txt ] && return 0
    local/eval_sqa.sh --part 1 --lm_exp $1 --data $4 --decode_config $3 --inference_lm $2 --nj 8 --tag $5
}

##############################################################################
# 0) + 1) 并行提交: Part II(全量 test_p2,2620题,finetune+zero-shot都跑)
#         + Part I 解码(penalty=0.3,500题子集)
##############################################################################
pids=()
local/run_mc_loglik.sh --lm_exp ${FT} --ckpt ${FT_CKPT} --data ${D}/test_p2 \
    --out_dir ${FT}/mc_loglik/test_p2/$(basename ${FT_CKPT} .pth) --variants A,B,L & pids+=($!)
local/run_mc_loglik.sh --lm_exp ${ZS} --ckpt ${ZS_CKPT} --data ${D}/test_p2 \
    --out_dir ${ZS}/mc_loglik/test_p2/$(basename ${ZS_CKPT} .pth) --variants A,B,L & pids+=($!)
decode ${FT} ${FT_CKPT} conf/decode_sqa_p1_lp0.3.yaml ${TEST} sub500 & pids+=($!)
decode ${ZS} ${ZS_CKPT} conf/decode_sqa_p1_lp0.3.yaml ${TEST} sub500 & pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done

##############################################################################
# 2) 打分
##############################################################################
FT_TAG=$(basename ${FT_CKPT} .pth)
ZS_TAG=$(basename ${ZS_CKPT} .pth)
rows=("finetune_${FT_TAG} 0.3 ${FT}/decode_sqa/${FT_TAG}_decode_sqa_p1_lp0.3_sub500/text"
      "zeroshot_${ZS_TAG} 0.3 ${ZS}/decode_sqa/${ZS_TAG}_decode_sqa_p1_lp0.3_sub500/text")
pids=()
for row in "${rows[@]}"; do
    set -- ${row}
    s=${OUT}/$1_lp$2; mkdir -p ${s}
    python local/score_librisqa.py --part 1 --hyp "$3" --ref_dir ${TEST} --out ${s}/result.txt > /dev/null
    extra=""; [ "$1_$2" = "finetune_${FT_TAG}_0.3" ] && extra=--with_transcript   # 转写基线只算一次
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
    s=${OUT}/finetune_${FT_TAG}_lp0.3
    printf "transcript_as_answer\t-\t-\t%s\t%s\t%s\t%s\t%s\t%s\t-\n" \
        "$(val ${s}/result.txt baseline_transcript_bleu4)" "$(val ${s}/result.txt baseline_transcript_rougeL_f1)" \
        "$(val ${s}/bertscore.txt baseline_transcript_bertscore_rescaled_P)" "$(val ${s}/bertscore.txt baseline_transcript_bertscore_rescaled_R)" \
        "$(val ${s}/bertscore.txt baseline_transcript_bertscore_rescaled_F1)" "$(val ${s}/bertscore.txt baseline_transcript_bertscore_raw_F1)"
} > ${OUT}/test_summary.tsv
echo "==== Part I test (500 题), BERTScore = rescaled"; column -t -s$'\t' ${OUT}/test_summary.tsv
echo "==== Part II test (2620 题, finetune ${FT_TAG})"; cat ${FT}/mc_loglik/test_p2/${FT_TAG}/result.txt
echo "==== Part II test (2620 题, zero-shot ${ZS_TAG})"; cat ${ZS}/mc_loglik/test_p2/${ZS_TAG}/result.txt
