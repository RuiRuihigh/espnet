#!/usr/bin/env bash
#SBATCH --job-name=sqa_p1_500lp03
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4000M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_sqa/log/myrun_sqa_p1_sub500_lp03.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_sqa/log/myrun_sqa_p1_sub500_lp03.%j.err

# 用法: sbatch myrun_sqa_p1_sub500_lp03.sh
# 对齐 voxtlm_v1/lm1_k1000_Bal_sqa/myrun_sqa_p1_sub500_lp03.sh:同一套 penalty=0.3
# 长度奖励测试,搬到 k2000 / Phonological Tokenizer。
#
# test_p1_sub500(dump/v1/raw/test_p1_sub500,500 题)不是抄 v1 现成的 uid 列表,是
# 用反推出来的同一算法在本recipe自己的 test_p1 上重新生成的:
#   chosen = set(random.Random(0).sample(sorted(全部 test_p1 uid), 500))
#   按 test_p1 原始顺序过滤出这些 uid
# v1/v2 的 test_p1 uid 集合和顺序完全一致(同一份 LibriSQA test-clean,Part I 的
# test 不做 unk/长度过滤),验证过重新生成的 500 个 uid 跟 v1 的 test_p1_sub500
# 逐行一致。
#
# BERTScore 依赖(tools/pylib、tools/hf_bertscore,roberta-large)软链接自 v1 ——
# 跟 k1000/k2000 无关的共享大资源,同一套 espnet venv,不用重装/重下载。
#
# penalty=0.3 是 v1 自己在 dev 上选出来的值(dev_p1_sub200 上 BERTScore F1 46.66,
# 跟 0.4 的 46.71 打平但精确率更高、重复答案更少)。这里直接复用这个值,没有为
# k2000 重新做一次 dev 扫描——如果 k2000 的重复生成问题跟 k1000 不一样严重,这个
# 值可能不是本recipe的最优解,先用它看量级,要更严谨可以另外跑一次 dev 扫描
# (仿 v1 的 myrun_sqa_p1_lp.sh)。
#
#   0) Part II(全量 test_p2,2620 题),finetune(18epoch)一个,跟 v1 的
#      exp/sqa_v1_ft/mc_loglik/test_p2/15epoch 对齐;zero-shot(16epoch)的 Part II
#      之前已经单独跑过,不在这里重跑
#   1) Part I 解码(全部并行,penalty 0.3,500 题子集):
#      finetune(18epoch.pth)  penalty 0.3
#      zero-shot(16epoch.pth) penalty 0.3
#      (只跑 penalty 0.3;不加奖励的对照数字之前在别的地方跑过,这里不重跑)
#      跟 0) 一起并行提交,不用等 Part II 跑完再开始 Part I
#   2) 打分: BLEU-4 / ROUGE-L / BERTScore(raw+rescaled, P/R/F1)+ 重复率,附"答案=转写"基线
#      BERTScore 放 GPU 上算
# 各步已有结果会跳过,可重复提交续跑。
set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_sqa
. ./path.sh
. ./cmd.sh
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1

FT=exp/sqa_v1_ft
FT_CKPT=18epoch.pth   # 训练过拟合后手动叫停,best_model_criterion=valid acc 选出的 checkpoint
ZS=exp/zeroshot_16epoch
ZS_CKPT=16epoch.pth
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
# 0) + 1) 并行提交: Part II(全量 test_p2,2620 题;跟 v1 的
#    exp/sqa_v1_ft/mc_loglik/test_p2/15epoch 对齐,不用 Part I 的 500 子集 ——
#    mc_loglik 打分不是 beam 生成,不需要为了省时间抽样,v1 自己测 Part II 时用的
#    也是全量) + Part I 解码(penalty=0.3,500 题子集)
##############################################################################
pids=()
local/run_mc_loglik.sh --lm_exp ${FT} --ckpt ${FT_CKPT} --data ${D}/test_p2 \
    --out_dir ${FT}/mc_loglik/test_p2/$(basename ${FT_CKPT} .pth) --variants A,B,L & pids+=($!)
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
