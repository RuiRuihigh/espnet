#!/usr/bin/env bash
#SBATCH --job-name=sqa_p1_500lp0
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4000M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_opt1.3B_sqa/log/myrun_sqa_p1_sub500_lp0.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v2/lm1_k2000_Bal_opt1.3B_sqa/log/myrun_sqa_p1_sub500_lp0.%j.err

# 用法: sbatch myrun_sqa_p1_sub500_lp0.sh
# myrun_sqa_p1_sub500_lp03.sh(照抄350M/v1用的penalty=0.3)跑出来的结果作废了,
# 两个问题:
#
#   1) penalty=0.3 在 OPT-1.3B 上彻底退化成乱码,FT 和 zero-shot 两个checkpoint
#      500句**全部**输出一样的垃圾CJK字符,跟模型权重无关,纯粹是解码配置问题。
#      证据(lm_inference.*.log 里任意一句):
#        -2.07 * 1.0 = -2.07 for lm
#         8.00 * 0.3 =  2.40 for length_bonus
#        total log probability: 0.33   <- 正数,真实log概率不可能是正的
#      length_bonus是每多生成一个token就加0.3分的flat奖励,跟token内容无关。
#      0.3这个值是在350M/v1上调出来的"安全值"(小于350M当时的真实每token代价),
#      但OPT-1.3B是明显更强/更自信的模型,对这类文本任务真实的每token平均代价
#      (负log似然)比350M更小,现在奖励(0.3)反而超过了真实代价,束搜索发现
#      "不管生成什么,多生成一个token就净赚",退化成堆垃圾堆到maxlen=120。
#      这里直接改用 penalty=0(conf/decode_sqa_p1.yaml,其余跟lp0.3版同一套
#      beam_size=5/maxlen=120/minlen=1),等彻底跑干净再考虑是否要为1.3B
#      重新扫一个安全的非零penalty。
#
#   2) Part II(run_mc_loglik.sh)在普通gpu分区(10.9GB)上跑OPT-1.3B直接CUDA OOM,
#      FT/zero-shot两个都是8/8分片全挂。跟SQA finetune训练(Stage6)踩过的
#      内存坑同一类问题,只是这次是推理不是训练。修法:跟 myrun_sqa_v1_train.sh
#      一样,用 SQA_CUDA_CMD 环境变量临时切到 A100(cmd.sh 里
#      cuda_cmd="${SQA_CUDA_CMD:-slurm.pl}",设了这个环境变量就会在A100跑)。
#      注意:decode()(Part I)不能跟着一起切到A100——它在普通GPU上跑没有OOM
#      (问题只在penalty,不在显存),且Part I/II原来是4个任务一起并行提交的,
#      但cmd.sh是在各个local/*.sh内部重新source的纯文件状态,两种GPU路由不能
#      在同一时刻并行生效,所以这版改成两段:先A100跑完两个Part II,再切回
#      普通GPU跑两个Part I,段内仍然并行。
#
# test_p1_sub500 / tools/pylib、tools/hf_bertscore 的来源跟 lp03 版一样,见那
# 个脚本的注释,这里不重复。
#
#   0) Part II(全量 test_p2,2620题,finetune+zero-shot都跑,A100)
#   1) Part I 解码(全部并行,penalty 0,500题子集,普通GPU):
#      finetune(valid.acc.best.pth)  penalty 0
#      zero-shot(7epoch.pth)         penalty 0
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
OUT=${FT}/p1_sub500_lp0
mkdir -p ${OUT}
val() { awk -F'\t' -v k="$2" '$1==k {print $2}' "$1"; }
decode() {  # $1=lm_exp $2=ckpt $3=config $4=data $5=tag ; skip if already scored
    local d=$1/decode_sqa/$(basename $2 .pth)_$(basename $3 .yaml)_$5
    [ -s ${d}/score_sqa/result.txt ] && return 0
    local/eval_sqa.sh --part 1 --lm_exp $1 --data $4 --decode_config $3 --inference_lm $2 --nj 8 --tag $5
}

##############################################################################
# 0) Part II(全量 test_p2,2620题,finetune+zero-shot都跑) -- A100,避免OOM
##############################################################################
export SQA_CUDA_CMD="slurm.pl --config conf/slurm_clsp_a100.conf"
. ./cmd.sh
# nj=1(不是默认的8),且两个checkpoint顺序跑(不并行) -- A100是稀缺资源,用户
# 明确说了没有那么多A100,之前一次性提交两个nj=8的并行调用相当于同时要16张,
# 排队里一堆别的任务也在抢,改成任何时刻最多占1张。
local/run_mc_loglik.sh --lm_exp ${FT} --ckpt ${FT_CKPT} --data ${D}/test_p2 \
    --out_dir ${FT}/mc_loglik/test_p2/$(basename ${FT_CKPT} .pth) --variants A,B,L --nj 1
local/run_mc_loglik.sh --lm_exp ${ZS} --ckpt ${ZS_CKPT} --data ${D}/test_p2 \
    --out_dir ${ZS}/mc_loglik/test_p2/$(basename ${ZS_CKPT} .pth) --variants A,B,L --nj 1
unset SQA_CUDA_CMD
. ./cmd.sh

##############################################################################
# 1) Part I 解码(penalty=0,500题子集) -- 普通GPU
##############################################################################
pids=()
decode ${FT} ${FT_CKPT} conf/decode_sqa_p1.yaml ${TEST} sub500 & pids+=($!)
decode ${ZS} ${ZS_CKPT} conf/decode_sqa_p1.yaml ${TEST} sub500 & pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done

##############################################################################
# 2) 打分
##############################################################################
FT_TAG=$(basename ${FT_CKPT} .pth)
ZS_TAG=$(basename ${ZS_CKPT} .pth)
rows=("finetune_${FT_TAG} 0 ${FT}/decode_sqa/${FT_TAG}_decode_sqa_p1_sub500/text"
      "zeroshot_${ZS_TAG} 0 ${ZS}/decode_sqa/${ZS_TAG}_decode_sqa_p1_sub500/text")
pids=()
for row in "${rows[@]}"; do
    set -- ${row}
    s=${OUT}/$1_lp$2; mkdir -p ${s}
    python local/score_librisqa.py --part 1 --hyp "$3" --ref_dir ${TEST} --out ${s}/result.txt > /dev/null
    extra=""; [ "$1_$2" = "finetune_${FT_TAG}_0" ] && extra=--with_transcript   # 转写基线只算一次
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
    s=${OUT}/finetune_${FT_TAG}_lp0
    printf "transcript_as_answer\t-\t-\t%s\t%s\t%s\t%s\t%s\t%s\t-\n" \
        "$(val ${s}/result.txt baseline_transcript_bleu4)" "$(val ${s}/result.txt baseline_transcript_rougeL_f1)" \
        "$(val ${s}/bertscore.txt baseline_transcript_bertscore_rescaled_P)" "$(val ${s}/bertscore.txt baseline_transcript_bertscore_rescaled_R)" \
        "$(val ${s}/bertscore.txt baseline_transcript_bertscore_rescaled_F1)" "$(val ${s}/bertscore.txt baseline_transcript_bertscore_raw_F1)"
} > ${OUT}/test_summary.tsv
echo "==== Part I test (500 题), BERTScore = rescaled"; column -t -s$'\t' ${OUT}/test_summary.tsv
echo "==== Part II test (2620 题, finetune ${FT_TAG})"; cat ${FT}/mc_loglik/test_p2/${FT_TAG}/result.txt
echo "==== Part II test (2620 题, zero-shot ${ZS_TAG})"; cat ${ZS}/mc_loglik/test_p2/${ZS_TAG}/result.txt
