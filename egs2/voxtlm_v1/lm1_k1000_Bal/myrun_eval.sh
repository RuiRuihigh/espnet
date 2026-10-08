#!/usr/bin/env bash
#SBATCH --job-name=k1000_eval
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal/log/myrun_eval.%j.out
#SBATCH --error=/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal/log/myrun_eval.%j.err

set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/voxtlm_officialtest/espnet/egs2/voxtlm_v1/lm1_k1000_Bal

. ./path.sh
. ./cmd.sh

# 一个脚本跑完 k1000 (D_Bal, hubert_base 第6层 + 官方 km_1000) 的全部评测。
# 用的 checkpoint 是 valid.acc.best.pth = 13epoch.pth(train_acc 还在涨但
# valid_loss 从第13轮的 2.728 连涨5轮到 2.750,已经过拟合,训练已手动停在19轮)。
#
# 四块内容,顺序执行(前面失败会直接退出,不会带着半成品往下走):
#   1) Stage 8   : textlm / speechlm 的 PPL
#   2) Stage 9-10: ASR 解码(WER/CER)、TTS 解码(出 hyp.tok 离散单元序列)
#   3) sLM21     : sWUGGY(词汇) + sBLIMP(句法),文本轨 + 语音轨 + 打分
#   4) vocoder   : 把 Stage 10 的 hyp.tok 合成成波形,再用 VERSA 算
#                  CER(whisper_wer) / UTMOS 等客观分
#
# GPU 策略跟之前一致:默认全部走普通 gpu 分区,只有第4步里 VERSA 那一下
# (同时加载所有指标模型要约11.5GB)才临时切 A100,脚本内部用 trap 改回来。
# 训练已经结束,这里没有任何一步需要长时间占 A100。

##############################################################################
# 1-2) Stage 8 (PPL) + Stage 9 (ASR 解码) + Stage 10 (TTS 解码)
#
# 三个 stage 都是推理:Stage 8 单次前向;Stage 9/10 是 beam search,batch=1,
# 普通 gpu 够用。--inference_lm valid.acc.best.pth 覆盖掉 run.sh 里默认的
# valid.acc.ave.pth(多checkpoint平均),跟老 checkout lm1 和 v2 k2000 的
# 评测口径保持一致,数字才可比。
#
# 结果分别在:
#   exp/lm_train_.../perplexity_test/{textlm,speechlm}/ppl
#   exp/lm_train_.../decode_test_asr/text.asr/score_wer/result.txt
#   exp/lm_train_.../decode_test_tts/text.tts/score_tts/hyp.tok
##############################################################################
#
# [已跑完,job 1801511(2026-09-20 14:49 ~ 09-21 08:07):
#    textLM PPL      27.26
#    speechLM PPL    42.67
#    ASR WER         合计 12.4%(test-clean 7.1% / test-other 17.8%)
#    TTS 解码        9957 条 hyp.tok 全部生成
#  注意 job 1801511 最后是 FAILED 状态,但不是这三个 stage 算错了:Stage 10 的
#  8个分片 08:07:23 全部 status 0 正常结束,16秒后控制脚本读 lm.sh 下一行时报
#  "Stale file handle" 挂掉。原因是当时往共享的 TEMPLATE/lm1/lm.sh 加
#  test-clean/test-other 分开打分的逻辑时,用的是"改副本再 mv 覆盖"的做法——
#  这在本地盘上对正在运行的脚本是安全的(旧 inode 还被进程持有),但这里是 NFS,
#  旧文件一旦被 unlink,客户端的文件句柄就失效,正在跑的 bash 再去读脚本就报错。
#  教训:要改正在被运行中的任务读取的脚本(哪怕是 mv 覆盖),也得等任务结束。
#  test-clean/test-other 的分开打分已经事后用现成的 ref.trn/hyp.trn 补算过了。]
export HF_HOME="${PWD}/hf_cache"
export HF_HUB_OFFLINE=1
# ./run.sh --stage 8 --stop_stage 10 --ngpu 1 \
#     --lm_config conf/tuning/train_transformer_opt350.yaml \
#     --inference_lm valid.acc.best.pth

##############################################################################
# 3) sLM21: sWUGGY + sBLIMP
#
# 脚本是从老 checkout /export/fs06/mliang17/espnet/egs2/voxtlm_v1/lm1 的
# v1(hubert)版本直接拿过来的,只改了两处:
#   - eval_slm21_speech.sh 的 --km_dir 换成本实验 Stage 3 用的那个官方模型
#     (/export/fs06/mliang17/shared_kmeans/hubert_base_l6_k1000_official),
#     不能用老 checkout 自己训的 kmeans,否则单元编号跟本 LM 学到的对不上。
#   - slm21_common.sh 里的 use_regular_gpu_partition 改成空操作:老 checkout
#     的 cmd.sh 默认 cuda_cmd 是 a100 所以要临时切,本 recipe 默认就是普通 gpu。
#
# 只跑 dev 不跑 test(论文 3.2 节就是报 dev)。
# 文本轨:gold.csv 里同一个词/句子会被不同说话人录多遍,文本打分跟音色无关,
#   所以去重后只算一次(20000/12566 条),再把分数展开回每个文件名。
# 语音轨:音频没法按内容去重,dev 的 80000+50400=130400 条全部要过一遍
#   特征提取+打标签,是两条轨里最重的一步。
#
# sLM21 的原始数据(30GB的zip和已解压的9GB dev音频)通过 downloads/sLM21
# 软链接复用老 checkout 的,解压脚本发现文件已存在会跳过,不会重复解压。
##############################################################################
#
# [已跑完,job 1801797(2026-09-21 09:54 ~ 10:29,约35分钟):
#    sWUGGY(lexical)   文本轨 82.61%  语音轨 61.43%   (各40000对)
#    sBLIMP(syntactic) 文本轨 68.41%  语音轨 54.35%   (各25200对)
#  没有跳过的对,submission 文件都在 exp/slm21_eval/submission/ 下。
#  这个 job 最后也是 FAILED,但不是 sLM21 的问题:它是接着往下跑 vocoder 那步,
#  在 parallel-wavegan-decode 上报 "command not found"(status 127)挂的,见下。]
# bash local/eval_slm21_text.sh
# bash local/eval_slm21_speech.sh
# bash local/eval_slm21_score.sh

##############################################################################
# 4) TTS 客观评分:hyp.tok -> 波形 -> VERSA 打分
#
# vocoder 用的是 hubert_voc1 那个 DiscreteSymbolHiFiGAN(LJSpeech 训的,
# 16kHz,hop=320 对应 50Hz 的离散单元)。关键是它训练时用的
# downloads/km_1000.mdl 跟本实验 Stage 3 打标签用的官方模型
# md5 完全一致(d46744528bf7f8dba295d560925fc1cf),所以两边的单元编号含义
# 相同,LM 生成的单元序列才能被它正确还原成语音。
#
# [第一次跑挂在这里(job 1801797):slurm.pl 生成的作业脚本开头会执行
#  ". ./path.sh",在计算节点上把 espnet 环境重新激活,盖掉了这里激活的
#  parallelwavegan,于是 parallel-wavegan-decode 在节点上 command not found
#  (status 127)。修法是在每个提交出去的命令里再激活一次 conda
#  (local/eval_tts_vocoder.sh 里的 CONDA_REACTIVATE 前缀),v2 那边早就踩过
#  同一个坑、已经修好了,这次把它那版的修改整份同步了过来:
#    - CONDA_REACTIVATE 包住 decode 和 versa 两个提交出去的命令
#    - versa 打分只用 1 个 A100(原来是切8份并行,会一下子占满配额)
#    - tok_to_pwg_feats.py 增加 --nclusters 过滤越界 token(这里传 1000)]
#
# 这一步必须用 parallelwavegan 那个 conda 环境(要 import parallel_wavegan
# 和 versa),不能用 path.sh 的 espnet 环境,所以放在子shell里单独激活,
# 不污染上面几步的环境。PATH 里补上 utils/ 是因为脚本内部要调 slurm.pl 提交
# GPU 任务,而那个脚本平时是靠 path.sh 加进 PATH 的。
##############################################################################
VOCODER_CKPT=/export/fs06/mliang17/phonologicaltokenizer_modify/ParallelWaveGAN/egs/ljspeech/hubert_voc1/exp/train_nodev_ljspeech_hifigan_hubert.v1/checkpoint-400000steps.pkl

(
    # 只 unset PYTHONPATH(espnet 的 path.sh 往里塞了 RawNet 的目录,跟
    # parallelwavegan 环境无关),千万不要 unset CONDA_PREFIX/CONDA_DEFAULT_ENV:
    # conda activate 在切换前会去读"上一个环境"的 deactivate.d 脚本,把
    # CONDA_PREFIX 清空会让它拼出空路径、scandir 报错,activate 直接失败,
    # python3 仍然是 espnet 那个,后面 import parallel_wavegan 必然失败。
    # (这个写法已在登录节点实测:切换后 python3 指向
    #  /home/mliang17/.conda/envs/parallelwavegan/bin/python3,
    #  parallel_wavegan / versa / h5py 都能 import,parallel-wavegan-decode 也在 PATH 里)
    unset PYTHONPATH
    export PATH="${PWD}/utils:${PATH}"
    # shellcheck disable=SC1091
    source /opt/slurm_files/software/conda25/2025-08-21/etc/profile.d/conda.sh
    conda activate parallelwavegan
    bash local/eval_tts_vocoder.sh "${VOCODER_CKPT}"
)

##############################################################################
echo "======================================================================"
echo "全部评测跑完。结果位置:"
echo "  PPL      : exp/lm_train_transformer_opt350_en_bpe10000/perplexity_test/*/ppl"
echo "  ASR WER  : exp/lm_train_transformer_opt350_en_bpe10000/decode_test_asr/*/score_wer/result.txt"
echo "  sLM21    : exp/slm21_eval/submission/ (准确率见本日志里 eval_slm21_score.sh 的输出)"
echo "  TTS 客观分: exp/lm_train_transformer_opt350_en_bpe10000/decode_test_tts/text.tts/vocoder_eval/*/versa_eval/avg_result.txt"
echo "======================================================================"
