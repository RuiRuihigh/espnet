# TTS 随机 100 句测试集(不限长度)+ 放开解码 maxlen

## 为什么有这一版
之前 `keys.txt` 那 100 句只挑了真值单元长度 <200 的短句,规避了"长句子被 maxlen=300
截断"的问题,但也因此看不出模型在长句上的真实表现。后来查明:全量测试集里 CER 高的
句子几乎都是撞满了 `decode_lm_tts.yaml` 里 `maxlen: 300`(BPE 步数上限,不是原始单元数)
被强制结束的——不同 tokenizer 的 BPE 压缩率不同(k1000 约 2.0 单元/步,音系学
tokenizer 约 1.7),同一个上限对应的语音时长不同(约 12 秒 vs 10 秒),所以 v2 被截得
更多。这一版:句子不限长度、maxlen 放开到能盖住最长的测试句。

## 选取方法
- `keys_random100.txt`:从全部 9957 条测试句里均匀随机抽 100 条,种子 1234
  (`select_random_subset.py`,可复现),跟上一版 keys.txt 只重叠 2 条。
- 真值单元长度:中位 218、P90 555、最长 857;>=400 的 22 条、>=600 的 8 条(这 8 条在
  旧的 maxlen=300 下都会被截断)。涉及 43 个说话人。
- 各 recipe 的 test tts utt id 集合、真值长度完全一致,所以四个 recipe 用同一份 keys。

## maxlen 怎么定的
用各 recipe 自己的 bpe.model 对全部 9957 条 tts 真值单元序列重新切分,数出"要完整
复现真值长度需要多少个 BPE 步"(=解码需要的 maxlen):

| recipe | 需要的 BPE 步:最长 | 每步平均单元数 |
|---|---|---|
| v1 lm1_k1000_Bal | 788 | 2.17 |
| v1 lm1_k2000_Bal | 977 | 1.78 |
| v1 lm1_k2000_Bal_ttskmeans | 966 | 1.78 |
| v2 lm1_k2000_Bal | 947 | 1.86 |

统一取 `maxlen: 1200`(约 23% 余量,给"生成比真值略长"留空间)。提示词+生成的总长最多
约 1385,低于训练时 >1500 被过滤的线(Stage 6)和 OPT 的 2048 位置上限,不会越界。
配置文件是 `conf/decode_lm_tts_maxlen1200.yaml`,其它参数(beam 5、penalty 0)跟
`conf/decode_lm_tts.yaml` 完全一致,只有 maxlen 不同;原来的配置不动,以免影响已有结果。

## 输出位置(不覆盖已有结果)
`exp/lm_train_.../decode_test_tts/text.tts.rand100/`(lm.sh 用输入文件名给目录命名),
vocoder 合成和 VERSA 打分在其下的 `vocoder_eval/`。
