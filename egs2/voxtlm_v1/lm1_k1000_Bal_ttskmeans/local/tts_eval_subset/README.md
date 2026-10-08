# TTS 100 句子短测试子集

用于跟三个 recipe(v1 lm1_k1000_Bal / v1 lm1_k2000_Bal / v2 lm1_k2000_Bal)
比较 TTS 客观分,不受"长句子在自由生成时提前收尾"这个已知问题拖累。

## 选取方法
- 三个 recipe 的 `dump/raw/test/speech/tts/token`(离散单元真值序列,50Hz)
  utt id 集合完全一致(已验证),且同一条 utt 在三边的真值长度也完全一致
  (帧率相同,跟用哪种 tokenizer 无关)。
- 以 k1000 的 `dump/raw/test/speech/tts/token` 为基准,筛出真值长度 < 200
  (候选池 4857/9957 条),固定随机种子 42,抽 100 条,按 utt id 排序。
- 长度范围 17~199,中位数 96,离 300(解码 maxlen)和历史观察到的真实截断阈值
  都还有很大余量,基本不会有"没生成完就被打断"的风险。
- 涉及 43 个不同说话人,不是单一说话人/单一书目的偏样本。

## 用法
`keys.txt` 是 utt id 列表,一行一个。给 TTS 解码/vocoder/VERSA 打分用
`--key_file` 或 `utils/filter_scp.pl keys.txt <文件>` 从全量结果里筛出这
100 条,或者直接对这 100 条重新跑 Stage 10(比全量 9957 条快得多)。

生成命令(供复现):见本目录 keys.txt 旁的 select_subset.py(如果需要重新生成)。
