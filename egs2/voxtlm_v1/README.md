# VoxtLM v1

[VoxtLM](https://arxiv.org/abs/2309.07937)-style decoder-only speech–text LMs trained with ESPnet. Each recipe trains one OPT-backbone model that does four tasks via special tokens: ASR (speech → text), TTS (text → speech units), text continuation and speech-unit continuation.

"v1" uses the paper's original speech tokenizer: **HuBERT-base, layer 6**, with a k-means codebook fit on this project's own data (either on the full speechlm training pool, or — for the `_ttskmeans` recipes — on the TTS training subset only). A separate line of `_wavlm` recipes swaps in **WavLM-large, layer 21** as the feature extractor instead.

Published checkpoints (model weights, k-means codebook, BPE model) are on Hugging Face; this GitHub branch has the training/eval scripts and configs only (no weights, no data).

| Recipe | Speech tokenizer | Backbone | Data scale | Hugging Face |
|---|---|---|---|---|
| [lm1_k1000_Bal](lm1_k1000_Bal) | HuBERT-base L6, k=1000 (full speechlm pool) | OPT-350M | D_Bal | — |
| [lm1_k1000_Bal_opt1.3B](lm1_k1000_Bal_opt1.3B) | same | OPT-1.3B | D_Bal | — |
| [lm1_k1000_Bal_sqa](lm1_k1000_Bal_sqa) | same | OPT-350M, [LibriSQA](https://arxiv.org/abs/2308.10390)-finetuned from lm1_k1000_Bal | D_Bal + SQA | — |
| [lm1_k1000_3M](lm1_k1000_3M) | same | OPT-350M | D_3M | — |
| [lm1_k1000_Bal_ttskmeans_wavlm](lm1_k1000_Bal_ttskmeans_wavlm) | WavLM-large L21, k=1000 (TTS subset only) | OPT-350M | D_Bal | [RuiRuihigh/voxtlm_v1_k1000_Bal_ttskmeans_wavlm](https://huggingface.co/RuiRuihigh/voxtlm_v1_k1000_Bal_ttskmeans_wavlm) |
| [lm1_k2000_Bal](lm1_k2000_Bal) | HuBERT-base L6, k=2000 (full speechlm pool) | OPT-350M | D_Bal | — |
| [lm1_k2000_Bal_ttskmeans](lm1_k2000_Bal_ttskmeans) | HuBERT-base L6, k=2000 (TTS subset only) | OPT-350M | D_Bal | — |
| [lm1_k2000_Bal_ttskmeans_opt1.3B](lm1_k2000_Bal_ttskmeans_opt1.3B) | same | OPT-1.3B | D_Bal | — |
| [lm1_k2000_Bal_ttskmeans_sqa](lm1_k2000_Bal_ttskmeans_sqa) | same | OPT-350M, LibriSQA-finetuned | D_Bal + SQA | — |
| [lm1_k2000_Bal_ttskmeans_wavlm](lm1_k2000_Bal_ttskmeans_wavlm) | WavLM-large L21, k=2000 (TTS subset only) | OPT-350M | D_Bal | [RuiRuihigh/voxtlm_v1_k2000_Bal_ttskmeans_wavlm](https://huggingface.co/RuiRuihigh/voxtlm_v1_k2000_Bal_ttskmeans_wavlm) |
| [lm1_k2000_3M_ttskmeans](lm1_k2000_3M_ttskmeans) | HuBERT-base L6, k=2000 (TTS subset only) | OPT-350M | D_3M | — |

`D_Bal` / `D_3M` follow the paper's data-size presets (Table 2): only the speechlm/textlm training splits scale with the preset; ASR/TTS training data and all dev/test splits are full-size in every preset.

`lm1`, `lm1_3m` and `lm1_k1000_Bal_ttskmeans` are earlier scaffolding that only got as far as fitting a k-means codebook and were superseded by the recipes above; they were never trained to a usable checkpoint.

## Usage

```bash
huggingface-cli download RuiRuihigh/<repo> --local-dir .
```

inside the matching recipe directory, then follow that recipe's `myrun_eval.sh` / the model card's `## Usage` section for the exact decode command.
