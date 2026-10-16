# VoxtLM v2

[VoxtLM](https://arxiv.org/abs/2309.07937)-style decoder-only speech–text LMs trained with ESPnet. Each recipe trains one OPT-backbone model that does four tasks via special tokens: ASR (speech → text), TTS (text → speech units), text continuation and speech-unit continuation.

"v2" replaces the paper's original HuBERT k-means units (see [voxtlm_v1](../voxtlm_v1)) with the **[Phonological Tokenizer](https://huggingface.co/Sony/Phonological-Tokenizer)** (arXiv:2601.19781): a fine-tuned WavLM-large, layer 21, with a 2000-entry k-means codebook, fit on the full speechlm training pool.

Published checkpoints (model weights, k-means codebook, BPE model) are on Hugging Face; this GitHub branch has the training/eval scripts and configs only (no weights, no data).

| Recipe | Backbone | Data scale | Hugging Face |
|---|---|---|---|
| [lm1_k2000_Bal](lm1_k2000_Bal) | OPT-350M | D_Bal | [RuiRuihigh/voxtlm_v2_k2000_Bal_opt350m](https://huggingface.co/RuiRuihigh/voxtlm_v2_k2000_Bal_opt350m) |
| [lm1_k2000_Bal_opt1.3B](lm1_k2000_Bal_opt1.3B) | OPT-1.3B | D_Bal | [RuiRuihigh/voxtlm_v2_k2000_Bal_opt1.3b](https://huggingface.co/RuiRuihigh/voxtlm_v2_k2000_Bal_opt1.3b) |
| [lm1_k2000_Bal_sqa](lm1_k2000_Bal_sqa) | OPT-350M, [LibriSQA](https://arxiv.org/abs/2308.10390)-finetuned from lm1_k2000_Bal | D_Bal + SQA | [RuiRuihigh/voxtlm_v2_k2000_Bal_opt350m_sqa](https://huggingface.co/RuiRuihigh/voxtlm_v2_k2000_Bal_opt350m_sqa) |
| [lm1_k2000_Bal_opt1.3B_sqa](lm1_k2000_Bal_opt1.3B_sqa) | OPT-1.3B, LibriSQA-finetuned from lm1_k2000_Bal_opt1.3B | D_Bal + SQA | — |
| [lm1_k2000_3M](lm1_k2000_3M) | OPT-350M | D_3M | — |

`D_Bal` / `D_3M` follow the paper's data-size presets (Table 2): only the speechlm/textlm training splits scale with the preset; ASR/TTS training data and all dev/test splits are full-size in every preset.

## Usage

```bash
huggingface-cli download RuiRuihigh/<repo> --local-dir .
```

inside the matching recipe directory, then follow that recipe's `myrun_eval.sh` / the model card's `## Usage` section for the exact decode command.
