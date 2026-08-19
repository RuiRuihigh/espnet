# Phonological Tokenizer

Recipe scaffold for training a prosody-aware phonetic speech tokenizer, following:

> Onda, Futami, Kashiwagi, Tsunoo, Watanabe. "Phonological Tokenizer: Prosody-Aware
> Phonetic Token via Multi-Objective Fine-Tuning with Differentiable K-Means."
> https://arxiv.org/pdf/2601.19781

The pretrained checkpoint released by the authors (`Sony/Phonological-Tokenizer` on
HuggingFace) is a WavLM-large encoder fine-tuned with a differentiable k-means
codebook (2000 entries, layer-21 hidden states). This recipe is for training that
kind of tokenizer from scratch instead of only consuming the released checkpoint
(see `egs2/voxtlm_v2/lm1` on `origin/voxtlm_dev` for the download/usage-side
integration).

## Status

Scaffold only — created via `egs2/TEMPLATE/ssl1/setup.sh`. No training code yet.

## Planned components

- **Encoder**: WavLM-large loaded via `S3prlFrontend`
  (`espnet2/asr/frontend/s3prl.py`), unfrozen, layer-21 output selected.
- **Quantizer**: differentiable/EMA vector-quantized codebook, reusing
  `VectorQuantization` / `EuclideanCodebook` from
  `espnet2/gan_codec/shared/quantizer/modules/core_vq.py` (kmeans-initialized,
  EMA-updated, straight-through estimator).
- **Losses (multi-objective)**:
  - phonetic prediction loss (CTC/CE against phoneme alignments)
  - prosody prediction loss (F0 / energy / duration regression)
  - VQ commitment loss (from the quantizer module above)
- **Task wiring**: new `espnet2/tasks/*.py` task class registering the above
  as frontend/quantizer/loss so training reuses the standard ESPnet2
  `Trainer` (`espnet2/bin/*_train.py`) instead of a bespoke training loop.
- **Data prep** (`local/`): TBD — needs a corpus with phoneme alignments and
  F0/energy annotations (e.g. LibriTTS/VCTK with forced alignment).
- **Inference/usage** (`scripts/` or new `pyscripts/`): forward pass through
  the fine-tuned encoder + quantizer to emit discrete token indices directly
  (no separate offline k-means step needed, unlike the classic HuBERT
  `learn_kmeans.py` / `perform_kmeans.sh` pipeline in `egs2/TEMPLATE/asr1`).

## Layout

Symlinked from `egs2/TEMPLATE/ssl1` and `egs2/TEMPLATE/asr1` (standard ESPnet2
recipe convention): `ssl.sh`, `path.sh`, `scripts`, `db.sh`, `pyscripts`,
`steps`, `utils`. Copied (recipe-local, editable): `cmd.sh`, `conf/`, `local/`.
