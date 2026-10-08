#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="test"

nbpe=10000
lm_config=conf/train.yaml
lm_inference_asr_config=conf/decode_lm_asr.yaml
lm_inference_tts_config=conf/decode_lm_tts.yaml

# Phonological Tokenizer (arXiv:2601.19781, https://huggingface.co/Sony/Phonological-Tokenizer):
# a fine-tuned WavLM-large + fixed 2000-entry codebook, used by lm.sh's own
# Stage 3 in place of a self-trained kmeans model (this recipe's lm.sh is a
# real file, not a symlink to TEMPLATE, so Stage 3 was edited directly).
# lm.sh's Stage 3 downloads/converts it itself, so nothing to do here but
# point it at the paths.
nclusters=2000
phonological_tokenizer_dir=downloads/phonological_tokenizer
km_dir="exp/kmeans/phonological_tokenizer_${nclusters}clusters"

# WavLM-large has no chunking in the SSL forward pass used by Stage 3a, and a
# handful of LibriLight segments run past 800s -- that alone saturates a
# shared GPU. Overriding the generic lm.sh default (20s, tuned for typical
# ASR-utterance-length assumptions) to 120s: a wide margin below the observed
# OOM point, while dropping only a small fraction of speechlm/train utterances.
max_wav_duration=120

./lm.sh \
    --stage 1 \
    --stop_stage 9 \
    --num_splits_lm 1 \
    --nj 16 \
    --ngpu 4 \
    --gpu_inference true \
    --inference_nj 8 \
    --lang en \
    --token_type bpe \
    --nbpe "${nbpe}" \
    --max_wav_duration "${max_wav_duration}" \
    --bpe_nlsyms data/nlsyms.txt \
    --bpe_train_text "data/${train_set}/bpe_text" \
    --lm_config "${lm_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --inference_lm valid.acc.ave.pth \
    --km_dir "${km_dir}" \
    --nclusters "${nclusters}" \
    --phonological_tokenizer_ssl_path "${phonological_tokenizer_dir}/ssl.pth" \
    --phonological_tokenizer_layer 21 \
    --lm_inference_asr_config "${lm_inference_asr_config}" \
    --lm_inference_tts_config "${lm_inference_tts_config}" \
    --lm_test_text_asr dump/raw/${test_sets}/text.asr \
    --lm_test_text_tts dump/raw/${test_sets}/text.tts \
    --lm_test_text_textlm dump/raw/${test_sets}/text.textlm \
    --lm_test_text_speechlm dump/raw/${test_sets}/text.speechlm "$@"
