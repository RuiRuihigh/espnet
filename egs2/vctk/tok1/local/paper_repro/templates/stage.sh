#!/usr/bin/env bash
set -euo pipefail
source @ROOT@/env.sh
cd "$PAPER_ROOT"
stage=$1
test ! -f STOPPED_BY_USER.json
mkdir -p stages
if [ -f "stages/$stage.done" ]; then echo "REUSE_COMPLETE $stage"; exit 0; fi
printf 'STAGE_START %s job=%s date=%s\n' "$stage" "${SLURM_JOB_ID:-none}" "$(date -Is)"
case "$stage" in
 data)
   test -f SETUP_OK.json
   for n in 1 2 3 4 5; do
     if [ ! -f "stages/data_$n.done" ]; then
       ./recipe.sh --stage "$n" --stop_stage "$n"
       touch "stages/data_$n.done"
     fi
   done
   "$PAPER_MAIN/bin/python" -u data_audit.py
   ;;
 subset)
   test -f SETUP_OK.json
   "$PAPER_MAIN/bin/python" -u libri_subset.py
   ;;
 features|kmeans)
   test -f libri_data/libri30h/COMPLETE.json
   cd "$PAPER_ROOT/espnet/egs2/vctk/tok1"
   if [ "$stage" = features ]; then n=1; gpu=true; nj=1; else n=2; gpu=false; nj=1; fi
   ./scripts/feats/perform_kmeans.sh --stage "$n" --stop_stage "$n" \
     --train_set libri30h --dev_set '' --other_sets '' \
     --datadir "$PAPER_ROOT/libri_data" --featdir "$PAPER_ROOT/libri_features" \
     --km_dir "$PAPER_ROOT/initialization" --nclusters 2000 --RVQ_layers 1 \
     --feature_type wavlm_large --layer 21 \
     --feature_conf "{type=s3prl,conf={s3prl_conf={upstream=wavlm_local,path_or_url=$PAPER_BASE/assets/wavlm_large.pt},download_dir=$PAPER_ROOT/hub,multilayer_feature=False,layer=21}}" \
     --audio_format flac --audio_sample_rate 16000 --portion 1.0 --storage_save_mode false \
     --use_gpu "$gpu" --nj "$nj" --batch_bins 320000 \
     --num_threads "${SLURM_CPUS_PER_TASK:-2}" --python "$PAPER_MAIN/bin/python" \
     --cpu_cmd run.pl --cuda_cmd run.pl
   ;;
 speakers)
   test -f DATA_READY.json
   # Separate compatible SpeechBrain environment; same pretrained encoder.
   ./recipe.sh --stage 6 --stop_stage 6 --python "$PAPER_AUDIO/bin/python"
   ;;
 stats)
   test -f DATA_READY.json; test -f initialization/km_2000.mdl
   ./recipe.sh --stage 8 --stop_stage 8 --nj 2
   ;;
 preflight)
   "$PAPER_MAIN/bin/python" -u preflight.py
   ;;
 *) echo "Unknown stage $stage" >&2; exit 2 ;;
esac
cd "$PAPER_ROOT"
date -Is > "stages/$stage.done"
printf 'STAGE_COMPLETE %s\n' "$stage"
