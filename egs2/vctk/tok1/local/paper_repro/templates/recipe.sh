#!/usr/bin/env bash
set -euo pipefail
source @ROOT@/env.sh
cd "$PAPER_ROOT/espnet/egs2/vctk/tok1"
exec ./run.sh --ngpu 4 --nj 4 --inference_nj 4 \
  --python "$PAPER_MAIN/bin/python" \
  --use_default_centroid false --tok_config conf/train_paper.yaml \
  --tok_exp "$PAPER_ROOT/train" --tok_stats_dir "$PAPER_ROOT/stats" \
  --dumpdir "$PAPER_ROOT/dump" --expdir "$PAPER_ROOT/exp" \
  --spk_embed_tool speechbrain --spk_embed_model speechbrain/spkrec-ecapa-voxceleb \
  --spk_embed_tag ecapa192 --spk_embed_parallel false "$@"
