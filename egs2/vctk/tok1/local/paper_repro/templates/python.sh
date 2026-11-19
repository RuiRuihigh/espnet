#!/usr/bin/env bash
set -euo pipefail
P=@BASE@/software/envs/phonological/bin/python
R=@ROOT@
if [ "${1:-}" = -m ] && [ "${2:-}" = espnet2.bin.gan_tok_train ]; then
  shift 2
  exec "$P" "$R/train_entry.py" "$@"
else
  exec "$P" "$@"
fi
