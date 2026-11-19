#!/usr/bin/env bash
set -euo pipefail
source @ROOT@/env.sh
cd "$PAPER_ROOT"
exec "$PAPER_MAIN/bin/python" -u train_chain.py run "$1"
