#!/usr/bin/env bash
set -euo pipefail
source @ROOT@/env.sh
cd "$PAPER_ROOT"
printf 'SETUP_START %s\n' "$(date -Is)"
source @CONDA_SH@
export CONDA_PKGS_DIRS=$PAPER_BASE/cache/conda-paper-tools
if [ ! -x "$PAPER_BASE/software/envs/paper-tools/bin/sox" ]; then
  conda create -y -p "$PAPER_BASE/software/envs/paper-tools" -c conda-forge sox bc
fi
source "$PAPER_ROOT/env.sh"
sox --version
"$PAPER_MAIN/bin/python" -u "$PAPER_ROOT/configure.py"
"$PAPER_MAIN/bin/python" -u "$PAPER_ROOT/runtime_audit.py"
