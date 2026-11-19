#!/usr/bin/env bash
# Run on an allocated CPU node, after init_workspace.py. Does not submit jobs.
set -euo pipefail
test "$#" -eq 1 || { echo "Usage: $0 /absolute/experiment/root" >&2; exit 2; }
source "$1/env.sh"
package=$(cd "$(dirname "$0")" && pwd)
conda_sh=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["settings"]["conda_sh"])' "$PAPER_ROOT/PORTING.json")
source "$conda_sh"
export CONDA_PKGS_DIRS="$PAPER_BASE/cache/conda-paper"
export PIP_CACHE_DIR="$PAPER_BASE/cache/pip"
mkdir -p "$PAPER_BASE/software/envs" "$PAPER_BASE/assets" "$HF_HOME"
for target in "$PAPER_MAIN" "$PAPER_AUDIO"; do
    if [ -x "$target/bin/python" ]; then
        echo "Refusing to overwrite existing environment $target; inspect it manually." >&2
        exit 2
    fi
done
conda create -y -p "$PAPER_MAIN" -c conda-forge python=3.12 pip libsndfile
conda create -y -p "$PAPER_AUDIO" -c conda-forge python=3.12 pip libsndfile
conda create -y -p "$PAPER_BASE/software/envs/paper-tools" -c conda-forge sox bc
for target in "$PAPER_MAIN" "$PAPER_AUDIO"; do
    "$target/bin/python" -m pip install 'setuptools<74' wheel
done
"$PAPER_MAIN/bin/python" -m pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu126
"$PAPER_MAIN/bin/python" -m pip install numpy==2.5.4
"$PAPER_MAIN/bin/python" -m pip install --no-build-isolation -r "$package/environment/main.requirements.txt" --extra-index-url https://download.pytorch.org/whl/cu126
"$PAPER_MAIN/bin/python" -m pip install --no-deps --no-build-isolation -e "$PAPER_ROOT/espnet"
"$PAPER_AUDIO/bin/python" -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
"$PAPER_AUDIO/bin/python" -m pip install -r "$package/environment/audio.requirements.txt" --extra-index-url https://download.pytorch.org/whl/cu124
"$PAPER_MAIN/bin/python" -m pip freeze > "$PAPER_ROOT/main.freeze.txt"
"$PAPER_AUDIO/bin/python" -m pip freeze > "$PAPER_ROOT/audio.freeze.txt"
"$PAPER_MAIN/bin/python" -m pip check
"$PAPER_AUDIO/bin/python" -m pip check
echo 'Environments installed. Download WavLM, run setup.sh, and validate CPU/audio imports before submission.'
