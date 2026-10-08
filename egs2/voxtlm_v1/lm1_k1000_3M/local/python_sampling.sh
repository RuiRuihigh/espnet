#!/usr/bin/env bash
# Drop-in for lm.sh's ${python}, same idea as local/python_norepeat.sh:
# `${python} -m espnet2.bin.lm_inference <args>` is routed through
# local/lm_inference_sampling.py (top-p sampling when SAMPLE_MODE=topp, otherwise stock);
# every other python invocation passes straight through.
if [ "${1:-}" = "-m" ] && [ "${2:-}" = "espnet2.bin.lm_inference" ]; then
    shift 2
    exec python3 local/lm_inference_sampling.py "$@"
fi
exec python3 "$@"
