#!/usr/bin/env bash
# Drop-in for lm.sh's ${python}: lm.sh runs
#     ${python} -m espnet2.bin.lm_inference <args>
# so passing --python local/python_norepeat.sh makes that one call go through
# local/lm_inference_norepeat.py (same args, same outputs, plus the optional
# NO_REPEAT_NGRAM ban); every other python invocation is passed straight through.
if [ "${1:-}" = "-m" ] && [ "${2:-}" = "espnet2.bin.lm_inference" ]; then
    shift 2
    exec python3 local/lm_inference_norepeat.py "$@"
fi
exec python3 "$@"
