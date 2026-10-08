#!/usr/bin/env python3
"""Report generated-step counts for a TTS decode dir, highlighting a fixed set
of known-short (premature-EOS) utterances. No vocoder/VERSA involved -- just
reads <decode_dir>/token (same quantity local/fallback_select.py's
`capped` subcommand uses), so this is cheap to run after every penalty value's
decode.

Usage: python3 local/penalty_sweep_report.py <decode_dir> <maxlen> <key1> [<key2> ...]
  keys are WITHOUT the "tts_" prefix (same convention as fallback_select.py).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fallback_select import steps_of  # noqa: E402


def main():
    d, maxlen, keys = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    steps = steps_of(d)
    n = len(steps)
    capped = [k for k, v in steps.items() if v >= maxlen - 1]
    vals = sorted(steps.values())
    print(f"{d}: n={n}  median_steps={vals[n // 2]}  max_steps={vals[-1]}  capped(>= {maxlen - 1})={len(capped)}")
    print("known-short keys (steps at this penalty):")
    for k in keys:
        print(f"  {k}\t{steps.get(k, 'MISSING')}")


if __name__ == "__main__":
    main()
