#!/usr/bin/env python3
"""After a TTS decode, report how many utterances hit the beam-search length
cap (generated BPE steps >= maxlen-1). If this is >0 with the raised maxlen,
the cap is still binding for those utterances.

Usage: python3 local/count_capped_steps.py <decode_dir> <maxlen>
  e.g. exp/lm_train_.../decode_test_tts/text.tts.rand100 1200
Counts BPE tokens after <generatespeech> in <decode_dir>/token (the same
quantity maxlen bounds; the beam-search warning is NOT a reliable detector
because it compares prompt+generation length, not generated steps).
"""
import sys

d, maxlen = sys.argv[1], int(sys.argv[2])
steps = {}
for line in open(f"{d}/token"):
    p = line.split()
    if "<generatespeech>" in p[1:]:
        t = p[1:]
        steps[p[0]] = len(t) - t.index("<generatespeech>") - 1
n = len(steps)
capped = [k for k, v in steps.items() if v >= maxlen - 1]
s = sorted(steps.values())
print(f"utts={n}  generated BPE steps: median={s[n//2]} max={s[-1]}  (cap={maxlen})")
print(f"utts at the cap (steps>={maxlen-1}): {len(capped)}")
for k in capped:
    print("  CAPPED:", k, steps[k])
