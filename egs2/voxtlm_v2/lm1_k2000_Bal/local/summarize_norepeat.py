#!/usr/bin/env python3
"""Summarise the no-repeat-ngram experiment (myrun_tts_norepeat.sh) for ONE recipe.

Usage (from the recipe root, espnet env or any python3):
    python3 local/summarize_norepeat.py [maxlen=1200]

Compares n=0 (baseline), n=6, n=8 on: (a) the runaway sentences (keys_norepeat_runaway.txt),
(b) the 20 long non-runaway control sentences. Prints how many still hit the cap, CER, and
-- for the controls -- how many generated sequences changed vs n=0 (a penalty that fixes
runaways but rewrites normal sentences would be useless).
"""
import json
import sys
from pathlib import Path

MAXLEN = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
BASE = Path("exp/lm_train_transformer_opt350_en_bpe10000/decode_test_tts")
RUNAWAY = [l.strip() for l in open("local/tts_eval_subset/keys_norepeat_runaway.txt") if l.strip()]
ALL = [l.strip() for l in open("local/tts_eval_subset/keys_norepeat_test.txt") if l.strip()]
CTRL = [k for k in ALL if k not in set(RUNAWAY)]
NS = [0, 6, 8]


def load(n):
    d = BASE / f"text.tts.rpt_n{n}"
    tok = {}
    for line in open(d / "token"):
        p = line.split()
        t = p[1:]
        if "<generatespeech>" in t:
            tok[p[0].replace("tts_", "", 1)] = t[t.index("<generatespeech>") + 1 :]
    res = {}
    f = d / "vocoder_eval/checkpoint-400000steps/versa_eval/result.1.txt"
    if f.exists():
        for line in open(f):
            r = json.loads(line)
            res[r["key"]] = r
    return tok, res


def cer(res, keys):
    e = t = 0
    for k in keys:
        r = res.get(k)
        if r is None:
            continue
        e += r["whisper_cer_delete"] + r["whisper_cer_insert"] + r["whisper_cer_replace"]
        t += r["whisper_cer_delete"] + r["whisper_cer_equal"] + r["whisper_cer_replace"]
    return (e / t if t else float("nan")), e, t


data = {}
for n in NS:
    try:
        data[n] = load(n)
    except FileNotFoundError:
        print(f"[n={n}] not finished yet")
if 0 not in data:
    sys.exit("baseline n=0 missing")

# --- sanity: does the n=0 wrapper run reproduce the earlier rand100 decode? ---
prev = {}
p = BASE / "text.tts.rand100/token"
if p.exists():
    for line in open(p):
        q = line.split()
        t = q[1:]
        if "<generatespeech>" in t:
            prev[q[0].replace("tts_", "", 1)] = t[t.index("<generatespeech>") + 1 :]
same = sum(1 for k in ALL if k in prev and prev[k] == data[0][0].get(k))
print(f"[sanity] n=0 (wrapper, no ban) vs earlier rand100 decode: identical token sequences for {same}/{len(ALL)} sentences")

print(f"\n=== runaway sentences ({len(RUNAWAY)}) ===")
print(f"{'n':<4}{'still at cap':>13}{'gen steps (per sentence)':>52}{'CER':>10}")
for n in NS:
    if n not in data:
        continue
    tok, res = data[n]
    steps = [len(tok.get(k, [])) for k in RUNAWAY]
    capped = sum(1 for s in steps if s >= MAXLEN - 1)
    c, e, t = cer(res, RUNAWAY)
    print(f"{n:<4}{capped:>13}{str(steps):>52}{c:>10.1%}   ({e}/{t} chars)")

print(f"\n=== control sentences ({len(CTRL)} longest non-runaway) ===")
print(f"{'n':<4}{'still at cap':>13}{'sequences changed vs n=0':>26}{'CER':>10}")
for n in NS:
    if n not in data:
        continue
    tok, res = data[n]
    capped = sum(1 for k in CTRL if len(tok.get(k, [])) >= MAXLEN - 1)
    changed = sum(1 for k in CTRL if tok.get(k) != data[0][0].get(k))
    c, e, t = cer(res, CTRL)
    print(f"{n:<4}{capped:>13}{f'{changed}/{len(CTRL)}':>26}{c:>10.2%}   ({e}/{t} chars)")

print("\n=== per-runaway detail ===")
for k in RUNAWAY:
    print(f"\n{k}")
    for n in NS:
        if n not in data:
            continue
        tok, res = data[n]
        r = res.get(k)
        txt = r["whisper_hyp_text"][:80] if r else "(not scored)"
        e = (r["whisper_cer_delete"] + r["whisper_cer_insert"] + r["whisper_cer_replace"]) if r else -1
        t = (r["whisper_cer_delete"] + r["whisper_cer_equal"] + r["whisper_cer_replace"]) if r else -1
        print(f"  n={n}: steps={len(tok.get(k, [])):>5}  errors={e}/{t}  whisper: {txt!r}")
    if r:
        print(f"  ref  : {r['ref_text'][:80]!r}")
