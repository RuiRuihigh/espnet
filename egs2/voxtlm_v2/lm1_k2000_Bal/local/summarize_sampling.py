#!/usr/bin/env python3
"""Summarise myrun_tts_sampling.sh (greedy + top-p 0.9 x 3 seeds) for ONE recipe,
against the beam-5 baseline (text.tts.rpt_n0) on the SAME sentences.

Usage (from the recipe root): python3 local/summarize_sampling.py [maxlen=1200]

Reports, per configuration, on (a) the runaway sentences and (b) the control sentences
(20 longest non-runaway): how many generated sequences still hit the length cap, and
CER. Sampling is random, so top-p is also pooled over its seeds.
"""
import json
import sys
from pathlib import Path

MAXLEN = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
BASE = Path("exp/lm_train_transformer_opt350_en_bpe10000/decode_test_tts")
RUNAWAY = [l.strip() for l in open("local/tts_eval_subset/keys_norepeat_runaway.txt") if l.strip()]
ALL = [l.strip() for l in open("local/tts_eval_subset/keys_norepeat_test.txt") if l.strip()]
CTRL = [k for k in ALL if k not in set(RUNAWAY)]
CFGS = [("beam5 (baseline)", "rpt_n0"), ("greedy", "samp_greedy"),
        ("top-p 0.9 seed0", "samp_topp09_s0"), ("top-p 0.9 seed1", "samp_topp09_s1"),
        ("top-p 0.9 seed2", "samp_topp09_s2")]


def load(sub):
    d = BASE / f"text.tts.{sub}"
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


def errs(res, keys):
    e = t = 0
    for k in keys:
        r = res.get(k)
        if r:
            e += r["whisper_cer_delete"] + r["whisper_cer_insert"] + r["whisper_cer_replace"]
            t += r["whisper_cer_delete"] + r["whisper_cer_equal"] + r["whisper_cer_replace"]
    return e, t


data = {}
for label, sub in CFGS:
    try:
        data[label] = load(sub)
    except FileNotFoundError:
        print(f"[{label}] not finished yet")

def table(title, keys):
    print(f"\n=== {title} ({len(keys)} sentences) ===")
    print(f"{'config':<20}{'at cap (>=%d steps)' % (MAXLEN - 1):>20}{'median steps':>14}{'CER':>10}   errors/chars")
    pooled = {"cap": 0, "n": 0, "e": 0, "t": 0}
    for label, _ in CFGS:
        if label not in data:
            continue
        tok, res = data[label]
        steps = sorted(len(tok.get(k, [])) for k in keys)
        cap = sum(1 for s in steps if s >= MAXLEN - 1)
        e, t = errs(res, keys)
        print(f"{label:<20}{f'{cap}/{len(keys)}':>20}{steps[len(steps)//2]:>14}{e/t if t else float('nan'):>10.1%}   {e}/{t}")
        if label.startswith("top-p"):
            pooled["cap"] += cap; pooled["n"] += len(keys); pooled["e"] += e; pooled["t"] += t
    if pooled["n"]:
        cap_str = "%d/%d" % (pooled["cap"], pooled["n"])
        cer_pooled = pooled["e"] / pooled["t"]
        print(f"{'top-p pooled':<20}{cap_str:>20}{'':>14}{cer_pooled:>10.1%}   {pooled['e']}/{pooled['t']}")

table("runaway sentences", RUNAWAY)
table("control sentences", CTRL)

print("\n=== per-runaway detail (steps / errors) ===")
for k in RUNAWAY:
    print(f"\n{k}")
    for label, _ in CFGS:
        if label not in data:
            continue
        tok, res = data[label]
        r = res.get(k)
        if r:
            e = r["whisper_cer_delete"] + r["whisper_cer_insert"] + r["whisper_cer_replace"]
            t = r["whisper_cer_delete"] + r["whisper_cer_equal"] + r["whisper_cer_replace"]
            print(f"  {label:<18} steps={len(tok.get(k, [])):>5}  errors={e:>3}/{t}  {r['whisper_hyp_text'][:60]!r}")
