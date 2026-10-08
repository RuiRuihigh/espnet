#!/usr/bin/env python3
"""Final numbers of the two-level fallback run (myrun_tts_fallback.sh) for ONE recipe.

Usage (from the recipe root): python3 local/summarize_fallback.py
Compares against the plain beam-5 result of the same 100 sentences (text.tts.rand100).
"""
import json
import statistics as st
from pathlib import Path

BASE = Path("exp/lm_train_transformer_opt350_en_bpe10000/decode_test_tts")
VOC = "vocoder_eval/checkpoint-400000steps/versa_eval/result.1.txt"


def load(sub):
    res = {}
    for line in open(BASE / f"text.tts.{sub}" / VOC):
        x = json.loads(line)
        res[x["key"]] = x
    return res


def cer(x):
    return (x["whisper_cer_delete"] + x["whisper_cer_insert"] + x["whisper_cer_replace"],
            x["whisper_cer_delete"] + x["whisper_cer_equal"] + x["whisper_cer_replace"])


def wer(x):
    return (x["whisper_wer_delete"] + x["whisper_wer_insert"] + x["whisper_wer_replace"],
            x["whisper_wer_delete"] + x["whisper_wer_equal"] + x["whisper_wer_replace"])


def summary(res):
    ce = sum(cer(x)[0] for x in res.values()); ct = sum(cer(x)[1] for x in res.values())
    we = sum(wer(x)[0] for x in res.values()); wt = sum(wer(x)[1] for x in res.values())
    per = [cer(x)[0] / cer(x)[1] for x in res.values() if cer(x)[1]]
    return ce / ct, st.mean(per), we / wt, st.mean(x["utmos"] for x in res.values()), st.mean(x["speech_bert"] for x in res.values())


rows = [l.rstrip("\n").split("\t") for l in open(BASE / "text.tts.fb_final/fallback_log.tsv")]
hdr, body = rows[0], rows[1:]
used = {}
for r in body:
    used[r[1]] = used.get(r[1], 0) + 1
plain, final = load("rand100"), load("fb_final")
print(f"{'':<26}{'CER':>8}{'CER(逐句均)':>12}{'WER':>8}{'UTMOS':>7}{'SpeechBERT':>11}")
for name, res in (("plain beam-5", plain), ("beam -> greedy -> top-p", final)):
    c, cp, w, u, s = summary(res)
    print(f"{name:<26}{c:>8.2%}{cp:>12.2%}{w:>8.2%}{u:>7.2f}{s:>11.3f}")
print(f"\nlevel used per sentence: {used}   still at the cap after all levels: {sum(int(r[-1]) for r in body)}")
for r in body:
    if r[1] != "beam5":
        k = r[0]
        e0, t0 = cer(plain[k]); e1, t1 = cer(final[k])
        print(f"  fell back: {k}  -> {r[1]:<10} steps {dict(zip(hdr[2:-1], r[2:-1]))}  errors {e0}/{t0} -> {e1}/{t1}")
