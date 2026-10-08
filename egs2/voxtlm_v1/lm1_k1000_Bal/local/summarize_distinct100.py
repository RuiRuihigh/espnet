#!/usr/bin/env python3
"""Final TTS comparison on 100 DISTINCT sentences for the four recipes.

Set = keys_random100.txt minus the 7 sentences that ran away (v2: 6, ttskmeans: 1;
they are keys_norepeat_runaway.txt of those two recipes) plus keys_new7.txt (7 fresh
random sentences). Everything was decoded with beam 5, maxlen 1200, valid.acc.best.pth.
Results come from text.tts.rand100 (93 of them) and text.tts.new7 (the 7 new ones).

Run from any python3 (paths are absolute). Prints corpus-level CER, per-sentence mean CER,
WER, UTMOS, SpeechBERT, and how many sentences hit the 1200-step cap.
"""
import json
import re
import statistics as st

E = "/export/fs06/mliang17/voxtlm_officialtest/espnet/egs2"
RECS = [
    ("v1 k1000", "voxtlm_v1/lm1_k1000_Bal"),
    ("v1 k2000", "voxtlm_v1/lm1_k2000_Bal"),
    ("v1 k2000 ttskm", "voxtlm_v1/lm1_k2000_Bal_ttskmeans"),
    ("v2 k2000", "voxtlm_v2/lm1_k2000_Bal"),
]
TS = "local/tts_eval_subset"
rand = [l.strip() for l in open(f"{E}/voxtlm_v1/lm1_k1000_Bal/{TS}/keys_random100.txt") if l.strip()]
new7 = [l.strip() for l in open(f"{E}/voxtlm_v1/lm1_k1000_Bal/{TS}/keys_new7.txt") if l.strip()]
bad = set()
for r in ("voxtlm_v2/lm1_k2000_Bal", "voxtlm_v1/lm1_k2000_Bal_ttskmeans"):
    bad |= {l.strip() for l in open(f"{E}/{r}/{TS}/keys_norepeat_runaway.txt") if l.strip()}
assert len(bad) == 7 and bad <= set(rand) and not (set(new7) & set(rand))
final = [k for k in rand if k not in bad] + new7
assert len(final) == 100 and len(set(final)) == 100


def load(r, sub):
    base = f"{E}/{r}/exp/lm_train_transformer_opt350_en_bpe10000/decode_test_tts/text.tts.{sub}"
    res = {}
    for line in open(f"{base}/vocoder_eval/checkpoint-400000steps/versa_eval/result.1.txt"):
        x = json.loads(line)
        res[x["key"]] = x
    steps = {}
    for line in open(f"{base}/token"):
        p = line.split()
        t = p[1:]
        if "<generatespeech>" in t:
            steps[p[0].replace("tts_", "", 1)] = len(t) - t.index("<generatespeech>") - 1
    return res, steps


def err(x):
    e = x["whisper_cer_delete"] + x["whisper_cer_insert"] + x["whisper_cer_replace"]
    t = x["whisper_cer_delete"] + x["whisper_cer_equal"] + x["whisper_cer_replace"]
    return e, t


def werr(x):
    e = x["whisper_wer_delete"] + x["whisper_wer_insert"] + x["whisper_wer_replace"]
    t = x["whisper_wer_delete"] + x["whisper_wer_equal"] + x["whisper_wer_replace"]
    return e, t


print(f"final set: {len(final)} distinct sentences = {len(rand) - len(bad)} from rand100 + {len(new7)} new")
print(f"\n{'recipe':<16}{'CER':>8}{'CER(逐句均)':>12}{'WER':>8}{'UTMOS':>7}{'SpeechBERT':>11}{'撞满1200':>9}   {'其中93条':>9}{'新7条':>8}")
for name, r in RECS:
    a, sa = load(r, "rand100")
    b, sb = load(r, "new7")
    res = {**a, **b}
    steps = {**sa, **sb}
    res = {k: res[k] for k in final}
    E_ = T = WE = WT = 0
    per = []
    for k in final:
        e, t = err(res[k])
        E_ += e; T += t
        we, wt = werr(res[k]); WE += we; WT += wt
        if t:
            per.append(e / t)
    old93 = [k for k in final if k in a and k not in b]
    e93 = sum(err(res[k])[0] for k in old93); t93 = sum(err(res[k])[1] for k in old93)
    e7 = sum(err(res[k])[0] for k in new7); t7 = sum(err(res[k])[1] for k in new7)
    cap = sum(1 for k in final if steps.get(k, 0) >= 1199)
    utmos = st.mean(res[k]["utmos"] for k in final)
    sbert = st.mean(res[k]["speech_bert"] for k in final)
    print(f"{name:<16}{E_/T:>8.2%}{st.mean(per):>12.2%}{WE/WT:>8.2%}{utmos:>7.2f}{sbert:>11.3f}{cap:>9}   {e93/t93:>9.2%}{e7/t7:>8.2%}")
