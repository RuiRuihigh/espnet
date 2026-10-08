"""Score LibriSQA decoding output of espnet2.bin.lm_inference.

Part II: accuracy of the option letter parsed from "the answer is option X"
         (fallback: first standalone a-d token), plus confusion / parse failures.
Part I : corpus BLEU-4 (nltk) and ROUGE-L F1 against the reference answer.
"""

import argparse
import re
from collections import Counter
from pathlib import Path

GEN_TEXT = "<generatetext>"


def read_after_sos(path):
    res = {}
    for line in open(path):
        uid = line.split(maxsplit=1)[0]
        res[uid] = line.rstrip("\n").split(GEN_TEXT)[-1].strip() if GEN_TEXT in line else ""
    return res


def parse_letter(hyp):
    m = re.search(r"\boption ([a-d])\b", hyp)
    if m:
        return m.group(1)
    m = re.search(r"(?:^|\s)([a-d])(?:[.)\s]|$)", hyp)
    return m.group(1) if m else "none"


def lcs(a, b):
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b):
            cur = dp[j + 1]
            dp[j + 1] = prev + 1 if x == y else max(dp[j + 1], dp[j])
            prev = cur
    return dp[-1]


def rouge_l(hyp, ref):
    h, r = hyp.split(), ref.split()
    if not h or not r:
        return 0.0
    l = lcs(h, r)
    if l == 0:
        return 0.0
    p, rc = l / len(h), l / len(r)
    return 2 * p * rc / (p + rc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, choices=[1, 2], required=True)
    ap.add_argument("--hyp", type=Path, required=True, help="decode text (uid + full sequence)")
    ap.add_argument("--ref_dir", type=Path, required=True, help="dir with lm_text (and answer for Part II)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    hyps = read_after_sos(args.hyp)
    refs = read_after_sos(args.ref_dir / "lm_text")
    missing = [u for u in refs if u not in hyps]
    lines = [f"n_ref\t{len(refs)}", f"n_hyp\t{len(hyps)}", f"missing_hyp\t{len(missing)}"]

    if args.part == 2:
        gold = dict(l.split() for l in open(args.ref_dir / "answer"))
        pred = {u: parse_letter(hyps.get(u, "")) for u in gold}
        correct = sum(pred[u] == gold[u] for u in gold)
        gold_cnt = Counter(gold.values())
        conf = Counter((gold[u], pred[u]) for u in gold)
        lines += [
            f"accuracy\t{correct / len(gold) * 100:.2f}\t({correct}/{len(gold)})",
            f"majority_baseline\t{gold_cnt.most_common(1)[0][1] / len(gold) * 100:.2f}\t(always '{gold_cnt.most_common(1)[0][0]}')",
            "random_baseline\t25.00",
            f"parse_fail\t{sum(p == 'none' for p in pred.values())}",
            f"pred_distribution\t{dict(sorted(Counter(pred.values()).items()))}",
            f"gold_distribution\t{dict(sorted(gold_cnt.items()))}",
            "confusion (rows=gold, cols=pred a b c d none)",
        ]
        for g in "abcd":
            lines.append(g + "\t" + "\t".join(str(conf[(g, p)]) for p in ["a", "b", "c", "d", "none"]))
        with open(args.out.parent / "pred_letter", "w") as fp:
            fp.writelines(f"{u} {pred[u]} {gold[u]}\n" for u in sorted(gold))
    else:
        from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

        uids = sorted(refs)
        h = [hyps.get(u, "").split() for u in uids]
        r = [[refs[u].split()] for u in uids]
        bleu = corpus_bleu(r, h, smoothing_function=SmoothingFunction().method1)
        rl = sum(rouge_l(hyps.get(u, ""), refs[u]) for u in uids) / len(uids)
        lines += [
            f"bleu4\t{bleu * 100:.2f}",
            f"rougeL_f1\t{rl * 100:.2f}",
            f"avg_hyp_words\t{sum(map(len, h)) / len(h):.1f}",
            f"avg_ref_words\t{sum(len(x[0]) for x in r) / len(r):.1f}",
        ]
        # baseline: answer = ground-truth transcript of the speech (no question understanding).
        # A model that only transcribes relevant words should not beat this by much.
        tpath = args.ref_dir / "transcript"
        if tpath.exists():
            tr = dict(l.rstrip("\n").split(" ", 1) for l in open(tpath))
            tb = corpus_bleu(r, [tr[u].split() for u in uids], smoothing_function=SmoothingFunction().method1)
            trl = sum(rouge_l(tr[u], refs[u]) for u in uids) / len(uids)
            lines += [f"baseline_transcript_bleu4\t{tb * 100:.2f}", f"baseline_transcript_rougeL_f1\t{trl * 100:.2f}"]

    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
