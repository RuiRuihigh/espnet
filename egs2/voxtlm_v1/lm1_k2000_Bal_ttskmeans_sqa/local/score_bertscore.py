"""BERTScore for LibriSQA Part I answers (roberta-large, layer 17, bert-score defaults for en).

Reports mean P/R/F1, both raw and rescaled with the bert-score baseline (rescaled is the
number usually reported: ~0 for unrelated sentences, higher = closer meaning).
Optionally scores the "answer = ground-truth transcript" baseline on the same utterances.

Needs bert-score, installed isolated from the shared venv:
    PYTHONPATH=tools/pylib HF_HOME=tools/hf_bertscore HF_HUB_OFFLINE=1
"""

import argparse
from pathlib import Path

import torch
from bert_score import BERTScorer

GEN = "<generatetext>"


def after_gen(path):
    res = {}
    for line in open(path):
        uid = line.split(maxsplit=1)[0]
        res[uid] = line.rstrip("\n").split(GEN)[-1].strip() if GEN in line else ""
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", type=Path, required=True, help="decode text (uid + full sequence)")
    ap.add_argument("--ref_dir", type=Path, required=True, help="dir with lm_text (+ transcript)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--with_transcript", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    refs = after_gen(args.ref_dir / "lm_text")
    hyps = after_gen(args.hyp)
    uids = sorted(refs)
    scorer_raw = BERTScorer(lang="en", rescale_with_baseline=False, device=args.device)
    scorer_rs = BERTScorer(lang="en", rescale_with_baseline=True, device=args.device)

    def run(cands, tag):
        # empty hypotheses get 0 instead of being scored
        idx = [i for i, c in enumerate(cands) if c]
        lines = [f"{tag}_n_empty\t{len(cands) - len(idx)}"]
        for name, sc in (("raw", scorer_raw), ("rescaled", scorer_rs)):
            P, R, F = sc.score([cands[i] for i in idx], [refs[uids[i]] for i in idx], batch_size=64)
            n = len(cands)
            for m, v in (("P", P), ("R", R), ("F1", F)):
                lines.append(f"{tag}_bertscore_{name}_{m}\t{v.sum().item() / n * 100:.2f}")
        return lines

    lines = [f"n\t{len(uids)}", f"missing_hyp\t{sum(u not in hyps for u in uids)}"]
    lines += run([hyps.get(u, "") for u in uids], "model")
    if args.with_transcript:
        tr = dict(l.rstrip("\n").split(" ", 1) for l in open(args.ref_dir / "transcript"))
        lines += run([tr[u] for u in uids], "baseline_transcript")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
