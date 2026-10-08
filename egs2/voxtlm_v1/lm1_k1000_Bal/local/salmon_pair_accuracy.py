#!/usr/bin/env python3
"""SALMon accuracy from espnet2.bin.lm_calc_perplexity output.

Same rule as the official salmon.py: per pair, 1 if loglik(pos) > loglik(neg),
0.5 on a tie, 0 otherwise; accuracy = mean over pairs (chance = 50%).
loglik is the whole-sequence log-likelihood (official default mean_nll=False),
recovered from espnet's per-token perplexity the same way as
local/slm21_score_from_ppl.py: loglik = -ntokens * ln(ppl).

A per-token (length-normalized) variant is also printed for reference only;
the number to report / compare with the SALMon leaderboard is acc(sum).

uttids are "<part>-<ind:04d>-<pos|neg>" (local/prepare_salmon.py), so all
parts can be scored from one combined ppl_out dir and split back by prefix.

Usage:
  salmon_pair_accuracy.py --ppl-dir exp/salmon_eval/lm/ppl_out [--out results.tsv]
"""

import argparse
import math
from collections import defaultdict
from pathlib import Path

PART_ORDER = [
    "sentiment_consistency",
    "speaker_consistency",
    "gender_consistency",
    "bg_domain_consistency",
    "bg_all_consistency",
    "rir_consistency",
    "sentiment_alignment",
    "bg_alignment",
]


def read_kv(path: Path) -> dict:
    with path.open() as f:
        return {k: float(v) for k, v in (line.split(maxsplit=1) for line in f)}


def accuracy(pairs, score):
    total = 0.0
    for pos, neg in pairs:
        p, n = score(pos), score(neg)
        total += 1.0 if p > n else (0.5 if p == n else 0.0)
    return 100.0 * total / len(pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppl-dir", type=Path, required=True,
                        help="Contains utt2ppl, utt2ntokens.")
    parser.add_argument("--out", type=Path, default=None, help="Optional TSV summary to write.")
    args = parser.parse_args()

    utt2ppl = read_kv(args.ppl_dir / "utt2ppl")
    utt2ntok = read_kv(args.ppl_dir / "utt2ntokens")

    part2pairs = defaultdict(list)
    for prefix in sorted({u.rsplit("-", 1)[0] for u in utt2ppl}):
        pos, neg = f"{prefix}-pos", f"{prefix}-neg"
        if pos not in utt2ppl or neg not in utt2ppl:
            raise ValueError(f"incomplete pair {prefix}")
        part2pairs[prefix.rsplit("-", 1)[0]].append((pos, neg))

    def loglik_sum(u):
        return -utt2ntok[u] * math.log(utt2ppl[u])

    def loglik_mean(u):
        return -math.log(utt2ppl[u])

    parts = [p for p in PART_ORDER if p in part2pairs] + sorted(set(part2pairs) - set(PART_ORDER))
    rows = [(p, len(part2pairs[p]), accuracy(part2pairs[p], loglik_sum), accuracy(part2pairs[p], loglik_mean))
            for p in parts]

    print(f"{'part':<24}{'pairs':>7}{'acc(sum)':>10}{'acc(mean)':>11}")
    for part, n, a_sum, a_mean in rows:
        print(f"{part:<24}{n:>7}{a_sum:>10.1f}{a_mean:>11.1f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            f.write("part\tpairs\tacc_sum\tacc_mean\n")
            f.writelines(f"{p}\t{n}\t{a:.2f}\t{m:.2f}\n" for p, n, a, m in rows)


if __name__ == "__main__":
    main()
