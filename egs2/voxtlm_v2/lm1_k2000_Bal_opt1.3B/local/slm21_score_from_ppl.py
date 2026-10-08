#!/usr/bin/env python3
"""Turn espnet2.bin.lm_calc_perplexity output + a filename map into a sLM21
submission file.

lm_calc_perplexity.py writes, per (deduped) text id: utt2ppl (its per-token
perplexity) and utt2ntokens. sLM21 submissions instead want one *raw*
pseudo-log-probability per item (see docs/sLM21.md in the zerospeech-
benchmarks package: e.g. "aAAfmkmQpVz -313.37445068359375" -- a whole-
sequence log-likelihood, not a per-token perplexity), keyed by .wav filename,
one line per file under <submission>/{lexical,syntactic}/{dev,test}.txt.

Recover the raw total nll from (ppl, ntokens) via ppl = exp(nll / ntokens)
(espnet2's own definition when --log_base is left at its default of None),
so nll = ntokens * ln(ppl); the submitted score is then -nll (higher / less
negative = the model found the sequence more probable = its "answer").

Then expand each deduped text id back out to every filename that shares its
text (from local/prepare_slm21_text.py's filename_map.tsv), since gold.csv
has one row -- one required submission line -- per recording, not per unique
text.
"""

import argparse
import math
from pathlib import Path


def read_kv_file(path: Path) -> dict:
    d = {}
    with path.open() as f:
        for line in f:
            k, v = line.split(maxsplit=1)
            d[k] = v.strip()
    return d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ppl-dir",
        type=Path,
        required=True,
        help="--output_dir espnet2.bin.lm_calc_perplexity was given "
        "(contains utt2ppl, utt2ntokens).",
    )
    parser.add_argument(
        "--filename-map",
        type=Path,
        required=True,
        help="filename_map.tsv from local/prepare_slm21_text.py.",
    )
    parser.add_argument("--out", type=Path, required=True,
                         help="Submission file to write, e.g. .../lexical/dev.txt")
    args = parser.parse_args()

    utt2ppl = read_kv_file(args.ppl_dir / "utt2ppl")
    utt2ntokens = read_kv_file(args.ppl_dir / "utt2ntokens")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_lines = 0
    with args.filename_map.open() as f_map, args.out.open("w") as f_out:
        for line in f_map:
            text_id, filenames = line.rstrip("\n").split("\t")
            ppl = float(utt2ppl[text_id])
            ntokens = float(utt2ntokens[text_id])
            nll = ntokens * math.log(ppl)
            score = -nll
            for filename in filenames.split(","):
                f_out.write(f"{filename} {score}\n")
                n_lines += 1

    print(f"{args.filename_map}: wrote {n_lines} lines to {args.out}")


if __name__ == "__main__":
    main()
