#!/usr/bin/env python3
"""Turn a perform_kmeans.sh pseudo_labels_km<N>.txt into ESPnet LM text.

Mirrors local/prepare_lm_data.py's prepare_speechlm() exactly (same unit2cjk
mapping, same "join with no separator", same <generatespeech> prefix) so the
sLM21 speech track is tokenized identically to how speechlm/unitlm training
data was built -- no deduplication of repeated consecutive units happens
anywhere in that pipeline, so this doesn't do any either, it just mirrors it.

Input line format (from perform_kmeans.sh --stage 3): "<uttid> <int> <int> ...".
Output line format: "<uttid> <generatespeech> <cjk-mapped unit string>".
"""

import argparse
from pathlib import Path


def unit2cjk(unit) -> str:
    """Same mapping as local/prepare_lm_data.py's unit2cjk."""
    return chr(0x4E00 + int(unit))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pseudo-labels", type=Path, required=True,
                         help="pseudo_labels_km<N>.txt from perform_kmeans.sh stage 3.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sos-sym", default="<generatespeech>")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.pseudo_labels.open() as fin, args.out.open("w") as fout:
        for line in fin:
            uttid, *units = line.split()
            if not units:
                raise ValueError(f"{args.pseudo_labels}: {uttid!r} has no units")
            cjk = "".join(unit2cjk(u) for u in units)
            fout.write(f"{uttid} {args.sos_sym} {cjk}\n")
            n += 1

    print(f"{args.pseudo_labels}: wrote {n} utterances to {args.out}")


if __name__ == "__main__":
    main()
