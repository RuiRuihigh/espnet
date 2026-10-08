#!/usr/bin/env python3
"""Convert Stage 10's predicted discrete-unit sequences (hyp.tok: "<uttid>
<tok1> <tok2> ..." with raw integer cluster ids, already un-CJK-mapped by
local/postprocess_speech.py) into the per-utterance HDF5 "feats" files
ParallelWaveGAN's DiscreteSymbolHiFiGANGenerator decoder expects.

Mirrors ../../../phonologicaltokenizer_modify/ParallelWaveGAN/egs/ljspeech/
phon_voc1/local/preprocess_phon.py's own output convention exactly (shape
(T, 1) float32, written under the "feats" key via parallel_wavegan.utils.
write_hdf5) -- but skips that script's audio loading / tokenizer-extraction
entirely, since the tokens here already came from our LM's TTS decoding, not
from re-running the Phonological Tokenizer over real audio. Must be run
inside the `parallelwavegan` conda env (needs parallel_wavegan installed).

Usage:
    python3 local/tok_to_pwg_feats.py --hyp-tok <path/to/hyp.tok> --dumpdir <outdir>
"""

import argparse
from pathlib import Path

import numpy as np
from parallel_wavegan.utils import write_hdf5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hyp-tok", type=Path, required=True)
    parser.add_argument("--dumpdir", type=Path, required=True)
    parser.add_argument(
        "--nclusters",
        type=int,
        default=2000,
        help="Valid codebook size; tokens outside [0, nclusters) are dropped "
        "rather than passed to the vocoder's embedding table. A handful of "
        "hypotheses (observed: 9/9957) end with the literal '<sos/eos>' "
        "string un-stripped by postprocess_speech.py's per-character "
        "cjk2unit(), which turns each ASCII char into a large negative "
        "pseudo-unit-id -- filtering here is cheaper than fixing that "
        "upstream script for an artifact this rare.",
    )
    args = parser.parse_args()

    args.dumpdir.mkdir(parents=True, exist_ok=True)

    n = 0
    n_dropped_toks = 0
    n_lines_affected = 0
    with args.hyp_tok.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            uttid, _, rest = line.partition(" ")
            all_toks = [int(t) for t in rest.split()]
            toks = [t for t in all_toks if 0 <= t < args.nclusters]
            if len(toks) != len(all_toks):
                n_dropped_toks += len(all_toks) - len(toks)
                n_lines_affected += 1
                print(
                    f"{uttid}: dropped {len(all_toks) - len(toks)} "
                    f"out-of-range token(s) (valid range [0, {args.nclusters}))"
                )
            if not toks:
                # empty generation -- skip rather than write a 0-length
                # feature the decoder/vocoder can't do anything useful with.
                print(f"skip {uttid}: empty token sequence")
                continue
            feats = np.array(toks, dtype=np.float32).reshape(-1, 1)
            out_path = args.dumpdir / f"{uttid}.h5"
            write_hdf5(str(out_path), "feats", feats)
            n += 1

    print(
        f"wrote {n} feature files to {args.dumpdir} "
        f"({n_dropped_toks} out-of-range tokens dropped across "
        f"{n_lines_affected} utterance(s))"
    )


if __name__ == "__main__":
    main()
