#!/usr/bin/env python3
"""Regenerate keys.txt: 100 TTS test utterances with ground-truth discrete
unit length < 200 (~< 4s of audio at 50Hz), fixed random seed for
reproducibility. See README.md in this directory for why.

Run from egs2/voxtlm_v1/lm1_k1000_Bal (needs dump/raw/test/speech/tts/token
from a completed Stage 3, which produced this recipe's tokenizer labels;
ground-truth unit-sequence *length* is identical across tokenizers sharing
the same 50Hz frame rate, so this selection is reused as-is by the other
two recipes -- do not regenerate per recipe).
"""
import random
from pathlib import Path

TOKEN_FILE = "dump/raw/test/speech/tts/token"
MAX_LEN = 200
N_SELECT = 100
SEED = 42
OUT = Path(__file__).parent / "keys.txt"


def main():
    gt = {}
    with open(TOKEN_FILE) as f:
        for line in f:
            parts = line.split()
            gt[parts[0]] = len(parts) - 1

    cand = sorted(u for u, l in gt.items() if l < MAX_LEN)
    random.seed(SEED)
    random.shuffle(cand)
    sel = sorted(cand[:N_SELECT])

    with open(OUT, "w") as f:
        for u in sel:
            f.write(u + "\n")

    lens = [gt[u] for u in sel]
    print(f"candidates(<{MAX_LEN}): {len(cand)}  selected: {len(sel)}")
    print(f"length range: {min(lens)}~{max(lens)}  median: {sorted(lens)[len(lens)//2]}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
