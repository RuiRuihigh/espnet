#!/usr/bin/env python3
"""Regenerate keys_random100.txt: 100 TTS test utterances drawn uniformly at
random from ALL of them (no length filter), fixed seed for reproducibility.

Why this exists: keys.txt (select_subset.py) only kept utterances with
ground-truth unit length < 200, which sidesteps the decode-length problem but
also hides how each model does on long sentences. This one deliberately does
NOT filter by length; it is meant to be decoded with a maxlen large enough to
cover the longest test sentence (see README_random100.md).

Run from any recipe root: needs data/test/speech/tts/text (identical in all
recipes -- same test set, same utt ids).
"""
import random
from pathlib import Path

TEXT_FILE = "data/test/speech/tts/text"
N_SELECT = 100
SEED = 1234
OUT = Path(__file__).parent / "keys_random100.txt"


def main():
    ids = sorted(line.split(" ", 1)[0] for line in open(TEXT_FILE))
    random.seed(SEED)
    sel = sorted(random.sample(ids, N_SELECT))
    OUT.write_text("\n".join(sel) + "\n")
    print(f"population: {len(ids)}  selected: {len(sel)}  wrote {OUT}")


if __name__ == "__main__":
    main()
