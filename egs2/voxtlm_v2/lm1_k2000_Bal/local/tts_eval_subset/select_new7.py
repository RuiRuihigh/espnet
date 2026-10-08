#!/usr/bin/env python3
"""Draw 7 NEW test sentences (uniformly at random, fixed seed) that are not in
keys_random100.txt. Used to replace the 7 sentences that ran away (v2: 6, ttskmeans: 1)
so the final comparison set is 93 kept + 7 new = 100 DISTINCT sentences.
Run from any recipe root (needs data/test/speech/tts/text, identical in all recipes)."""
import random
from pathlib import Path

d = Path(__file__).parent
ids = sorted(line.split(" ", 1)[0] for line in open("data/test/speech/tts/text"))
used = {l.strip() for l in open(d / "keys_random100.txt") if l.strip()}
pool = [k for k in ids if k not in used]
random.seed(4321)
sel = sorted(random.sample(pool, 7))
(d / "keys_new7.txt").write_text("\n".join(sel) + "\n")
print(f"pool={len(pool)} (all {len(ids)} minus the {len(used)} already used); wrote {d/'keys_new7.txt'}")
