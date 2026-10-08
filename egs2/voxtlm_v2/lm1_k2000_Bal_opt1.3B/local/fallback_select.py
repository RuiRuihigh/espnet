#!/usr/bin/env python3
"""Helpers for the two-level TTS decoding fallback:

    beam search (beam 5, maxlen 1200)  ->  if the output hits the length cap  ->  greedy
    ->  if greedy also hits the cap  ->  top-p 0.9 sampling

A sentence "hits the cap" when its generated BPE steps (tokens after <generatespeech>) reach
maxlen-1: beam search / greedy / sampling all stop at maxlen by force-appending EOS. This is
detectable without any reference text. (The stock "best hypo length == max output length"
warning is NOT usable: it compares prompt+generation length, not generated steps.)

Subcommands (all keys are printed / read WITHOUT the "tts_" prefix):

  capped   <decode_dir> <maxlen>                  print the capped keys of a decode dir
  make_text <src_prompt_file> <keys_file> <out>   filter a prompt file (ids "tts_<key>") to keys
  assemble <out_dir> <maxlen> <dir_L1> [<dir_L2> [<dir_L3>]]
           per sentence, take the output of the first level that did not hit the cap; if every
           level capped, keep the last level's output. Writes <out_dir>/score_tts/hyp.tok
           (what local/eval_tts_vocoder.sh reads) and <out_dir>/fallback_log.tsv.
           <out_dir> is wiped first so a stale vocoder_eval/ (VERSA --resume) can never leak in.
"""
import shutil
import sys
from pathlib import Path

MARKER = "<generatespeech>"


def strip(k):
    return k[4:] if k.startswith("tts_") else k


def steps_of(d):
    """key -> number of generated BPE steps, from <d>/token"""
    out = {}
    for line in open(Path(d) / "token"):
        p = line.split()
        t = p[1:]
        if MARKER in t:
            out[strip(p[0])] = len(t) - t.index(MARKER) - 1
    return out


def hyp_tok_of(d):
    """key -> full hyp.tok line (already without the tts_ prefix)"""
    out = {}
    for line in open(Path(d) / "score_tts" / "hyp.tok"):
        line = line.rstrip("\n")
        if line:
            out[strip(line.split(" ", 1)[0])] = line
    return out


def cmd_capped(argv):
    d, maxlen = argv[0], int(argv[1])
    for k, s in sorted(steps_of(d).items()):
        if s >= maxlen - 1:
            print(k)


def cmd_make_text(argv):
    src, keys_file, out = argv
    keys = {l.strip() for l in open(keys_file) if l.strip()}
    n = 0
    with open(out, "w") as f:
        for line in open(src):
            if strip(line.split(" ", 1)[0]) in keys:
                f.write(line)
                n += 1
    print(f"{out}: {n} lines")
    if n != len(keys):
        sys.exit(f"expected {len(keys)} lines, wrote {n}")


def cmd_assemble(argv):
    out_dir, maxlen, levels = Path(argv[0]), int(argv[1]), argv[2:]
    names = ["beam5", "greedy", "top-p 0.9"]
    steps = [steps_of(d) for d in levels]
    toks = [hyp_tok_of(d) for d in levels]
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "score_tts").mkdir(parents=True)
    keys = list(hyp_tok_of(levels[0]).keys())
    log = ["key\tlevel_used\t" + "\t".join(f"steps_{n}" for n in names[: len(levels)]) + "\tstill_capped"]
    counts = {n: 0 for n in names}
    still = 0
    with open(out_dir / "score_tts" / "hyp.tok", "w") as f:
        for k in keys:
            used = None
            for li, (st, tk) in enumerate(zip(steps, toks)):
                if k in st and k in tk and st[k] < maxlen - 1:
                    used = li
                    break
            capped = used is None
            if capped:  # every level capped: keep the deepest level that has this key
                used = max(li for li in range(len(levels)) if k in toks[li])
                still += 1
            counts[names[used]] += 1
            f.write(toks[used][k] + "\n")
            log.append(f"{k}\t{names[used]}\t" + "\t".join(str(st.get(k, "-")) for st in steps) + f"\t{int(capped)}")
    (out_dir / "fallback_log.tsv").write_text("\n".join(log) + "\n")
    print(f"assembled {len(keys)} sentences -> {out_dir}/score_tts/hyp.tok")
    print("  level used: " + ", ".join(f"{n}={c}" for n, c in counts.items() if c or n == "beam5"))
    print(f"  still at the length cap after all levels: {still}")


if __name__ == "__main__":
    cmds = {"capped": cmd_capped, "make_text": cmd_make_text, "assemble": cmd_assemble}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    cmds[sys.argv[1]](sys.argv[2:])
