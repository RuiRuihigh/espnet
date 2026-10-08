#!/usr/bin/env python3
"""Trace WHY beam search (beam 5) loses a sentence that greedy decoding gets right.

For each requested utterance this runs the SAME decode as lm.sh Stage 10 (same LM, same
preprocessing, beam 5, maxlen 1200, penalty 0) but records, at every step, the scores of
the hypotheses kept in the beam and whether the GREEDY path's prefix is still among them.
It also teacher-forces the greedy sequence and the beam's final sequence through the LM
to get each path's own cumulative log-probability at every step, so the two can be compared
directly even after the greedy prefix has left the beam.

Usage (from a recipe root, espnet env, GPU):
  python3 local/trace_beam.py --keys 8131_117029_000007_000000,... \
      --text dump/raw/test/text.tts.rpt_n0 \
      --greedy_dir exp/.../decode_test_tts/text.tts.samp_greedy \
      --beam_dir   exp/.../decode_test_tts/text.tts.rpt_n0 \
      --out exp/.../beam_trace  [--maxlen 1200]
Nothing under exp/.../decode_test_tts/<existing> is written; output goes to --out only.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import torch

import espnet2.bin.lm_inference as li
from espnet2.tasks.lm import LMTask

MARKER = "<generatespeech>"


def read_tokens(dir_, key):
    """generated BPE pieces (after <generatespeech>) of one utt from a decode dir's token file"""
    for line in open(Path(dir_) / "token"):
        p = line.split()
        if p[0] == "tts_" + key or p[0] == key:
            t = p[1:]
            return t[t.index(MARKER) + 1 :]
    raise KeyError(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--greedy_dir", required=True)
    ap.add_argument("--beam_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--maxlen", type=int, default=1200)
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--lm_dir", default="exp/lm_train_transformer_opt350_en_bpe10000")
    ap.add_argument("--lm_file", default="valid.acc.best.pth")
    args = ap.parse_args()
    logging.basicConfig(level="WARNING")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = [k for k in args.keys.split(",") if k]
    # the streaming iterator walks the text file once, so the key file must follow the file's order
    order = {l.split()[0].replace("tts_", "", 1): n for n, l in enumerate(open(args.text))}
    keys.sort(key=lambda k: order[k])
    key_file = out / "keys.scp"
    key_file.write_text("\n".join("tts_" + k for k in keys) + "\n")

    gen = li.GenerateText(
        lm_train_config=f"{args.lm_dir}/config.yaml",
        lm_file=f"{args.lm_dir}/{args.lm_file}",
        token_type="bpe",
        bpemodel="data/en_token_list/bpe_unigram10000/bpe.model",
        device="cuda",
        maxlen=args.maxlen,
        minlen=1,
        dtype="float32",
        beam_size=args.beam_size,
        ngram_weight=0.0,
        penalty=0.0,
        nbest=1,
        quantize_lm=False,
        quantize_modules=["Linear"],
        quantize_dtype="qint8",
    )
    bs = gen.beam_search
    token_list = gen.lm_train_args.token_list
    eos = gen.lm.eos_id
    conv = gen.converter
    Base = type(bs)
    trace = {"P": None, "steps": [], "greedy_ids": None}

    class Traced(Base):
        def post_process(self, i, maxlen, minlen, maxlenratio, running_hyps, ended_hyps):
            n = len(running_hyps)
            lens = running_hyps.length.tolist()
            if trace["P"] is None:
                trace["P"] = lens[0] - 1  # prompt (sos + primer) length
            P = trace["P"]
            yseq = running_hyps.yseq.tolist()
            scores = running_hyps.score.tolist()
            G = trace["greedy_ids"]
            rows = []
            g_rank = None
            order = sorted(range(n), key=lambda j: -scores[j])
            for rank, j in enumerate(order):
                gen_ids = yseq[j][P : lens[j]]
                is_g = G is not None and gen_ids == G[: len(gen_ids)] and len(gen_ids) == i + 1
                if is_g:
                    g_rank = rank
                rows.append((scores[j], is_g, yseq[j][lens[j] - 1] == eos))
            trace["steps"].append(
                {
                    "i": i,
                    "n": n,
                    "scores": [round(r[0], 2) for r in rows],
                    "greedy_rank": g_rank,
                    "greedy_score": next((round(r[0], 2) for r in rows if r[1]), None),
                    "n_eos": sum(1 for r in rows if r[2]),
                    "n_ended_total": len(ended_hyps),
                    "top_last": token_list[yseq[order[0]][lens[order[0]] - 1]],
                }
            )
            return super().post_process(i, maxlen, minlen, maxlenratio, running_hyps, ended_hyps)

    bs.__class__ = Traced

    @torch.no_grad()
    def teacher_forced(prefix, gen_ids):
        """per-step log-prob of gen_ids given prefix, plus log P(eos) after each position"""
        ys = torch.tensor([prefix + gen_ids], device="cuda")
        m = gen.lm.lm
        h = m.decoder(ys, use_cache=False, return_dict=True).last_hidden_state
        logp = m.lm_head(h)[0].log_softmax(-1)  # (L, V); row t predicts token t+1
        P = len(prefix)
        tok = [logp[P - 1 + t, gen_ids[t]].item() for t in range(len(gen_ids))]
        eos_lp = [logp[P - 1 + t, eos].item() for t in range(len(gen_ids) + 1)]  # after t tokens
        return tok, eos_lp

    loader = LMTask.build_streaming_iterator(
        [(args.text, "text", "text")],
        dtype="float32",
        batch_size=1,
        key_file=str(key_file),
        num_workers=1,
        preprocess_fn=LMTask.build_preprocess_fn(gen.lm_train_args, False),
        collate_fn=LMTask.build_collate_fn(gen.lm_train_args, False),
        allow_variable_data_keys=False,
        inference=True,
    )
    for ks, batch in loader:
        key = ks[0].replace("tts_", "", 1)
        batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}
        g_tokens = read_tokens(args.greedy_dir, key)
        prev_beam = read_tokens(args.beam_dir, key)
        trace["P"], trace["steps"], trace["greedy_ids"] = None, [], conv.tokens2ids(g_tokens)
        res = gen(**batch)
        _, tokens, token_int, hyp = res[0]
        gi = trace["greedy_ids"]
        P = trace["P"]
        prefix = hyp.yseq[:P].tolist()
        b_tokens = tokens[tokens.index(MARKER) + 1 :] if MARKER in tokens else tokens
        b_ids = conv.tokens2ids(b_tokens)
        g_tok, g_eos = teacher_forced(prefix, gi)
        b_tok, b_eos = teacher_forced(prefix, b_ids)
        import itertools

        cum = lambda xs: list(itertools.accumulate(xs))
        S_g, S_b = cum(g_tok), cum(b_tok)
        steps = trace["steps"]
        # first step at which the greedy prefix is no longer in the beam
        drop = next((s["i"] for s in steps if s["greedy_rank"] is None), None)
        rec = {
            "key": key,
            "same_as_earlier_beam_decode": b_tokens == prev_beam,
            "P": P,
            "greedy_len": len(gi),
            "beam_len": len(b_ids),
            "greedy_total_with_eos": S_g[-1] + g_eos[-1],
            "greedy_eos_logp_at_end": g_eos[-1],
            "beam_total": S_b[-1],
            "greedy_dropped_at_step": drop,
            "n_ended_hyps_final": steps[-1]["n_ended_total"] if steps else None,
            "steps": steps,
            "S_greedy": S_g,
            "S_beam": S_b,
            "greedy_pieces": g_tokens,
            "beam_pieces": b_tokens,
        }
        json.dump(rec, open(out / f"{key}.json", "w"))
        print(f"\n===== {key} =====")
        print(f"reproduces the earlier beam-5 decode exactly: {rec['same_as_earlier_beam_decode']}")
        print(f"greedy: {len(gi)} steps, total log-prob incl. EOS = {rec['greedy_total_with_eos']:.1f} "
              f"(log P(EOS) at its end = {g_eos[-1]:.2f})")
        print(f"beam-5: {len(b_ids)} steps, total log-prob = {S_b[-1]:.1f}; ended hyps overall = {rec['n_ended_hyps_final']}")
        if drop is None:
            print("greedy prefix stayed in the beam for the whole decode")
        else:
            print(f"greedy prefix LEFT the beam at step {drop} (of {len(gi)} it needs)")
            for s in steps[max(0, drop - 2) : drop + 2]:
                sg = S_g[s["i"]] if s["i"] < len(S_g) else None
                print(f"  step {s['i']:>4}: beam scores {s['scores']} | greedy prefix in beam? "
                      f"{'rank ' + str(s['greedy_rank']) if s['greedy_rank'] is not None else 'NO'} "
                      f"| greedy own cum score {sg:.2f}" if sg is not None else f"  step {s['i']}")
        # where does the beam's final path first out-score the greedy path (same length prefix)?
        t_div = next((t for t in range(min(len(g_tokens), len(b_tokens))) if g_tokens[t] != b_tokens[t]), None)
        t_first = next((t for t in range(min(len(S_g), len(S_b))) if S_b[t] > S_g[t] + 1e-3), None)
        print(f"beam-final and greedy tokens first DIFFER at step {t_div}")
        print(f"beam-final path first has a higher cumulative score than greedy (by >0.001) at step {t_first}")
        sp = sorted(s["scores"][0] - s["scores"][-1] for s in steps if len(s["scores"]) >= 5)
        if sp:
            print(f"beam flatness: best-minus-5th cumulative score, median over steps = {sp[len(sp)//2]:.2f} nats, "
                  f"first 30 steps max = {max(s['scores'][0]-s['scores'][-1] for s in steps[:30] if len(s['scores'])>=5):.2f}")
        first_eos = next((s["i"] for s in steps if s["n_eos"] > 0), None)
        print(f"first step at which ANY hypothesis in the beam emitted EOS: {first_eos}")
        print(f"greedy own EOS log-prob at its end step {len(gi)}: {g_eos[-1]:.2f}; "
              f"beam-5 kept-in-beam greedy prefix until step {drop}")
        for t in (0, 1, 2, 5, 10, 20, 50, 100, 150, 200):
            if t < min(len(S_g), len(S_b)):
                print(f"  t={t:>3}: cum greedy {S_g[t]:8.2f} | cum beam-final {S_b[t]:8.2f} | "
                      f"beam-token {b_tokens[t]!r} vs greedy-token {g_tokens[t]!r}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
