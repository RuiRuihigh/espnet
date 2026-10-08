"""LibriSQA Part II scored by continuation log-likelihood (lm-eval-harness style).

Instead of generating "the answer is option x", every candidate continuation is
scored with log P(continuation | prompt) and the argmax is taken. Variants:

  A  cloze    : <startofspeech>U<startoftext> question: q<generatetext> {option text}
               (harness style: options not shown in the prompt, no letters)
  B  listed   : same prompt as training (options listed)       -> {option text}
  L  letter   : same prompt as training                         -> the answer is option {x}

Each variant is also computed with an empty speech segment (<startofspeech><startoftext>),
which gives the calibrated score  log P(cont | speech) - log P(cont | no speech)
("what the speech adds"; removes the model's default-option bias).

Reported per variant: acc (sum logprob), acc_norm (sum logprob / #chars, as in the
harness), acc_calib (speech minus no-speech sum logprob).

Tokenization follows training exactly (espnet CommonPreprocessor: bpe text2tokens ->
token ids, then the first token (BPE leading space) is dropped as in
ESPnetMultitaskLanguageModel._calc_att_loss). Sequences are right-padded with eos like
pad_list(..., eos_id) in training, so the attention mask is all ones.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

from espnet2.tasks.lm import LMTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter

sys.path.insert(0, str(Path(__file__).parent))
from prepare_librisqa import normalize, parse_question_and_options  # noqa: E402

GEN = "<generatetext>"
LETTERS = "abcd"
VARIANTS = ["A", "B", "L"]


def build_items(test_dir, librisqa_json):
    """uid -> dict(units, qA, promptB, options[4], answer)

    V1 dirs (prepare_librisqa.py --p2_format option_text) carry mc.jsonl and their
    prompt is already the cloze prompt (question only); B is rebuilt from the pieces.
    Legacy dirs (dump/part2) have the options-listed prompt; options come from the json.
    """
    units, prompt = {}, {}
    for line in open(test_dir / "text.sqa"):
        uid, body = line.rstrip("\n").split(" ", 1)
        u, rest = body.split("<startoftext>", 1)
        units[uid] = u[len("<startofspeech>"):]
        prompt[uid] = rest[: -len(GEN)]
    items = {}
    if (test_dir / "mc.jsonl").exists():
        for r in map(json.loads, open(test_dir / "mc.jsonl")):
            uid = r["uid"]
            items[uid] = dict(
                units=units[uid],
                qA=prompt[uid],
                promptB=prompt[uid] + " options: " + " ".join(
                    f"{l}. {o}" for l, o in zip(LETTERS, r["options"])),
                options=r["options"],
                answer=r["answer"],
            )
        return items
    gold = dict(l.split() for l in open(test_dir / "answer"))
    for x in json.load(open(librisqa_json)):
        uid = "sqa2_" + Path(x["speech_path"]).stem
        if uid not in units:
            continue
        q, opts = parse_question_and_options(x["question_and_option"])
        items[uid] = dict(
            units=units[uid],
            qA=" " + normalize("question: " + q),
            promptB=prompt[uid],
            options=[normalize(opts[k]) for k in "ABCD"],
            answer=gold[uid],
        )
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm_train_config", required=True)
    ap.add_argument("--lm_file", required=True)
    ap.add_argument("--test_dir", type=Path, default=Path("dump/part2/raw/test"))
    ap.add_argument("--librisqa_json", default="downloads/LibriSQA/LibriSQA-PartII-test.json")
    ap.add_argument("--keys", type=Path, default=None, help="subset of uids (for sharding)")
    ap.add_argument("--out", type=Path, required=True, help="per-uid scores (jsonl)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--variants", default="A,B,L", help="subset of A,B,L")
    ap.add_argument("--with_rand", type=lambda x: x.lower() == "true", default=True,
                    help="also score with a mismatched utterance's speech (calib_rand baseline)")
    args = ap.parse_args()

    model, train_args = LMTask.build_model_from_file(args.lm_train_config, args.lm_file, args.device)
    model.eval()
    tokenizer = build_tokenizer(token_type="bpe", bpemodel=train_args.bpemodel)
    converter = TokenIDConverter(token_list=train_args.token_list)
    eos = model.eos_id

    def encode(text):
        return converter.tokens2ids(tokenizer.text2tokens(text))[1:]

    items = build_items(args.test_dir, args.librisqa_json)
    uids = sorted(items)
    # mismatched speech: a fixed derangement over the FULL sorted list (independent of sharding)
    rand_src = {u: uids[(i + len(uids) // 2) % len(uids)] for i, u in enumerate(uids)}
    variants = args.variants.split(",")
    if args.keys:
        keep = {l.split()[0] for l in open(args.keys)}
        uids = [u for u in uids if u in keep]

    n_boundary_mismatch = 0

    @torch.no_grad()
    def score(pairs):
        """pairs: list of (context, continuation) -> list of summed logprob of continuation"""
        nonlocal n_boundary_mismatch
        seqs, starts = [], []
        for ctx, cont in pairs:
            c_ids = encode(ctx)
            full = encode(ctx + cont)
            if full[: len(c_ids)] != c_ids:
                n_boundary_mismatch += 1
            seqs.append(full)
            starts.append(len(c_ids))
        L = max(map(len, seqs))
        x = torch.full((len(seqs), L), eos, dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, : len(s)] = torch.tensor(s)
        x = x.to(args.device)
        logits, _ = model.lm(x, None)
        logp = torch.log_softmax(logits.float(), dim=-1)
        res = []
        for i, s in enumerate(seqs):
            pos = torch.arange(starts[i], len(s), device=args.device)
            tgt = x[i, starts[i]: len(s)]
            res.append(logp[i, pos - 1, tgt].sum().item())
        return res

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fp:
        for n, uid in enumerate(uids):
            it = items[uid]
            rec = {"uid": uid, "answer": it["answer"],
                   "opt_chars": [len(o) for o in it["options"]]}
            conds = [("sp", it["units"]), ("nosp", "")]
            if args.with_rand:
                conds.append(("rand", items[rand_src[uid]]["units"]))
            for sp_name, u in conds:
                head = f"<startofspeech>{u}<startoftext>"
                pairs = []
                if "A" in variants:
                    pairs += [(head + it["qA"] + GEN, " " + o) for o in it["options"]]
                if "B" in variants:
                    pairs += [(head + it["promptB"] + GEN, " " + o) for o in it["options"]]
                if "L" in variants:
                    pairs += [(head + it["promptB"] + GEN, f" the answer is option {l}") for l in LETTERS]
                s = score(pairs)
                for vi, v in enumerate(v for v in VARIANTS if v in variants):
                    rec[f"{v}_{sp_name}"] = s[4 * vi: 4 * vi + 4]
            fp.write(json.dumps(rec) + "\n")
            if n % 200 == 0:
                print(f"{n}/{len(uids)} done", flush=True)
    print(f"boundary_mismatch {n_boundary_mismatch}", flush=True)


def summarize(jsonl_files, out):
    recs = [json.loads(l) for f in jsonl_files for l in open(f)]
    argmax = lambda v: LETTERS[max(range(4), key=lambda i: v[i])]  # noqa: E731
    lines = [f"n\t{len(recs)}", f"gold_distribution\t{dict(sorted(Counter(r['answer'] for r in recs).items()))}", ""]
    header = "variant\tscoring\tacc\tpred_distribution"
    lines.append(header)
    for v in VARIANTS:
        if f"{v}_sp" not in recs[0]:
            continue
        scorers = {
            "sum": lambda r: r[f"{v}_sp"],
            "norm": lambda r: [s / c for s, c in zip(r[f"{v}_sp"], r["opt_chars"])],
            "calib": lambda r: [a - b for a, b in zip(r[f"{v}_sp"], r[f"{v}_nosp"])],
            "calib_rand": lambda r: [a - b for a, b in zip(r[f"{v}_sp"], r[f"{v}_rand"])],
            "nospeech": lambda r: r[f"{v}_nosp"],
        }
        if f"{v}_rand" not in recs[0]:
            scorers.pop("calib_rand")
        if v == "L":
            scorers.pop("norm")  # all letter continuations have the same length
        for name, fn in scorers.items():
            preds = [argmax(fn(r)) for r in recs]
            acc = sum(p == r["answer"] for p, r in zip(preds, recs)) / len(recs) * 100
            dist = dict(sorted(Counter(preds).items()))
            lines.append(f"{v}\t{name}\t{acc:.2f}\t{dist}")
    Path(out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summarize(sys.argv[3:], sys.argv[2])
    else:
        main()
