"""Build VoxtLM-style SQA finetuning data from LibriSQA.

Line format (same conventions as local/prepare_lm_data.py of lm1_k1000_Bal):
    sqa{part}_<uttid> <startofspeech>{units as CJK}<startoftext> {prompt}<generatetext> {answer}

The multitask LM masks everything up to the first <generatetext>, so loss is
only on {answer}. Speech units are NOT re-extracted: LibriSQA uses LibriSpeech
train-clean-360 (train) and test-clean (test), whose k1000 units already exist in
lm1_k1000_Bal/dump/raw/{train,test}/speech/asr/token.

Outputs (under --out, which becomes lm.sh's ${dumpdir}/raw):
    lm_train.txt            SQA train (minus dev speakers) + replay of the 4
                            original tasks, shuffled
    dev/lm_text             SQA dev (held-out train-clean-360 speakers)
    test/lm_text            SQA test with references (all 2620 test-clean utts)
    test/text.sqa           test prompts, cut right after <generatetext>
    test/answer             Part II only: "<uid> <letter a-d>"
    prep_report.txt         counts of everything dropped / shuffled
"""

import argparse
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

import sentencepiece as spm

START_SPEECH = "<startofspeech>"
START_TEXT = "<startoftext>"
GEN_TEXT = "<generatetext>"


def unit2cjk(unit):
    return chr(int("4e00", 16) + int(unit))


def read_tokens(path):
    res = {}
    with open(path) as fp:
        for line in fp:
            k, v = line.rstrip("\n").split(maxsplit=1)
            res[k] = "".join(unit2cjk(t) for t in v.split())
    return res


# ---------------------------------------------------------------------------
# Text normalization. The BPE (trained on LibriSpeech/LibriTTS text) has no
# uppercase and no digits, so everything is lowercased, digits are spelled out,
# and samples that still contain characters the BPE maps to <unk> are dropped.
# ---------------------------------------------------------------------------
ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
ORD_IRREG = {"one": "first", "two": "second", "three": "third", "five": "fifth",
             "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}


def int2words(n):
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + int2words(rest))
    for div, name in ((10**9, "billion"), (10**6, "million"), (1000, "thousand")):
        if n >= div:
            rest = n % div
            return int2words(n // div) + " " + name + ("" if rest == 0 else " " + int2words(rest))


def year2words(n):
    # 1845 -> eighteen forty five, 1800 -> eighteen hundred, 1905 -> nineteen o five
    hi, lo = divmod(n, 100)
    if lo == 0:
        return int2words(hi) + " hundred"
    if lo < 10:
        return int2words(hi) + " o " + ONES[lo]
    return int2words(hi) + " " + int2words(lo)


def ordinal(words):
    last = words.split()[-1]
    if last in ORD_IRREG:
        new = ORD_IRREG[last]
    elif last.endswith("y"):
        new = last[:-1] + "ieth"
    else:
        new = last + "th"
    return words[: len(words) - len(last)] + new


NUM_RE = re.compile(r"(\$)?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(st|nd|rd|th|s)?(%)?")


def num_repl(m):
    dollar, intpart, dec, suffix, pct = m.groups()
    digits = intpart.replace(",", "")
    n = int(digits)
    if n >= 10**12:
        raise ValueError("number too large")
    if not dollar and not dec and "," not in intpart and len(digits) == 4 and 1100 <= n <= 1999:
        words = year2words(n)
    else:
        words = int2words(n)
    if dec:
        words += " point " + " ".join(ONES[int(d)] for d in dec[1:])
    if suffix in ("st", "nd", "rd", "th"):
        words = ordinal(words)
    elif suffix == "s":
        words = words[:-1] + "ies" if words.endswith("y") else words + "s"
    if pct:
        words += " percent"
    if dollar:
        words += " dollars"
    return " " + words + " "


CHAR_MAP = {"’": "'", "‘": "'", "“": '"', "”": '"',
            "–": "-", "—": "-", "£": " pounds ", "&": " and ",
            "\n": " ", "\t": " "}


def normalize(s):
    for a, b in CHAR_MAP.items():
        s = s.replace(a, b)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = NUM_RE.sub(num_repl, s)
    s = s.lower()
    # "word ." spacing artefacts from number substitution
    s = re.sub(r"\s+([.,;:?!%)])", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------
# Part II option parsing / shuffling
# ---------------------------------------------------------------------------
OPT_LINE = re.compile(r"^\s*([A-D])[.)]\s*(.+?)\s*$")
# "Option A", "Options A and C", "Options A, B, and D", "option B or C"
OPT_REF = re.compile(r"\b[Oo]ptions?\s+[A-D]\b(?:(?:\s*(?:,|and|or|/|&))+\s*(?:[Oo]ption\s+)?[A-D]\b)*")
# a bare capital B/C/D (or "(A)") outside any "Option X" span means the
# analysis references options in a way we cannot safely remap
LOOSE_REF = re.compile(r"\b[B-D]\b|\([A-D]\)")


def parse_question_and_options(qo):
    lines = [ln for ln in qo.split("\n") if ln.strip()]
    q = None
    opts = {}
    for ln in lines:
        m = OPT_LINE.match(ln)
        if m:
            opts[m.group(1)] = m.group(2)
        elif ln.strip().lower().startswith("question:"):
            q = ln.split(":", 1)[1].strip()
    if q is None or sorted(opts) != ["A", "B", "C", "D"]:
        return None
    return q, opts


def remap_analysis(analysis, perm):
    """perm: old letter -> new letter. Returns None if not safely remappable."""
    if LOOSE_REF.search(OPT_REF.sub("", analysis)):
        return None
    return OPT_REF.sub(lambda m: re.sub(r"\b[A-D]\b", lambda x: perm[x.group(0)], m.group(0)), analysis)


def build_part2(x, rng, shuffle, stats):
    parsed = parse_question_and_options(x["question_and_option"])
    if parsed is None:
        stats["drop_parse_options"] += 1
        return None
    q, opts = parsed
    ans = x["answer"].strip()
    analysis = x["answer_and_analysis"].strip()
    m = re.match(r"The answer is Option ([A-D])\b", analysis)
    if not m or m.group(1) != ans:
        stats["drop_answer_mismatch"] += 1
        return None
    if shuffle:
        letters = ["A", "B", "C", "D"]
        new_order = letters[:]
        rng.shuffle(new_order)  # new_order[i] = old letter placed at position i
        perm = {old: letters[i] for i, old in enumerate(new_order)}
        new_analysis = remap_analysis(analysis, perm)
        if new_analysis is None:
            stats["shuffle_skipped_unsafe_refs"] += 1
        else:
            opts = {perm[k]: v for k, v in opts.items()}
            ans, analysis = perm[ans], new_analysis
            stats["shuffled"] += 1
    prompt = "question: {} options: {}".format(
        q, " ".join(f"{k.lower()}. {opts[k]}" for k in "ABCD"))
    return normalize(prompt), normalize(analysis), ans.lower()


def build_part2_option_text(x, stats):
    """V1 format (matches cloze/A scoring): prompt = question only, target = correct option text."""
    parsed = parse_question_and_options(x["question_and_option"])
    if parsed is None:
        stats["drop_parse_options"] += 1
        return None
    q, opts = parsed
    ans = x["answer"].strip()
    if ans not in opts:
        stats["drop_answer_mismatch"] += 1
        return None
    return normalize("question: " + q), normalize(opts[ans]), ans.lower(), [normalize(opts[k]) for k in "ABCD"]


def build_part1(x):
    return normalize("question: " + x["question"]), normalize(x["answer"]), None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, nargs="+", choices=[1, 2], required=True,
                    help="one part (legacy layout: test/) or both (test_p1/, test_p2/, dev_p1/, dev_p2/)")
    ap.add_argument("--p2_format", choices=["analysis", "option_text"], default="analysis",
                    help="analysis: options listed in prompt, target = 'the answer is option x because ...'; "
                         "option_text: prompt = question only, target = correct option text (cloze / A scoring)")
    ap.add_argument("--librisqa_dir", type=Path, default=Path("downloads/LibriSQA"))
    ap.add_argument("--src_dump", type=Path, default=Path("../lm1_k1000_Bal/dump/raw"),
                    help="dump/raw of the pretraining recipe (tokens + lm_train.txt for replay)")
    ap.add_argument("--bpemodel", type=Path, default=Path("data/en_token_list/bpe_unigram10000/bpe.model"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dev_speakers_from", type=Path, default=None,
                    help="lm_text of an existing dev set; reuse exactly its speakers (for controlled comparisons)")
    ap.add_argument("--part_dirs", action="store_true",
                    help="write test_p{n}/dev_p{n} even for a single part")
    ap.add_argument("--dev_utts", type=int, default=2000,
                    help="min #utterances (audio files) in the held-out-speaker dev, shared by all parts")
    ap.add_argument("--replay_counts", default="asr=25000,tts=25000,textlm=25000,unitlm=25000",
                    help="#lines sampled per pretraining task; e.g. asr=25000,tts=10000,textlm=25000,unitlm=5000")
    ap.add_argument("--shuffle_options", type=lambda s: s.lower() == "true", default=True,
                    help="Part II analysis format, train only: randomly permute options")
    ap.add_argument("--max_len", type=int, default=1500, help="max BPE length (same as pretraining filter)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    sp = spm.SentencePieceProcessor(model_file=str(args.bpemodel))
    unk = sp.unk_id()
    stats = Counter()
    multi = len(args.parts) > 1 or args.part_dirs

    def bpe_ok(line_body, key):
        ids = sp.encode(line_body)
        if unk in ids:
            stats[f"drop_unk_{key}"] += 1
            return False
        if len(ids) > args.max_len:
            stats[f"drop_len_{key}"] += 1
            return False
        return True

    def make(part, split, tokens):
        tag = "PartI" if part == 1 else "PartII"
        data = json.load(open(args.librisqa_dir / f"LibriSQA-{tag}-{split}.json"))
        out = []
        for x in data:
            uid = Path(x["speech_path"]).stem
            if uid not in tokens:
                stats[f"p{part}_drop_no_token_{split}"] += 1
                continue
            try:
                if part == 1:
                    r = build_part1(x)
                elif args.p2_format == "option_text":
                    r = build_part2_option_text(x, stats)
                else:
                    r = build_part2(x, rng, args.shuffle_options and split == "train", stats)
                    r = r + (None,) if r else None
            except ValueError:
                stats[f"p{part}_drop_number_{split}"] += 1
                continue
            if r is None or r[0] is None:
                continue
            prompt, answer, letter, options = r
            head = f"{START_SPEECH}{tokens[uid]}{START_TEXT} {prompt}{GEN_TEXT}"
            if split == "train" and not bpe_ok(f"{head} {answer}", f"p{part}_{split}"):
                continue
            if split == "test":  # test is never dropped silently: report instead
                ids = sp.encode(f"{head} {answer}")
                stats[f"p{part}_test_with_unk"] += int(unk in ids)
                stats[f"p{part}_test_over_max_len"] += int(len(ids) > args.max_len)
            out.append(dict(uid=f"sqa{part}_{uid}", spk=uid.split("-")[0], head=head, answer=answer,
                            letter=letter, options=options, transcript=normalize(x["text"])))
        stats[f"p{part}_kept_{split}"] = len(out)
        return out

    train_tok = read_tokens(args.src_dump / "train/speech/asr/token")
    test_tok = read_tokens(args.src_dump / "test/speech/asr/token")
    train = {p: make(p, "train", train_tok) for p in args.parts}
    test = {p: make(p, "test", test_tok) for p in args.parts}

    # dev = whole held-out speakers (LibriSpeech uttid = spk-chapter-idx), shared by all parts
    # so that no dev audio appears in training through the other part
    if args.dev_speakers_from:
        dev_spk = {l.split()[0].split("_", 1)[1].split("-")[0]
                   for l in open(args.dev_speakers_from) if l.startswith("sqa")}
    else:
        spk2n = Counter(r["spk"] for r in train[args.parts[0]])
        spks = sorted(spk2n)
        rng.shuffle(spks)
        dev_spk, n = set(), 0
        for s in spks:
            if n >= args.dev_utts:
                break
            dev_spk.add(s)
            n += spk2n[s]
    stats["dev_speakers"] = len(dev_spk)
    dev = {p: [r for r in train[p] if r["spk"] in dev_spk] for p in args.parts}
    train = {p: [r for r in train[p] if r["spk"] not in dev_spk] for p in args.parts}
    for p in args.parts:
        stats[f"p{p}_final_train"] = len(train[p])
        stats[f"p{p}_final_dev"] = len(dev[p])

    lines = [f"{r['uid']} {r['head']} {r['answer']}" for p in args.parts for r in train[p]]

    # replay of the pretraining tasks against catastrophic forgetting
    counts = {k: int(v) for k, v in (kv.split("=") for kv in args.replay_counts.split(","))}
    if any(counts.values()):
        pools = {t: [] for t in counts}
        with open(args.src_dump / "lm_train.txt") as fp:
            for line in fp:
                t = line.split("_", 1)[0]
                if t in pools:
                    pools[t].append(line.rstrip("\n"))
        for t, pool in pools.items():
            if counts[t] == 0:
                continue
            cand = rng.sample(pool, min(len(pool), counts[t] * 2))
            picked = [ln for ln in cand if len(sp.encode(ln.split(maxsplit=1)[1])) <= args.max_len]
            picked = picked[: counts[t]]
            stats[f"replay_{t}"] = len(picked)
            lines += picked
    rng.shuffle(lines)

    # loss-token share per task (what the LM loss is actually averaged over)
    share = Counter()
    for ln in lines:
        u, body = ln.split(" ", 1)
        for s in (GEN_TEXT, "<generatespeech>"):
            if s in body:
                share[u.split("_", 1)[0]] += len(sp.encode(body.split(s, 1)[1])) + 1
                break
    tot = sum(share.values())
    for t, c in share.items():
        stats[f"loss_token_share_{t}"] = f"{c / tot * 100:.1f}%"

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "dev").mkdir(exist_ok=True)
    with open(args.out / "lm_train.txt", "w") as fp:
        fp.writelines(ln + "\n" for ln in lines)
    with open(args.out / "dev/lm_text", "w") as fp:
        fp.writelines(f"{r['uid']} {r['head']} {r['answer']}\n" for p in args.parts for r in dev[p])

    def write_eval_dir(d, recs, part):
        d.mkdir(exist_ok=True)
        with open(d / "lm_text", "w") as fp:
            fp.writelines(f"{r['uid']} {r['head']} {r['answer']}\n" for r in recs)
        with open(d / "text.sqa", "w") as fp:
            fp.writelines(f"{r['uid']} {r['head']}\n" for r in recs)
        if part == 1:  # "answer = transcript" baseline for Part I ROUGE
            with open(d / "transcript", "w") as fp:
                fp.writelines(f"{r['uid']} {r['transcript']}\n" for r in recs)
        if part == 2:
            with open(d / "answer", "w") as fp:
                fp.writelines(f"{r['uid']} {r['letter']}\n" for r in recs)
            if recs and recs[0]["options"] is not None:  # candidates for log-likelihood scoring
                with open(d / "mc.jsonl", "w") as fp:
                    fp.writelines(json.dumps({"uid": r["uid"], "options": r["options"],
                                              "answer": r["letter"]}) + "\n" for r in recs)

    for p in args.parts:
        write_eval_dir(args.out / ("test" if not multi else f"test_p{p}"), test[p], p)
        if multi:
            write_eval_dir(args.out / f"dev_p{p}", dev[p], p)
        if p == 2:
            stats.update({f"p2_test_answer_{l}": c for l, c in Counter(r["letter"] for r in test[p]).items()})

    report = "\n".join(f"{k}\t{v}" for k, v in sorted(stats.items()))
    (args.out / "prep_report.txt").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
