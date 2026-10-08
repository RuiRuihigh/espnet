#!/usr/bin/env python3
"""Compare several BPE vocab sizes' effect on token length.

Unlike local/report_bpe_stats.py (which reads the shape files of the ONE
already-trained BPE model this recipe uses), this trains a temporary
SentencePiece model per --nbpe value -- with the same settings Stage 5
(lm.sh) uses -- and re-tokenizes the target corpus with each one, so you can
see the length/vocab-size tradeoff (e.g. the paper's k-vs-BPE-size ablation)
without re-running the full recipe at each size.

Temporary models are cached under --work-dir/k<nclusters>/ and reused on a
later run with the same --nbpe-list, so only new sizes actually retrain.

--nclusters is a *label* for which k the given --bpe-train-text/--corpus were
labeled with -- it namespaces the cache dir and shows up in the report, but
doesn't itself change k (that's fixed by whichever kmeans model actually
produced --corpus). To compare a genuinely different k later: re-run Stage 3
with --nclusters set to the new k (a new kmeans model + a full relabel of
asr/tts/speechlm, which is the slow part -- k itself fits faster for smaller
k), point --bpe-train-text/--corpus at that k's dump/ output, and pass the
matching --nclusters here so its cache/report don't collide with k=1000's.

Example:
    local/compare_bpe_sizes.py --nclusters 1000 --nbpe-list 3000,5000,7000,9000,10000
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import sentencepiece as spm

from bpe_stats_common import Row, format_table, group_by_task, summarize, task_of


def build_nlsyms_arg(bpe_nlsyms: Path) -> str:
    if not bpe_nlsyms.exists():
        return ""
    symbols = []
    with bpe_nlsyms.open() as f:
        for line in f:
            line = line.strip()
            if line:
                symbols.append(line.split()[0])
    return ",".join(symbols)


def train_bpe_model(
    *,
    nbpe: int,
    train_text_stripped: Path,
    model_prefix: Path,
    bpemode: str,
    character_coverage: float,
    input_sentence_size: int,
    user_defined_symbols: str,
    num_threads: int,
) -> Path:
    model_path = model_prefix.with_suffix(".model")
    if model_path.exists():
        print(f"  [nbpe={nbpe}] reusing cached model at {model_path}", file=sys.stderr)
        return model_path

    print(f"  [nbpe={nbpe}] training...", file=sys.stderr)
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        input=str(train_text_stripped),
        vocab_size=nbpe,
        model_type=bpemode,
        model_prefix=str(model_prefix),
        character_coverage=character_coverage,
        input_sentence_size=input_sentence_size,
        num_threads=num_threads,
    )
    if user_defined_symbols:
        kwargs["user_defined_symbols"] = user_defined_symbols
    try:
        spm.SentencePieceTrainer.train(**kwargs)
    except RuntimeError as e:
        if "required_chars" in str(e):
            # At character_coverage=1.0 (Stage 5's own default), sentencepiece
            # needs one vocab slot per distinct character it sees -- and this
            # corpus's "characters" include the CJK-mapped discrete speech
            # units, so the alphabet is much bigger than plain English text
            # would suggest. A small --nbpe can simply be too small to cover
            # it; skip rather than crash the whole comparison.
            print(
                f"  [nbpe={nbpe}] SKIPPED: too small to cover this corpus's "
                f"alphabet at character_coverage={character_coverage} "
                f"({e})",
                file=sys.stderr,
            )
            return None
        raise
    return model_path


def strip_uttid_column(src: Path, dst: Path) -> None:
    """Mirrors Stage 5's `cut -f 2- -d" "` on --bpe-train-text."""
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            _, _, rest = line.rstrip("\n").partition(" ")
            fout.write(rest + "\n")


def measure_lengths(
    corpus: Path, sp: spm.SentencePieceProcessor, max_lines: int = None
) -> List[Row]:
    rows = []
    with corpus.open() as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            line = line.rstrip("\n")
            if not line:
                continue
            uttid, _, text = line.partition(" ")
            n_pieces = len(sp.encode(text, out_type=int))
            rows.append(Row(task_of(uttid), n_pieces))
    return rows


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--nclusters",
        type=int,
        default=1000,
        help="k this recipe's kmeans model currently uses (default: "
        "%(default)s). A label, not a lever: it namespaces --work-dir and the "
        "report so results from a future different-k run don't collide with "
        "or get confused for this one. Changing k for real means retraining "
        "kmeans at that k and relabeling asr/tts/speechlm, then pointing "
        "--bpe-train-text/--corpus at that output -- see the module docstring.",
    )
    parser.add_argument(
        "--nbpe-list",
        default="3000,5000,7000,9000,10000",
        help="Comma-separated BPE vocab sizes to compare (default: "
        "%(default)s -- for k=1000, character_coverage=1.0 needs the "
        "alphabet covered before any merges happen, and that alphabet is "
        "~2051 symbols here (k CJK-mapped units + the English text "
        "characters + special tokens), so anything at or below 2000 is "
        "infeasible; this starts one round number above that floor).",
    )
    parser.add_argument(
        "--bpe-train-text",
        type=Path,
        default=Path("data/train/bpe_text"),
        help="Corpus to fit each BPE model on -- same file Stage 5 uses "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("dump/raw/lm_train.txt"),
        help="Corpus to measure resulting token lengths on -- the full "
        "multitask training text (default: %(default)s).",
    )
    parser.add_argument(
        "--bpe-nlsyms",
        type=Path,
        default=Path("data/nlsyms.txt"),
        help="Non-linguistic symbols file, same as Stage 5's --bpe_nlsyms "
        "(default: %(default)s). Ignored if it doesn't exist.",
    )
    parser.add_argument("--bpemode", default="unigram", choices=["unigram", "bpe"])
    parser.add_argument("--character-coverage", type=float, default=1.0)
    parser.add_argument("--input-sentence-size", type=int, default=100_000_000)
    parser.add_argument(
        "--num-threads",
        type=int,
        default=8,
        help="Passed straight to SentencePieceTrainer (default: %(default)s -- "
        "match this to your job's allocated CPU count).",
    )
    parser.add_argument(
        "--max-corpus-lines",
        type=int,
        default=None,
        help="Only measure the first N lines of --corpus, for a quick "
        "preview instead of the full pass (default: use all lines).",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Where temporary/cached SentencePiece models are written "
        "(default: exp/bpe_compare/k<nclusters>). Safe to delete between "
        "runs.",
    )
    parser.add_argument("--percentile", type=float, default=95)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Also write the Markdown report to this file.",
    )
    args = parser.parse_args()
    if args.work_dir is None:
        args.work_dir = Path("exp/bpe_compare") / f"k{args.nclusters}"

    for p, name in [
        (args.bpe_train_text, "--bpe-train-text"),
        (args.corpus, "--corpus"),
    ]:
        if not p.exists():
            raise SystemExit(f"{name} file not found: {p}")

    nbpe_list = [int(x) for x in args.nbpe_list.split(",")]
    user_defined_symbols = build_nlsyms_arg(args.bpe_nlsyms)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    stripped_train_text = args.work_dir / "bpe_train_text.stripped.txt"
    if not stripped_train_text.exists():
        strip_uttid_column(args.bpe_train_text, stripped_train_text)

    print(f"kmeans k (label): {args.nclusters}", file=sys.stderr)
    print(f"Training corpus:  {args.bpe_train_text}", file=sys.stderr)
    print(f"Measuring on:     {args.corpus}", file=sys.stderr)
    print(f"Comparing nbpe:   {nbpe_list}", file=sys.stderr)
    print(f"Work dir:         {args.work_dir}", file=sys.stderr)

    all_stats: Dict[str, Dict[str, object]] = {}
    for nbpe in nbpe_list:
        model_prefix = args.work_dir / f"bpe_{args.bpemode}{nbpe}" / "bpe"
        model_path = train_bpe_model(
            nbpe=nbpe,
            train_text_stripped=stripped_train_text,
            model_prefix=model_prefix,
            bpemode=args.bpemode,
            character_coverage=args.character_coverage,
            input_sentence_size=args.input_sentence_size,
            num_threads=args.num_threads,
            user_defined_symbols=user_defined_symbols,
        )
        if model_path is None:
            continue
        sp = spm.SentencePieceProcessor(model_file=str(model_path))

        print(f"  [nbpe={nbpe}] measuring lengths on {args.corpus}...", file=sys.stderr)
        rows = measure_lengths(args.corpus, sp, args.max_corpus_lines)
        by_task = group_by_task(rows)

        all_stats[f"ALL (nbpe={nbpe})"] = {
            **summarize(rows, args.percentile),
            "bpe_size": nbpe,
        }
        for task in sorted(by_task):
            all_stats[f"{task} (nbpe={nbpe})"] = {
                **summarize(by_task[task], args.percentile),
                "bpe_size": nbpe,
            }

    # Re-order so rows are grouped by task across sizes, not by size across
    # tasks -- easier to eyeball the length/vocab-size tradeoff per task.
    def sort_key(group_name: str):
        task_part, _, nbpe_part = group_name.partition(" (nbpe=")
        return (task_part != "ALL", task_part, int(nbpe_part.rstrip(")")))

    ordered = {k: all_stats[k] for k in sorted(all_stats, key=sort_key)}

    report = format_table(
        f"Token length vs. BPE size (k={args.nclusters}, {args.corpus})",
        ordered,
        args.percentile,
        extra_cols=["bpe_size"],
    )
    print("\n" + report + "\n")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n")
        print(f"(also wrote to {args.out})")


if __name__ == "__main__":
    main()
