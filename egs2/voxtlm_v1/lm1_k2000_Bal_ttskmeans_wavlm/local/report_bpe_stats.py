#!/usr/bin/env python3
"""Report a BPE-vocab-size / token-length table from ESPnet2 LM shape files.

Reads the *_shape.bpe files that Stage 6 (lm_calc_stats) writes -- lines of
"<uttid> <length>,<bpe_vocab_size>" -- and prints a Markdown table of token
count / length statistics, broken out by split (train/valid/...) and by task
(inferred from the uttid prefix before the first "_", e.g. asr_/tts_/
textlm_/unitlm_ in voxtlm_v1/lm1's own data).

This reports on the ONE already-trained BPE model this recipe currently
uses. To compare several different BPE vocab sizes against each other
instead, use local/compare_bpe_sizes.py.

Example:
    local/report_bpe_stats.py \
        --token-list data/en_token_list/bpe_unigram10000/tokens.txt \
        --shape-file train:exp/lm_stats_en_bpe10000/train/text_shape.bpe \
        --shape-file valid:exp/lm_stats_en_bpe10000/valid/text_shape.bpe
"""

import argparse
from pathlib import Path
from typing import Dict, List

from bpe_stats_common import Row, format_table, group_by_task, summarize, task_of


def read_shape_file(path: Path) -> List[Row]:
    rows = []
    bpe_vocab_sizes = set()
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            uttid, shape = line.split(maxsplit=1)
            dims = shape.split(",")
            if len(dims) < 2:
                raise ValueError(
                    f"{path}:{lineno}: expected '<length>,<bpe_vocab_size>', "
                    f"got shape={shape!r}"
                )
            rows.append(Row(task_of(uttid), int(dims[0])))
            bpe_vocab_sizes.add(int(dims[1]))
    return rows, bpe_vocab_sizes


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--token-list",
        type=Path,
        default=Path("data/en_token_list/bpe_unigram10000/tokens.txt"),
        help="Vocabulary file, one token per line (used only to cross-check "
        "the BPE vocab size embedded in the shape files).",
    )
    parser.add_argument(
        "--shape-file",
        action="append",
        default=None,
        metavar="LABEL:PATH",
        help="A '<split-name>:<path-to-*_shape.bpe>' pair. Repeatable. "
        "Defaults to train+valid under exp/lm_stats_en_bpe10000/ if omitted.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=95,
        help="Extra percentile column to report alongside min/mean/median/max "
        "(default: 95, useful for spotting long-tail sequences).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Also write the Markdown report to this file (in addition to stdout).",
    )
    args = parser.parse_args()

    shape_files = args.shape_file or [
        "train:exp/lm_stats_en_bpe10000/train/text_shape.bpe",
        "valid:exp/lm_stats_en_bpe10000/valid/text_shape.bpe",
    ]

    vocab_size_from_list = (
        sum(1 for _ in args.token_list.open()) if args.token_list.exists() else None
    )

    report_sections = []
    if vocab_size_from_list is not None:
        report_sections.append(
            f"Token list `{args.token_list}`: **{vocab_size_from_list}** tokens.\n"
        )

    for entry in shape_files:
        if ":" not in entry:
            raise SystemExit(f"--shape-file expects 'LABEL:PATH', got {entry!r}")
        label, path_str = entry.split(":", 1)
        path = Path(path_str)
        if not path.exists():
            print(f"(skipping {label}: {path} not found)\n")
            continue
        rows, bpe_vocab_sizes = read_shape_file(path)

        by_task = group_by_task(rows)
        stats_by_group: Dict[str, Dict[str, object]] = {
            "ALL": summarize(rows, args.percentile)
        }
        for task in sorted(by_task):
            stats_by_group[task] = summarize(by_task[task], args.percentile)

        bpe_size_str = (
            str(sorted(bpe_vocab_sizes)[0])
            if len(bpe_vocab_sizes) == 1
            else "/".join(str(v) for v in sorted(bpe_vocab_sizes)) + " (mixed!)"
        )
        report_sections.append(
            f"`{label}` shape file's embedded BPE vocab size: **{bpe_size_str}**\n"
            + format_table(f"{label} ({path})", stats_by_group, args.percentile)
        )
        if vocab_size_from_list is not None and bpe_size_str != str(
            vocab_size_from_list
        ):
            report_sections.append(
                f"\n> Note: shape file's embedded BPE vocab size ({bpe_size_str}) "
                f"does not match `--token-list`'s {vocab_size_from_list} tokens -- "
                f"double check `--token-list` points at the right vocabulary."
            )

    report = "\n\n".join(report_sections) + "\n"
    print(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        print(f"(also wrote to {args.out})")


if __name__ == "__main__":
    main()
