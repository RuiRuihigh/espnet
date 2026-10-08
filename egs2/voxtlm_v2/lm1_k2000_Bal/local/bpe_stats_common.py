"""Shared helpers for local/report_bpe_stats.py and local/compare_bpe_sizes.py.

Both scripts group per-utterance token lengths by task (from the uttid
prefix) and print the same Markdown table shape; this module holds that
common logic so the two scripts don't drift apart.
"""

import statistics
from typing import Dict, List, NamedTuple


class Row(NamedTuple):
    task: str
    length: int


def task_of(uttid: str) -> str:
    """First underscore-delimited field of the uttid, e.g. 'asr' from
    'asr_1089-134686-0000' or 'textlm' from 'textlm_textlm_librispeech_...'.
    """
    return uttid.split("_", 1)[0] if "_" in uttid else uttid


def percentile(values: List[int], p: float) -> int:
    values = sorted(values)
    k = (len(values) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return round(values[f] + (values[c] - values[f]) * (k - f))


def summarize(rows: List[Row], pct: float) -> Dict[str, object]:
    lengths = [r.length for r in rows]
    return {
        "n_utt": len(rows),
        "min": min(lengths),
        "mean": statistics.mean(lengths),
        "median": statistics.median(lengths),
        f"p{pct:g}": percentile(lengths, pct),
        "max": max(lengths),
        "total_tokens": sum(lengths),
    }


def group_by_task(rows: List[Row]) -> Dict[str, List[Row]]:
    by_task: Dict[str, List[Row]] = {}
    for r in rows:
        by_task.setdefault(r.task, []).append(r)
    return by_task


def format_table(
    title: str,
    stats_by_group: Dict[str, Dict[str, object]],
    pct: float,
    extra_cols: List[str] = (),
) -> str:
    """extra_cols: additional stat keys (already present in each group's
    dict) to render as leading columns before the usual n_utt/min/.../max
    set -- e.g. a 'bpe_size' column when comparing across vocab sizes.
    """
    stat_cols = ["n_utt", "min", "mean", "median", f"p{pct:g}", "max",
                 "total_tokens"]
    stat_header = ["#Utts", "Min", "Mean", "Median", f"P{pct:g}", "Max",
                    "Total tokens"]
    header = ["Group"] + [c.replace("_", " ").title() for c in extra_cols] + stat_header
    lines = [f"### {title}", "", "| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for group, s in stats_by_group.items():
        cells = [group] + [str(s[c]) for c in extra_cols]
        for c in stat_cols:
            v = s[c]
            cells.append(f"{v:.1f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
