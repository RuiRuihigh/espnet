#!/usr/bin/env python3
"""Computes the final sWUGGY/sBLIMP accuracy from a raw-score submission file.

local/slm21_score_from_ppl.py only produces the per-filename raw score that
the official zerospeech-benchmarks toolkit takes as input (one line per
<filename> <score>); it doesn't compute the actual accuracy number the paper
reports. This script does that last step directly from gold.csv, without
needing the official (and here, uninstalled) zerospeech-benchmarks package.

gold.csv's own "id" + "voice" columns already give exactly the matched pairs
needed: for lexical (sWUGGY), each (id, voice) has one real word row
(correct=1) and one matched nonword row (correct=0), e.g. id=2 pairs
"oxidation" against "accidation" for each of voices A/C/D/F. For syntactic
(sBLIMP), each (id, voice) instead pairs a grammatical sentence (correct=1)
against its ungrammatical minimal-pair counterpart (correct=0). Same two
columns, same pairing logic, works for both tasks unmodified.

Accuracy = fraction of pairs where the model scored the correct=1 item
higher than the correct=0 item (ties count as half credit, per the standard
zerospeech scoring convention).

The pairing key differs by task: lexical's (id, voice) already uniquely
identifies one word/nonword pair (one each per id/voice). syntactic's id
instead batches all 63 grammatical phenomena (type/subtype) tested for a
given voice -- (id, voice) alone collapses 126 rows (63 subtypes x 2) down
to 2, silently keeping only the last subtype seen. type+subtype has to be
part of the key there too.

For the text track, all voices of the same id share one deduplicated text
score (see local/prepare_slm21_text.py), so those pairs are trivially the
same comparison repeated per voice; for the speech track, each voice's
recording gets its own distinct score, so the per-voice pairing is what
actually gets exercised.
"""
import argparse
import csv
import sys
from collections import defaultdict


def load_scores(path):
    scores = {}
    with open(path) as f:
        for line in f:
            filename, score = line.split()
            scores[filename] = float(score)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-csv", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--task", required=True, help="label only, e.g. lexical/syntactic")
    args = parser.parse_args()

    scores = load_scores(args.submission)

    # pair key -> {0: filename, 1: filename}
    pairs = defaultdict(dict)
    with open(args.gold_csv) as f:
        reader = csv.DictReader(f)
        has_subtype = "type" in reader.fieldnames and "subtype" in reader.fieldnames
        for row in reader:
            if has_subtype:
                key = (row["id"], row["voice"], row["type"], row["subtype"])
            else:
                key = (row["id"], row["voice"])
            pairs[key][int(row["correct"])] = row["filename"]

    n_pairs = 0
    n_incomplete = 0
    n_missing_score = 0
    wins = 0.0
    for key, by_correct in pairs.items():
        if 0 not in by_correct or 1 not in by_correct:
            n_incomplete += 1
            continue
        f_good, f_bad = by_correct[1], by_correct[0]
        if f_good not in scores or f_bad not in scores:
            n_missing_score += 1
            continue
        n_pairs += 1
        s_good, s_bad = scores[f_good], scores[f_bad]
        if s_good > s_bad:
            wins += 1.0
        elif s_good == s_bad:
            wins += 0.5

    if n_pairs == 0:
        print(f"{args.task}: no scoreable pairs found", file=sys.stderr)
        sys.exit(1)

    accuracy = 100.0 * wins / n_pairs
    print(
        f"{args.task}: accuracy={accuracy:.2f}% "
        f"({n_pairs} pairs, {n_incomplete} incomplete pair groups skipped, "
        f"{n_missing_score} pairs missing a score skipped)"
    )


if __name__ == "__main__":
    main()
