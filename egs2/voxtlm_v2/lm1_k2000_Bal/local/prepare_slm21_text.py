#!/usr/bin/env python3
"""Turn a sLM21 (sWUGGY/sBLIMP) gold.csv into ESPnet LM input + a filename map.

The paper scores sWUGGY/sBLIMP items with the trained LM's own sequence
log-likelihood (higher/less-negative = "more probable" = the model's answer
for that item), matching the zerospeech "sLM21" benchmark's submission
format: one pseudo-log-probability per item, keyed by its .wav filename.

Many rows in gold.csv are the same word/sentence recorded by different
voices -- for the *text* track the voice doesn't matter, so this only scores
each distinct text once (dedup by lowercased text) and writes a filename map
to expand that one score back out to every filename that shares it.

Outputs, under --out-dir:
  text_dedup.txt          ESPnet text format: "<text_id> <generatetext> <text>"
                           (feed straight to espnet2.bin.lm_calc_perplexity
                           as --data_path_and_name_and_type, or lm_inference).
  filename_map.tsv         "<text_id>\t<filename1>,<filename2>,..."

--task selects which gold.csv column holds the text to score: 'lexical'
(sWUGGY, real word / phonotactic non-word pairs) uses the "word" column,
falling back to "phones" for any row where "word" is empty (matches the
official zerospeech scoring code's own fallback for non-words with no
orthographic form); 'syntactic' (sBLIMP) uses "transcription". Transcription
text there ships upper-cased; this recipe's training text is lower-cased
(LibriSpeech-derived), so it's lower-cased here to match -- otherwise every
sentence would score as if made of mostly-unseen characters.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold-csv", type=Path, required=True)
    parser.add_argument("--task", required=True, choices=["lexical", "syntactic"])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--sos-sym",
        default="<generatetext>",
        help="Prefix marking this as an (unconditional) text-generation "
        "sequence, matching lm_train.txt's own textlm-task lines "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--id-prefix",
        default="txt",
        help="Prefix for the synthetic per-unique-text ids (default: %(default)s).",
    )
    args = parser.parse_args()

    text_of_filename = {}
    with args.gold_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            filename = row["filename"]
            if args.task == "lexical":
                text = row["word"].strip() or row["phones"].strip()
            else:
                text = row["transcription"].strip().lower()
            if not text:
                raise ValueError(f"{args.gold_csv}: row {row!r} has no usable text")
            text_of_filename[filename] = text

    filenames_of_text = defaultdict(list)
    for filename, text in text_of_filename.items():
        filenames_of_text[text].append(filename)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    text_dedup_path = args.out_dir / "text_dedup.txt"
    filename_map_path = args.out_dir / "filename_map.tsv"

    n_digits = len(str(len(filenames_of_text)))
    with text_dedup_path.open("w") as f_text, filename_map_path.open("w") as f_map:
        for i, (text, filenames) in enumerate(sorted(filenames_of_text.items())):
            text_id = f"{args.id_prefix}{i:0{n_digits}d}"
            f_text.write(f"{text_id} {args.sos_sym} {text}\n")
            f_map.write(f"{text_id}\t{','.join(filenames)}\n")

    print(
        f"{args.gold_csv}: {len(text_of_filename)} rows -> "
        f"{len(filenames_of_text)} unique '{args.task}' texts "
        f"-> {text_dedup_path}, {filename_map_path}"
    )


if __name__ == "__main__":
    main()
