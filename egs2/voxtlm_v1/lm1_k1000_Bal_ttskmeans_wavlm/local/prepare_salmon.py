#!/usr/bin/env python3
"""Unpack the SALMon HF parquet files into wavs + one kaldi-style datadir per part.

SALMon (https://github.com/slp-rl/salmon, HF dataset slprl/SALMon) ships each
part as one parquet with explicit `positive_audio` / `negative_audio` columns,
so pos/neg is read from the column name, never inferred from file-name sort
order (the official salmon.py's local-folder loader relies on sorted
"sample_<i>_<j>.wav" names; the parquet route avoids that ambiguity).

All parts are already 16 kHz mono, so no resampling / format_wav_scp.sh step is
needed -- this writes wav.scp, utt2num_samples, utt2spk, spk2utt directly, which
is everything scripts/feats/perform_kmeans.sh stage 3 reads.

The 8 parts go into just two datadirs, salmon_consistency (6 parts) and
salmon_alignment (2 parts): perform_kmeans.sh stage 3 needs a non-empty
train_set and dev_set and runs one GPU array job (one WavLM-large load) per
datadir, so two datadirs = two jobs instead of eight serial ones. The part is
recovered from the uttid prefix at scoring time.

uttid = "<part>-<ind:04d>-<pos|neg>"; utt2spk maps each uttid to itself (same
convention as the sLM21 speech track).

Output:
  <wav-dir>/<part>/<uttid>.wav
  <data-dir>/salmon_{consistency,alignment}/{wav.scp,utt2num_samples,utt2spk,spk2utt}
"""

import argparse
import io
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

GROUPS = {
    "salmon_consistency": [
        "bg_all_consistency",
        "bg_domain_consistency",
        "gender_consistency",
        "rir_consistency",
        "sentiment_consistency",
        "speaker_consistency",
    ],
    "salmon_alignment": [
        "bg_alignment",
        "sentiment_alignment",
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-dir", type=Path, required=True,
                        help="snapshot_download of slprl/SALMon (contains <part>/*.parquet).")
    parser.add_argument("--wav-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--fs", type=int, default=16000)
    parser.add_argument("--read-only", action="store_true",
                        help="Never write wavs: fail if one is missing. For recipes whose "
                        "downloads/SALMon is a symlink to another recipe's copy.")
    args = parser.parse_args()

    for group, parts in GROUPS.items():
        entries = []
        for part in parts:
            parquets = sorted((args.hf_dir / part).glob("*.parquet"))
            if not parquets:
                raise FileNotFoundError(f"no parquet under {args.hf_dir / part}")
            wav_dir = args.wav_dir / part
            wav_dir.mkdir(parents=True, exist_ok=True)
            n_before = len(entries)
            for parquet in parquets:
                for row in pq.read_table(parquet).to_pylist():
                    for label, col in (("pos", "positive_audio"), ("neg", "negative_audio")):
                        uttid = f"{part}-{row['ind']:04d}-{label}"
                        audio, sr = sf.read(io.BytesIO(row[col]["bytes"]), dtype="int16")
                        if sr != args.fs or audio.ndim != 1:
                            raise ValueError(f"{uttid}: expected {args.fs} Hz mono, got {sr} Hz shape {audio.shape}")
                        wav = (wav_dir / f"{uttid}.wav").resolve()
                        if not wav.exists():
                            if args.read_only:
                                raise FileNotFoundError(f"{wav} missing and --read-only is set")
                            sf.write(wav, audio, sr, subtype="PCM_16")
                        entries.append((uttid, wav, len(audio)))
            print(f"{part}: {(len(entries) - n_before) // 2} pairs")

        entries.sort()
        uttids = [e[0] for e in entries]
        if len(set(uttids)) != len(uttids):
            raise ValueError(f"{group}: duplicate uttids (repeated 'ind' in parquet?)")
        data_dir = args.data_dir / group
        data_dir.mkdir(parents=True, exist_ok=True)
        with (data_dir / "wav.scp").open("w") as f:
            f.writelines(f"{u} {w}\n" for u, w, _ in entries)
        with (data_dir / "utt2num_samples").open("w") as f:
            f.writelines(f"{u} {n}\n" for u, _, n in entries)
        for name in ("utt2spk", "spk2utt"):
            with (data_dir / name).open("w") as f:
                f.writelines(f"{u} {u}\n" for u in uttids)
        print(f"{group}: {len(entries)} utts -> {data_dir}")


if __name__ == "__main__":
    main()
