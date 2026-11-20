#!/usr/bin/env python3
"""Write deterministic frame-level prosody targets for prepared VCTK waveforms."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import kaldiio
import soundfile as sf

from espnet2.tok.prosody_targets import extract_prosody


def read_wavs(path):
    records = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        key, wav = line.split(maxsplit=1)
        if key in records or wav.endswith('|'):
            raise ValueError(f'Expected unique IDs and materialized waveforms: {line}')
        records[key] = wav
    if not records:
        raise ValueError(f'Empty waveform list: {path}')
    return records


def read_audio(path):
    return sf.read(path, dtype='float32')


def extract_record(item):
    key, path = item
    return key, extract_prosody(*read_audio(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wav_scp', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = read_wavs(args.wav_scp)
    archive = args.output_dir / 'prosody.ark'
    pending_index = args.output_dir / 'prosody.scp.partial'
    with ProcessPoolExecutor(args.workers) as pool, \
            kaldiio.WriteHelper(f'ark,scp:{archive},{pending_index}') as writer:
        for i, (key, targets) in enumerate(pool.map(extract_record, records.items(), chunksize=16), 1):
            writer(key, targets)
            if i % 1000 == 0:
                print(f'PROSODY_LABELS {i}/{len(records)}', flush=True)
    pending_index.replace(args.output_dir / 'prosody.scp')
    print(f'PROSODY_LABELS_COMPLETE {len(records)}', flush=True)


if __name__ == '__main__':
    main()
