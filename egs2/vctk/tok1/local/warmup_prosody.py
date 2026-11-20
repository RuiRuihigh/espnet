#!/usr/bin/env python3
"""Train and validate only the new prosody head on epoch-30 hard tokens."""

import argparse
import hashlib
import json
import math
import random
import subprocess
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from espnet2.bin.tok_inference import build_tokenizer
from espnet2.tok.objective.prosody import ProsodyHead, prosody_loss, prosody_masks
from espnet2.tok.prosody_targets import extract_prosody
from prepare_prosody import read_audio, read_wavs


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def cache_records(tokenizer, wavs, keys, batch_size, device, split):
    records = []
    for start in range(0, len(keys), batch_size):
        selected = keys[start:start + batch_size]
        signals = [read_audio(wavs[key]) for key in selected]
        waves = [torch.from_numpy(wave) for wave, _ in signals]
        lengths = torch.tensor([len(wave) for wave in waves], device=device)
        with torch.no_grad():
            output = tokenizer.encode(pad_sequence(waves, batch_first=True).to(device), lengths)
            ids = output.token_ids.cpu()
            token_lengths = output.lengths.tolist()
        for offset, ((wave, rate), length) in enumerate(zip(signals, token_lengths)):
            targets = torch.from_numpy(extract_prosody(wave, rate))
            if len(targets) != length or not targets[:, 5].any():
                raise ValueError(f'Aligned targets and token count disagree for {selected[offset]}')
            records.append((ids[offset, :length].clone(), targets))
        if start == 0 or start // batch_size % 32 == 0 or len(records) == len(keys):
            print(f'PROSODY_CACHE {split} {len(records)}/{len(keys)}', flush=True)
    return records


def batch(records, indices, device):
    selected = [records[i] for i in indices]
    ids, targets = zip(*selected)
    ids = pad_sequence(ids, batch_first=True).to(device)
    targets = pad_sequence(targets, batch_first=True).to(device)
    lengths = torch.tensor([len(r[0]) for r in selected], device=device)
    valid = torch.arange(ids.shape[1], device=device)[None] < lengths[:, None]
    valid &= targets[..., 5].bool()
    return ids, targets, valid


@torch.no_grad()
def evaluate(head, baseline, codebook, records, batch_size):
    totals = torch.zeros(11, dtype=torch.float64, device=codebook.device)
    for start in range(0, len(records), batch_size):
        idx = range(start, min(start + batch_size, len(records)))
        ids, targets, valid = batch(records, idx, codebook.device)
        if head is None:
            prediction = baseline.expand_as(targets[..., :3])
        else:
            prediction = head(F.embedding(ids, codebook), valid)
        pitch, energy, voicing = prosody_masks(targets, valid)
        _, losses = prosody_loss(prediction, targets, valid)
        ref_voiced = targets[..., 2].bool() & voicing
        ref_unvoiced = ~targets[..., 2].bool() & voicing
        pred_voiced = prediction[..., 2] >= 0
        totals += torch.stack((
            losses['pitch_loss'] * pitch.sum(), pitch.sum(),
            losses['energy_loss'] * energy.sum(), energy.sum(),
            losses['voicing_loss'] * voicing.sum(), voicing.sum(),
            ((prediction[..., 0] - targets[..., 0]).abs() * pitch).sum(),
            ((prediction[..., 1] * 3 - targets[..., 1]).abs() * energy).sum(),
            (pred_voiced & ref_voiced).sum(), (~pred_voiced & ref_unvoiced).sum(),
            ref_voiced.sum(),
        )).to(totals)
    pl, pn, el, en, vl, vn, pa, ea, tp, tn, vp = totals.tolist()
    if min(pn, en, vp, vn - vp) <= 0:
        raise ValueError('Validation requires valid pitch, energy and both voicing classes')
    return {
        'loss': pl/pn + el/en + vl/vn,
        'pitch_mae_st': pa/pn,
        'energy_mae_db': ea/en,
        'voicing_balanced_accuracy': 0.5 * (tp/vp + tn/(vn - vp)),
        'pitch_frames': int(pn), 'energy_frames': int(en), 'voicing_frames': int(vn),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-config', type=Path, required=True)
    parser.add_argument('--model-file', type=Path, required=True)
    parser.add_argument('--train-wav-scp', type=Path, required=True)
    parser.add_argument('--valid-wav-scp', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-utterances', type=int, default=4096)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--encode-batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    if min(args.steps, args.train_utterances, args.batch_size, args.encode_batch_size) < 1:
        parser.error('positive sizes are required')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('learning rate must be finite and positive')
    args.output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')

    train_wavs, valid_wavs = read_wavs(args.train_wav_scp), read_wavs(args.valid_wav_scp)
    if train_wavs.keys() & valid_wavs.keys():
        raise ValueError('Training and validation utterance IDs overlap')
    train_keys = sorted(rng.sample(sorted(train_wavs), min(len(train_wavs), args.train_utterances)))
    valid_keys = sorted(valid_wavs)
    source_hash = sha256(args.model_file)
    tokenizer = build_tokenizer(str(args.train_config), str(args.model_file), args.device, 'float32')
    tokenizer.requires_grad_(False)
    if tokenizer.current_epoch.item() != 30 or tokenizer.num_clusters != 2000:
        raise ValueError('Expected the full-precision, 2,000-code epoch-30 tokenizer')
    codebook = tokenizer.quantizer.centroids.detach().clone()
    train = cache_records(tokenizer, train_wavs, train_keys, args.encode_batch_size, args.device, 'train')
    valid = cache_records(tokenizer, valid_wavs, valid_keys, args.encode_batch_size, args.device, 'dev')
    del tokenizer
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    head = ProsodyHead(codebook.size(1)).to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate)
    voiced = sum(targets[:, 2][targets[:, 4].bool()].sum().item() for _, targets in train)
    counted = sum(targets[:, 4].sum().item() for _, targets in train)
    baseline = codebook.new_tensor([0., 0., 0.])
    baseline[2] = codebook.new_tensor(voiced / counted).clamp(.001, .999).logit()

    settings = {'version': 1, 'source_epoch': 30, 'source_sha256': source_hash,
                'source_config_sha256': sha256(args.train_config),
                'head_input_dim': codebook.size(1), 'hidden_dim': 128, 'kernel_size': 11,
                'seed': args.seed, 'batch_size': args.batch_size, 'learning_rate': args.learning_rate,
                'steps': args.steps, 'train_utterances': len(train), 'valid_utterances': len(valid),
                'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    (args.output / 'train_ids.txt').write_text('\n'.join(train_keys) + '\n')
    (args.output / 'valid_ids.txt').write_text('\n'.join(valid_keys) + '\n')
    trivial = evaluate(None, baseline, codebook, valid, args.batch_size)
    head.eval()
    initial = evaluate(head, baseline, codebook, valid, args.batch_size)
    print('PROSODY_BASELINES', json.dumps({'constant': trivial, 'initial_head': initial}), flush=True)
    history = []
    best = None
    order = list(range(len(train)))
    rng.shuffle(order)
    cursor = 0
    for step in range(1, args.steps + 1):
        if cursor == len(order):
            rng.shuffle(order)
            cursor = 0
        ids, targets, mask = batch(train, order[cursor:cursor + args.batch_size], args.device)
        cursor = min(cursor + args.batch_size, len(order))
        head.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = prosody_loss(head(F.embedding(ids, codebook), mask), targets, mask)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            head.eval()
            metrics = dict(evaluate(head, baseline, codebook, valid, args.batch_size), step=step,
                           train_loss=loss.item(), grad_norm=grad.item())
            history.append(metrics)
            print('PROSODY_VALID', json.dumps(metrics, allow_nan=False), flush=True)
            if best is None or metrics['loss'] < best['loss']:
                best = metrics
                temp = args.output / 'prosody_head.pth.partial'
                torch.save({'head': head.state_dict(), 'settings': settings, 'step': step}, temp)
                temp.replace(args.output / 'prosody_head.pth')

    summary = dict(settings=settings, constant_baseline=trivial, initial_head=initial,
                   best=best, validations=history)
    temp = args.output / 'metrics.json.partial'
    temp.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    temp.replace(args.output / 'metrics.json')
    print('PROSODY_WARMUP_COMPLETE', json.dumps(summary, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
