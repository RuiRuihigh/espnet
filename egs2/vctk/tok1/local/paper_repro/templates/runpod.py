"""Foreground single-node training on an already provisioned Pod; no cloud API calls."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def atomic_json(path, value):
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def signature(root, ngpu, batch_bins):
    paths = [
        root / 'espnet/egs2/vctk/tok1/conf/train_paper.yaml',
        root / 'libri_data/libri30h/manifest.json',
        root / 'initialization/km_2000.mdl',
        root / 'DATA_READY.json',
    ]
    return {
        'ngpu': ngpu,
        'batch_bins': batch_bins,
        'accum_grad': 1,
        'commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=root / 'espnet', text=True
        ).strip(),
        'sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in paths},
    }


def validate_checkpoint(checkpoint):
    """Validate a trusted local full trainer checkpoint, not a weights-only export."""
    epoch = checkpoint['reporter']['epoch']
    if not isinstance(epoch, int) or not 1 <= epoch <= 90:
        raise ValueError('Invalid checkpoint epoch')
    if 'model' not in checkpoint or len(checkpoint['optimizers']) != 2:
        raise ValueError('Expected model and both GAN optimizer states')
    if not all(o['state'] for o in checkpoint['optimizers']):
        raise ValueError('Empty optimizer state: cannot faithfully resume')
    if len(checkpoint['schedulers']) != 2 or any(s is None for s in checkpoint['schedulers']):
        raise ValueError('Expected both scheduler states')
    return epoch


def train_command(root, ngpu, batch_bins, target, output, cudnn_benchmark=None):
    tok_args = f'--max_epoch {target} --batch_bins {batch_bins} --accum_grad 1'
    if cudnn_benchmark is not None:
        tok_args += f' --cudnn_benchmark {cudnn_benchmark}'
    return [str(root / 'recipe.sh'), '--stage', '9', '--stop_stage', '9',
            '--ngpu', str(ngpu), '--num_nodes', '1',
            '--python', str(root / 'python.sh'), '--tok_exp', str(output),
            '--tok_args', tok_args]


def check_resume(action, output, expected, load):
    checkpoint = output / 'checkpoint.pth'
    metadata = output / 'RUNPOD_RUN.json'
    if action == 'start':
        if checkpoint.exists() or metadata.exists():
            raise ValueError('Run already exists; use resume, or choose a new --name')
        return 0
    if not checkpoint.is_file() or not metadata.is_file():
        raise ValueError('Resume requires checkpoint.pth and RUNPOD_RUN.json')
    if json.loads(metadata.read_text()) != expected:
        raise ValueError('Config, code, data, GPU count or batch changed; refusing resume')
    return validate_checkpoint(load(checkpoint))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'start', 'resume'])
    parser.add_argument('--ngpu', type=int, choices=[1, 2], required=True)
    parser.add_argument('--batch-bins', type=int, default=16000000,
                        help='Global batch budget; lowering it changes the experiment')
    parser.add_argument('--cudnn-benchmark', choices=('true', 'false'),
                        help='Override cuDNN convolution autotuning in preflight and training')
    parser.add_argument('--target-epoch', type=int, default=90,
                        help='Absolute last epoch, not number of additional epochs')
    parser.add_argument('--name', default='runpod', help='Separate output directory name')
    args = parser.parse_args()
    if not 1 <= args.target_epoch <= 90 or args.batch_bins <= 0:
        parser.error('target epoch must be 1..90 and batch-bins must be positive')
    if not args.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.name):
        parser.error('name must contain only letters, numbers, underscore or hyphen')
    if (ROOT / 'STOPPED_BY_USER.json').exists():
        raise RuntimeError('STOPPED_BY_USER.json exists; no training started')
    for marker in ['SETUP_OK.json', 'DATA_READY.json', 'stages/stats.done']:
        if not (ROOT / marker).is_file():
            raise RuntimeError(f'Missing prerequisite: {marker}')
    # Source edits to the model/trainer invalidate this workflow.
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', 'espnet2', 'egs2/TEMPLATE'],
                   cwd=ROOT / 'espnet', check=True)
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < args.ngpu:
        raise RuntimeError(f'Need {args.ngpu} visible CUDA GPUs')
    output = ROOT / 'train_runs' / args.name
    output.mkdir(parents=True, exist_ok=True)
    # Share the lock across names: avoid accidentally running two jobs on the same GPUs.
    with (ROOT / 'runpod.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Keep the lock alive in the subprocess if this wrapper is interrupted.
        pass_fds = (lock.fileno(),)
        expected = signature(ROOT, args.ngpu, args.batch_bins)
        audit_name = f'preflight_runpod_{args.ngpu}gpu_{args.batch_bins}'
        preflight_flags = []
        if args.cudnn_benchmark is not None:
            expected.update(cudnn_benchmark=args.cudnn_benchmark == 'true', torch_version=str(torch.__version__))
            audit_name += f'_cudnn_benchmark_{args.cudnn_benchmark}'
            preflight_flags = ['--cudnn-benchmark', args.cudnn_benchmark]
        audit = ROOT / audit_name
        if args.action == 'preflight':
            audit.mkdir(exist_ok=True)
            (audit / 'READY.json').unlink(missing_ok=True)
            for rank in range(args.ngpu):
                env = dict(os.environ)
                visible = env.get('CUDA_VISIBLE_DEVICES')
                devices = visible.split(',') if visible else [str(i) for i in range(torch.cuda.device_count())]
                env['CUDA_VISIBLE_DEVICES'] = devices[rank]
                subprocess.run(
                    [sys.executable,
                     str(ROOT / 'preflight.py'), '--ngpu', str(args.ngpu),
                     '--rank', str(rank), '--batch-bins', str(args.batch_bins),
                     '--output', str(audit / f'rank{rank}'), *preflight_flags],
                    check=True, env=env, pass_fds=pass_fds,
                )
                report = json.loads((audit / f'rank{rank}/COMPLETE.json').read_text())
                if not report['passed'] or report['training_gpus'] != args.ngpu or report['preflight_rank'] != rank:
                    raise RuntimeError('Preflight report does not match requested rank')
            atomic_json(audit / 'READY.json', expected)
            print('All rank samples passed. This is not an NCCL/DDP integration test.')
            return
        if not (audit / 'READY.json').is_file() or json.loads((audit / 'READY.json').read_text()) != expected:
            raise RuntimeError('Run matching preflight first; code/config/data changes invalidate it')
        def load(path):
            return torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        epoch = check_resume(args.action, output, expected, load)
        if epoch >= args.target_epoch:
            print(f'Checkpoint already at epoch {epoch}; no training started')
            return
        if args.action == 'start':
            atomic_json(output / 'RUNPOD_RUN.json', expected)
        print(f'Training epochs {epoch + 1}..{args.target_epoch}; output: {output}', flush=True)
        subprocess.run(train_command(ROOT, args.ngpu, args.batch_bins, args.target_epoch,
                                     output, args.cudnn_benchmark),
                       check=True, pass_fds=pass_fds,
                       env={**os.environ, 'PAPER_TRAIN_PYTHON': sys.executable})
        completed = validate_checkpoint(load(output / 'checkpoint.pth'))
        if completed != args.target_epoch:
            raise RuntimeError(f'Expected epoch {args.target_epoch}, got {completed}')
        atomic_json(output / 'PROGRESS.json', {'epoch': completed, 'target_epoch': args.target_epoch})
        print(f'Completed epoch {completed}. No Pod was stopped or provisioned.')


if __name__ == '__main__':
    main()
