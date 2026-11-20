#!/usr/bin/env python3
"""Prepare an isolated epoch-30 resume checkpoint with a warmed prosody head."""

import argparse
import math
from pathlib import Path

import torch
import yaml

from espnet2.tasks.gan_tok import GANSpeechTokenizerTask
from warmup_prosody import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch30-run', type=Path, required=True)
    parser.add_argument('--head', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prosody-weight', type=float, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.prosody_weight) or args.prosody_weight <= 0:
        parser.error('--prosody-weight must be finite and positive')

    baseline = args.epoch30_run
    init = torch.load(args.head, map_location='cpu', weights_only=True)
    config_path, weights_path = baseline / 'config.yaml', baseline / '30epoch.pth'
    settings = init['settings']
    if (settings['source_epoch'] != 30
            or settings['source_config_sha256'] != sha256(config_path)
            or settings['source_sha256'] != sha256(weights_path)):
        raise ValueError('The warmup head and epoch-30 source do not match')
    source = torch.load(baseline / 'checkpoint.pth', map_location='cpu', mmap=True, weights_only=True)
    if source['reporter']['epoch'] != 30 or source['model']['tokenizer.current_epoch'].item() != 30:
        raise ValueError('The source continuation checkpoint must be at epoch 30')
    if len(source['optimizers']) != 2 or len(source['schedulers']) != 2 or any(
        scheduler is None for scheduler in source['schedulers']
    ):
        raise ValueError('The source must contain both GAN optimizers and schedulers')
    weights = torch.load(weights_path, map_location='cpu', mmap=True, weights_only=True)
    if weights.keys() != source['model'].keys() or any(
        not torch.equal(tensor, source['model'][key]) for key, tensor in weights.items()
    ):
        raise ValueError('The epoch-30 weights and continuation checkpoint differ')
    del weights

    config = yaml.safe_load(config_path.read_text())
    config.update(max_epoch=90, prosody_weight=args.prosody_weight, prosody_conf={
        'hidden_dim': settings['hidden_dim'], 'kernel_size': settings['kernel_size'], 'ramp_steps': 500,
    })
    model_args = argparse.Namespace(**config)
    model = GANSpeechTokenizerTask.build_model(model_args)
    incompatible = model.load_state_dict(source['model'], strict=False)
    objective = model.objectives['prosody']
    if (incompatible.unexpected_keys or set(incompatible.missing_keys) != {
            f'objectives.prosody.{key}' for key in objective.state_dict()}):
        raise ValueError(f'Unexpected difference between joint and source models: {incompatible}')
    objective.head.load_state_dict(init['head'], strict=True)

    optimizers = GANSpeechTokenizerTask.build_optimizers(model_args, model)
    states = [optimizer.state_dict() for optimizer in optimizers]
    # Start with the new parameter layout and fresh Adam moments. The baseline
    # trainer also clears moments at the 30-to-31 transition.
    for fresh, previous in zip(states, source['optimizers']):
        if len(fresh['param_groups']) != len(previous['param_groups']):
            raise ValueError('Unexpected optimizer parameter group change')
        for group, old_group in zip(fresh['param_groups'], previous['param_groups']):
            group.update({k: v for k, v in old_group.items() if k != 'params'})

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'prosody_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    temp = args.output / 'checkpoint.pth.partial'
    torch.save(dict(source, model=model.state_dict(), optimizers=states), temp)
    temp.replace(args.output / 'checkpoint.pth')
    print(f'Prepared epoch-30 prosody continuation in {args.output}')


if __name__ == '__main__':
    main()
