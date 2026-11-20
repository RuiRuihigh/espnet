"""A token-only auxiliary objective for frame-level relative prosody."""

import torch
from torch import nn
from torch.nn import functional as F

from espnet2.tok.objective.abs_objective import AbsSpeechTokenizerObjective
from espnet2.tok.prosody_targets import TARGET_COLUMNS


class ProsodyHead(nn.Module):
    """Predict the current frame from its token and five neighbors on each side."""

    def __init__(self, input_dim, hidden_dim=128, kernel_size=11):
        super().__init__()
        if min(input_dim, hidden_dim, kernel_size) < 1 or kernel_size % 2 == 0:
            raise ValueError('Positive dimensions and an odd kernel size are required')
        self.project = nn.Linear(input_dim, hidden_dim)
        self.context = nn.Conv1d(hidden_dim, hidden_dim, kernel_size,
                                 padding=kernel_size // 2)
        self.output = nn.Linear(hidden_dim, 3)

    def forward(self, vectors, valid):
        hidden = self.project(vectors).masked_fill(~valid.unsqueeze(-1), 0)
        hidden = F.gelu(self.context(hidden.transpose(1, 2)).transpose(1, 2))
        return self.output(hidden)


def prosody_masks(targets, valid):
    """Separate alignment padding, pitch validity and voicing validity."""
    if targets.ndim != 3 or targets.shape[-1] != TARGET_COLUMNS:
        raise ValueError('Expected prosody targets of shape (B, T, 6)')
    energy_mask = valid & targets[..., 5].bool()
    return energy_mask & targets[..., 3].bool(), energy_mask, energy_mask & targets[..., 4].bool()


def prosody_loss(prediction, targets, valid):
    masks = prosody_masks(targets, valid)
    target = targets[..., :3]
    values = (
        F.smooth_l1_loss(prediction[..., 0], target[..., 0], reduction='none'),
        F.smooth_l1_loss(prediction[..., 1], target[..., 1] / 3.0, reduction='none'),
        F.binary_cross_entropy_with_logits(prediction[..., 2], target[..., 2], reduction='none'),
    )
    losses = [(value * mask).sum() / mask.sum().clamp_min(1)
              for value, mask in zip(values, masks)]
    return sum(losses), dict(zip(('pitch_loss', 'energy_loss', 'voicing_loss'),
                                 (loss.detach() for loss in losses)))


class ProsodyObjective(AbsSpeechTokenizerObjective):
    """Use hard assignment values with straight-through gradients to the tokenizer."""

    def __init__(self, input_dim, hidden_dim=128, kernel_size=11, ramp_steps=500):
        super().__init__()
        if ramp_steps < 0:
            raise ValueError('ramp_steps must be non-negative')
        self.head = ProsodyHead(input_dim, hidden_dim, kernel_size)
        self.ramp_steps = ramp_steps
        self.register_buffer('joint_updates', torch.zeros((), dtype=torch.long))

    def forward(self, tokenizer_output, *, codebook, prosody, prosody_lengths, **batch):
        assignment = tokenizer_output.assignment
        lengths = tokenizer_output.lengths
        if assignment.shape[:2] != prosody.shape[:2] or not torch.equal(lengths, prosody_lengths):
            raise ValueError('Prosody labels and tokenizer lengths disagree; check waveform alignment')
        valid = torch.arange(assignment.shape[1], device=assignment.device)[None] < lengths[:, None]
        vectors = assignment @ codebook.to(assignment)
        prediction = self.head(vectors, valid & prosody[..., 5].bool())
        loss, stats = prosody_loss(prediction, prosody, valid)
        ramp = loss.new_ones(())
        if self.training and self.ramp_steps and codebook.requires_grad:
            self.joint_updates.add_(1)
            ramp = (self.joint_updates.to(loss) / self.ramp_steps).clamp_max(1.0)
        stats['ramp'] = ramp.detach()
        stats['loss'] = loss.detach()
        return loss * ramp, stats, assignment.size(0)
