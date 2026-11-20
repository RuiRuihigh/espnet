import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from espnet2.tok.objective.prosody import ProsodyHead, ProsodyObjective, prosody_loss
from espnet2.tok.prosody_targets import extract_prosody


class ProsodyTests(unittest.TestCase):
    def test_target_alignment_and_mask(self):
        samples = np.arange(16037)
        tone = 0.1 * np.sin(2 * np.pi * 150 * samples / 16000)
        targets = extract_prosody(tone, 16000)
        self.assertEqual(targets.shape, (51, 6))
        self.assertTrue(np.isfinite(targets).all())
        self.assertTrue(targets[12:35, 3].all())
        self.assertLess(np.abs(targets[12:35, 0]).max(), 0.1)
        self.assertFalse(targets[-2:, 5].any())
        unvoiced = extract_prosody(np.zeros(8000), 16000)
        self.assertFalse(unvoiced[:, 3].any())
        self.assertTrue(unvoiced[:-1, 4].all())

    def test_radius_and_padding(self):
        torch.manual_seed(1)
        head = ProsodyHead(8, hidden_dim=4)
        inputs = torch.randn(1, 17, 8, requires_grad=True)
        valid = torch.ones(1, 17, dtype=torch.bool)
        valid[0, 12] = False
        head(inputs, valid)[0, 8].sum().backward()
        influence = inputs.grad.abs().sum(dim=-1)[0] > 0
        expected = torch.zeros(17, dtype=torch.bool)
        expected[3:14] = True
        expected[12] = False
        torch.testing.assert_close(influence, expected)

    def test_joint_gradients_and_length_audit(self):
        torch.manual_seed(1)
        objective = ProsodyObjective(8, hidden_dim=4, ramp_steps=2)
        logits = torch.randn(2, 15, 4, requires_grad=True)
        soft = logits.softmax(-1)
        hard = F.one_hot(soft.argmax(-1), 4).float()
        output = SimpleNamespace(assignment=hard - soft.detach() + soft,
                                 lengths=torch.tensor([15, 12]))
        codebook = torch.randn(4, 8, requires_grad=True)
        targets = torch.rand(2, 15, 6)
        targets[..., 2:] = 1
        loss, stats, _ = objective(output, codebook=codebook, prosody=targets,
                                   prosody_lengths=output.lengths)
        self.assertEqual(stats['ramp'].item(), .5)
        loss.backward()
        for grad in (logits.grad, codebook.grad, objective.head.output.weight.grad):
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0)
        with self.assertRaisesRegex(ValueError, 'lengths disagree'):
            objective(output, codebook=codebook, prosody=targets,
                      prosody_lengths=torch.tensor([15, 13]))

    def test_no_voiced_frames(self):
        prediction = torch.zeros(1, 4, 3, requires_grad=True)
        targets = torch.zeros(1, 4, 6)
        targets[..., 0] = 100
        targets[..., 4:] = 1
        loss, stats = prosody_loss(prediction, targets, torch.ones(1, 4, dtype=torch.bool))
        self.assertEqual(stats['pitch_loss'].item(), 0.)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())


if __name__ == '__main__':
    unittest.main()
