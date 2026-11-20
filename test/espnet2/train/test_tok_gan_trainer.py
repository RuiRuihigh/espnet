from dataclasses import fields

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.tok.output import SpeechTokenizerOutput
from espnet2.tok.quantizer.abs_quantizer import (
    AbsSpeechTokenizerQuantizer,
)
from espnet2.tok.tokenizer import SpeechTokenizer
from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.reporter import Reporter
from espnet2.train.tok_gan_trainer import (
    SpeechTokenizerGANTrainer,
    SpeechTokenizerGANTrainerOptions,
)


class DummyFrontend(AbsFrontend):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(1, 2)

    def output_size(self):
        return 2

    def forward(self, input, input_lengths):
        return self.projection(input.unsqueeze(-1)), input_lengths


class DummyQuantizer(AbsSpeechTokenizerQuantizer):
    feature_dim = 2

    def __init__(self):
        super().__init__()
        self.centroids = torch.nn.Parameter(torch.randn(2, 2))

    @property
    def num_clusters(self):
        return 2

    def _output(self, features, lengths):
        logits = features @ self.centroids.transpose(0, 1)
        assignment = logits.softmax(dim=-1)
        token_ids = assignment.argmax(dim=-1)
        return SpeechTokenizerOutput(
            continuous=features,
            assignment=assignment,
            token_ids=token_ids,
            lengths=lengths,
            soft_assignment=assignment,
        )

    def forward(self, features, feature_lengths):
        return self._output(features, feature_lengths)

    def encode(self, features, feature_lengths):
        return self._output(features, feature_lengths)


def test_prepare_epoch_can_reset_main_optimizer():
    tokenizer = SpeechTokenizer(DummyFrontend(), DummyQuantizer(), freeze_epochs=1)
    optimizer = torch.optim.Adam(tokenizer.parameters(), lr=1.0e-3)
    tokenizer.set_epoch(2)
    loss = tokenizer(torch.randn(1, 2), torch.tensor([2])).assignment.square().sum()
    loss.backward()
    optimizer.step()
    assert len(optimizer.state) > 0

    tokenizer.set_epoch(1)
    just_unfroze = SpeechTokenizerGANTrainer.prepare_epoch(
        tokenizer,
        [optimizer],
        epoch=2,
        reset_optimizer_on_unfreeze=True,
    )

    assert just_unfroze
    assert len(optimizer.state) == 0


def test_prepare_epoch_can_reset_both_optimizers():
    model = SpeechTokenizer(DummyFrontend(), DummyQuantizer(), freeze_epochs=1)
    main = torch.optim.Adam(model.parameters(), lr=1e-3)
    discriminator_param = torch.nn.Parameter(torch.ones(()))
    discriminator = torch.optim.Adam([discriminator_param], lr=1e-3)
    for optimizer, parameter in (
        (main, next(model.parameters())),
        (discriminator, discriminator_param),
    ):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        assert len(optimizer.state) > 0

    model.set_epoch(1)

    just_unfroze = SpeechTokenizerGANTrainer.prepare_epoch(
        model=model,
        optimizers=[main, discriminator],
        epoch=2,
        reset_optimizer_on_unfreeze=True,
    )

    assert just_unfroze
    assert len(main.state) == 0
    assert len(discriminator.state) == 0


def test_prosody_logging_preserves_raw_steps_and_epoch_plots(tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tokenizer = SpeechTokenizer(DummyFrontend(), DummyQuantizer(), 30)
            self.generator = torch.nn.Parameter(torch.tensor(1.0))
            self.discriminator = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, forward_generator, signal):
            if not forward_generator:
                return dict(
                    loss=self.discriminator.square(), stats={}, weight=1, optim_idx=1
                )
            stats = {
                "prosody_pitch_loss": signal,
                "prosody_energy_loss": signal + 2,
                "prosody_voicing_loss": signal + 4,
                "asr_loss": signal * 2,
            }
            return dict(loss=self.generator.square(), stats=stats, weight=1, optim_idx=0)

    saved_reporter = Reporter(epoch=1)
    with saved_reporter.observe("train") as sub:
        for _ in range(3):
            sub.register({"asr_loss": 1.0}, weight=1)
            sub.next()
    with saved_reporter.observe("valid") as sub:
        sub.register({"asr_loss": 1.0}, weight=1)
        sub.next()
    reporter = Reporter()
    reporter.load_state_dict(saved_reporter.state_dict())
    reporter.set_epoch(2)
    option_values = {
        field.name: None for field in fields(SpeechTokenizerGANTrainerOptions)
    }
    option_values.update(
        grad_noise=False,
        accum_grad=1,
        grad_clip=0.0,
        log_interval=2,
        no_forward_run=False,
        ngpu=0,
        use_wandb=False,
        generator_first=True,
        skip_discriminator_prob=0.0,
        reset_optimizer_on_unfreeze=False,
    )
    options = SpeechTokenizerGANTrainerOptions(**option_values)
    model = Model()
    optimizers = [
        torch.optim.SGD([parameter], lr=0.01)
        for parameter in (model.generator, model.discriminator)
    ]
    batches = [(["utt"], {"signal": torch.tensor(v)}) for v in (1.0, 3.0, 5.0)]
    with SummaryWriter(tmp_path / "train") as writer, reporter.observe("train") as sub:
        invalid = SpeechTokenizerGANTrainer.train_one_epoch(
            model,
            batches,
            optimizers,
            [None, None],
            None,
            sub,
            writer,
            options,
            DistributedOption(),
        )
    assert not invalid

    events = EventAccumulator(str(tmp_path / "train")).Reload()
    prosody_keys = ("prosody_pitch_loss", "prosody_energy_loss", "prosody_voicing_loss")
    for offset, key in enumerate(prosody_keys):
        raw = events.Scalars(f"step/{key}")
        assert [(event.step, event.value) for event in raw] == [
            (4, 1 + offset * 2),
            (5, 3 + offset * 2),
            (6, 5 + offset * 2),
        ]
    assert "step/asr_loss" not in events.Tags()["scalars"]
    assert [(e.step, e.value) for e in events.Scalars("asr_loss")] == [(5, 4.0)]
    assert [(e.step, e.value) for e in events.Scalars("prosody_pitch_loss")] == [(5, 2.0)]

    with reporter.observe("valid") as sub:
        sub.register({key: 2.0 for key in prosody_keys}, weight=1)
        sub.next()
    plot_dir = tmp_path / "images"
    reporter.matplotlib_plot(plot_dir)
    assert all((plot_dir / f"{key}.png").stat().st_size > 0 for key in prosody_keys)
