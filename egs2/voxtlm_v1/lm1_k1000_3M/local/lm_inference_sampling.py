#!/usr/bin/env python3
"""espnet2.bin.lm_inference with optional top-p (nucleus) SAMPLING -- WITHOUT
editing the shared espnet2 source.

The stock LM inference only has beam search. Two decoding modes are reached
through the SAME code path by choosing beam_size in the decode yaml plus env vars:

  greedy          beam_size: 1, no env vars. Beam search with one beam IS greedy
                  decoding (argmax every step); this wrapper is a no-op.
  top-p sampling  beam_size: 1 plus SAMPLE_MODE=topp. Each step the LM's next-token
                  distribution is temperature-scaled, truncated to the smallest set of
                  tokens whose cumulative probability reaches SAMPLE_TOP_P, and a token
                  is drawn from it. Implemented with the Gumbel-max trick: argmax of
                  (truncated log-probs + Gumbel noise) is an exact sample from the
                  truncated distribution, so the unmodified beam search (beam 1) does the
                  drawing and everything downstream (files, sharding, EOS handling,
                  maxlen) is identical to a normal run.

Env vars: SAMPLE_MODE=topp | (unset), SAMPLE_TOP_P (default 0.9),
          SAMPLE_TEMP (default 1.0), SAMPLE_SEED (default 0),
          SAMPLE_PER_UTT_SEED=1  re-seed before EVERY sentence from (SAMPLE_SEED, crc32 of the
          prompt token ids), so each sentence's sample depends only on the sentence itself --
          not on how the sentences were sharded across jobs or in what order they ran.
EOS is an ordinary token in the distribution, so it can be sampled; top-p cuts the
low-probability tail, which is what stops spurious mid-sentence EOS.
"""
import logging
import os
import zlib

import torch

import espnet2.bin.lm_inference as li
from espnet2.legacy.nets.scorer_interface import BatchScorerInterface

SAMPLE_MODE = os.environ.get("SAMPLE_MODE", "")
TOP_P = float(os.environ.get("SAMPLE_TOP_P", "0.9"))
TEMP = float(os.environ.get("SAMPLE_TEMP", "1.0"))
SEED = int(os.environ.get("SAMPLE_SEED", "0"))
PER_UTT_SEED = os.environ.get("SAMPLE_PER_UTT_SEED", "") == "1"
NEG = -1.0e9


def top_p_gumbel(logp: torch.Tensor, top_p: float, temp: float) -> torch.Tensor:
    """logp: (B, V) log-probs -> (B, V) scores whose argmax is a top-p sample."""
    if temp != 1.0:
        logp = (logp / temp).log_softmax(dim=-1)
    if top_p < 1.0:
        sorted_logp, idx = logp.sort(dim=-1, descending=True)
        probs = sorted_logp.exp()
        # drop a token when the mass BEFORE it already reached top_p (so the token
        # that crosses the threshold is kept, and the top-1 token is always kept)
        drop = (probs.cumsum(dim=-1) - probs) >= top_p
        sorted_logp = sorted_logp.masked_fill(drop, NEG)
        logp = torch.full_like(logp, NEG).scatter(-1, idx, sorted_logp)
    u = torch.rand_like(logp).clamp_(1e-10, 1.0 - 1e-7)
    return logp - torch.log(-torch.log(u))


class SamplingLM(BatchScorerInterface):
    """Wraps the real LM scorer; same interface, sampled scores."""

    def __init__(self, inner, top_p: float, temp: float):
        self.inner = inner
        self.top_p = top_p
        self.temp = temp

    def init_state(self, x):
        return self.inner.init_state(x)

    def select_state(self, state, i):
        return self.inner.select_state(state, i)

    def score(self, y, state, x):
        raise NotImplementedError("batch_score only (batch beam search)")

    def batch_score(self, ys, states, xs):
        logp, new_states = self.inner.batch_score(ys, states, xs)
        return top_p_gumbel(logp, self.top_p, self.temp), new_states


_Orig = li.GenerateText


class GenerateTextSampling(_Orig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if SAMPLE_MODE != "topp":
            logging.info(f"SAMPLE_MODE='{SAMPLE_MODE}': stock decoding (beam_size={self.beam_search.beam_size})")
            return
        bs = self.beam_search
        assert bs.beam_size == 1, "Gumbel-max sampling needs beam_size: 1 in the decode yaml"
        assert "lm" in bs.full_scorers, list(bs.full_scorers)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        wrapped = SamplingLM(bs.full_scorers["lm"], TOP_P, TEMP)
        bs.full_scorers["lm"] = wrapped
        bs.scorers["lm"] = wrapped
        logging.info(
            f"SAMPLE_MODE=topp: top_p={TOP_P} temperature={TEMP} seed={SEED} "
            f"beam_size=1 class={type(bs).__name__}"
        )


    def __call__(self, text):
        if SAMPLE_MODE == "topp" and PER_UTT_SEED:
            ids = text if isinstance(text, str) else text.tolist()
            s = (SEED * 1_000_003 + zlib.crc32(str(ids).encode())) % (2**31 - 1)
            torch.manual_seed(s)
            torch.cuda.manual_seed_all(s)
        return super().__call__(text)


li.GenerateText = GenerateTextSampling

if __name__ == "__main__":
    li.main()
