#!/usr/bin/env python3
"""espnet2.bin.lm_inference with an optional "no repeated n-gram" ban added to
the beam search -- WITHOUT editing the shared espnet2 source.

Why: TTS decoding of a few test sentences "runs away" -- the beam search
locks into a 2-token limit cycle (e.g. "A B A B A B ...") in the generated
speech tokens, never emits EOS, and runs to maxlen. This wrapper bans, at every
step, any token that would complete an n-gram (over the generated speech
tokens, i.e. everything after <generatespeech>) that already occurred earlier
in the same hypothesis. n is read from the NO_REPEAT_NGRAM environment
variable; unset or 0 = disabled = byte-for-byte the stock behaviour.

n=8 is chosen because real speech BPE sequences (all 9957 test utterances, both
tokenizers) contain NO repeated 8-gram, so the ban never fires on anything
resembling real speech, while a 2-token loop violates it ~9 steps in.

How it hooks in: patches espnet2.bin.lm_inference.GenerateText so that, after
the stock constructor builds its (batch) beam search, one extra full scorer is
registered. Then it calls the stock main(), so I/O, sharding (--key_file),
output files and logs are identical to the normal run. lm.sh reaches this via
its ${python} variable (see local/python_norepeat.sh).
"""
import logging
import os
import sys

import torch

import espnet2.bin.lm_inference as li
from espnet2.legacy.nets.scorer_interface import BatchScorerInterface

NO_REPEAT_NGRAM = int(os.environ.get("NO_REPEAT_NGRAM", "0") or 0)
MARKER = "<generatespeech>"
BAN_SCORE = -1.0e4  # effectively -inf, but keeps the sum finite


class NoRepeatNgram(BatchScorerInterface):
    """Stateless scorer: BAN_SCORE for tokens completing an already-seen n-gram."""

    def __init__(self, n: int, marker_id: int, n_vocab: int):
        self.n = n
        self.marker_id = marker_id
        self.n_vocab = n_vocab
        self.n_banned_events = 0  # for the log line at the end

    def _banned(self, seq):
        """seq: python list of token ids of ONE hypothesis (prompt included)."""
        try:
            gen = seq[seq.index(self.marker_id) + 1 :]
        except ValueError:
            return ()
        n = self.n
        if len(gen) < n - 1:
            return ()
        prefix = tuple(gen[len(gen) - (n - 1) :]) if n > 1 else ()
        banned = set()
        # every earlier occurrence of the same (n-1)-gram bans the token that
        # followed it back then
        for i in range(len(gen) - (n - 1)):
            if tuple(gen[i : i + n - 1]) == prefix:
                banned.add(gen[i + n - 1])
        return banned

    def score(self, y, state, x):
        scores = torch.zeros(self.n_vocab, device=y.device, dtype=x.dtype)
        for t in self._banned(y.tolist()):
            scores[t] = BAN_SCORE
        return scores, None

    def batch_score(self, ys, states, xs):
        n_batch = ys.shape[0]
        scores = torch.zeros(n_batch, self.n_vocab, device=ys.device, dtype=xs.dtype)
        rows = ys.tolist()
        for b in range(n_batch):
            banned = self._banned(rows[b])
            if banned:
                self.n_banned_events += 1
                scores[b, list(banned)] = BAN_SCORE
        return scores, None


_Orig = li.GenerateText


class GenerateTextNoRepeat(_Orig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if NO_REPEAT_NGRAM <= 0:
            logging.info("NO_REPEAT_NGRAM not set: stock decoding")
            return
        token_list = self.lm_train_args.token_list
        scorer = NoRepeatNgram(NO_REPEAT_NGRAM, token_list.index(MARKER), len(token_list))
        bs = self.beam_search
        bs.scorers["norepeat"] = scorer
        bs.full_scorers["norepeat"] = scorer
        bs.weights["norepeat"] = 1.0
        self._norepeat_scorer = scorer
        logging.info(
            f"NO_REPEAT_NGRAM={NO_REPEAT_NGRAM}: banning repeated {NO_REPEAT_NGRAM}-grams "
            f"of generated tokens (after {MARKER}); beam search class={type(bs).__name__}"
        )


li.GenerateText = GenerateTextNoRepeat

if __name__ == "__main__":
    li.main()
