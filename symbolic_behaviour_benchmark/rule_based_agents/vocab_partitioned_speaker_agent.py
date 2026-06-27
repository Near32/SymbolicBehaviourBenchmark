"""
Vocab-partitioned positionally-disentangled speaker agent.

Extends PositionallyDisentangledSpeakerAgent with a disjoint token-range
encoding: each latent dimension i gets a reserved block of tokens so that
the token alone (without its message position) identifies both the dimension
and its value.

Encoding:
    token = latent_idx * block_size + latent_value + 1
    block_size = (vocab_size - 1) // nbr_latents

Token 0 is reserved as EoS and is never emitted.

Requirement: vocab_size - 1 >= nbr_latents * max_nbr_values_per_latent.
Raises ValueError at construction if the vocabulary is too small.
"""
from __future__ import annotations

import numpy as np

from .positionally_disentangled_speaker_agent import PositionallyDisentangledSpeakerAgent


class VocabPartitionedSpeakerAgent(PositionallyDisentangledSpeakerAgent):
    """
    PosDis speaker with vocab-partitioned encoding.

    Each latent dimension gets a disjoint token range, making tokens globally
    unambiguous: a listener can recover both dimension and value from the token
    alone without relying on message position.
    """

    def __init__(self, *args, max_nbr_values_per_latent: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self._block_size = (self.vocab_size - 1) // self.nbr_latents
        if self._block_size < max_nbr_values_per_latent:
            needed = self.nbr_latents * max_nbr_values_per_latent + 1
            raise ValueError(
                f"vocab_partition requires block_size ({self._block_size}) >= "
                f"max_nbr_values_per_latent ({max_nbr_values_per_latent}). "
                f"Increase vocab_size to at least {needed} "
                f"(nbr_latents={self.nbr_latents} × max_val={max_nbr_values_per_latent} + 1)."
            )

    def _utter(self, state, infos):
        action_dict = {
            "communication_channel": np.zeros(
                (1, max(self.nbr_latents, self.max_sentence_length))
            ),
            "decision": np.zeros((1, 1)),
        }
        target_stimulus = infos["speaker_exp_latents"][0, 0]
        for sid in range(self.nbr_latents):
            token = sid * self._block_size + int(target_stimulus[sid]) + 1
            assert token < self.vocab_size, (
                f"token {token} >= vocab_size {self.vocab_size} "
                f"(sid={sid}, val={int(target_stimulus[sid])}, block={self._block_size})"
            )
            action_dict["communication_channel"][0, sid] = token
        return action_dict
