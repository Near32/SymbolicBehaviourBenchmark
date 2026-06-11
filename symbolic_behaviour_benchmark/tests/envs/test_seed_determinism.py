"""Acceptance test: an S2B episode's structure is a pure function of the seed.

Uses ONLY the S2B public API (the env factory + env.step). After the RNG
fixes, an episode's per-step structure — (mode, listener stimulus, observed
message) — must be:

  1. reproducible:        same seed + same actions      -> identical, and
  2. action-independent:  same seed + different actions  -> identical.

Only surface feedback text that *reports* an action may differ; the underlying
stimuli/messages/modes may not. This is the prerequisite for the two-env
(correct-world vs corrupted-world) matched-pair steering experiment.

This mirrors meta-rg-s2b/tests/test_s2b_determinism.py but without any meta_rg
dependency, so it lives with the env it guards.
"""
import random

import numpy as np
import pytest

from symbolic_behaviour_benchmark.envs import generate_receptive_constructive_test_env

SEED = 0
VOCAB_SIZE = 16
MAX_SENTENCE_LENGTH = 3


def _make_env():
    return generate_receptive_constructive_test_env(
        nbr_latents=3,
        nbr_distractors=0,
        vocab_size=VOCAB_SIZE,
        max_sentence_length=MAX_SENTENCE_LENGTH,
        min_nbr_values_per_latent=2,
        max_nbr_values_per_latent=5,
        nbr_communication_rounds=1,
        descriptive=True,
        include_prompts=True,
        seed=SEED,
        nbr_object_centric_samples=1,
        provide_listener_feedback=True,
        sampling_strategy="component-focused-1shot",
        discussion_mode=True,
        domain="categorical",
        verbose_prompts=False,
        allow_cot_response=True,
        elicitation_strategies=[],
    )


def _capture(obs, infos):
    mode = str(infos[0].get("mode"))
    stim = tuple(np.asarray(infos[1].get("listener_exp_latents")).flatten().tolist())
    msg = tuple(np.asarray(obs[1].get("communication_channel")).flatten().tolist())
    return (mode, msg, stim)


def _run_world(listener_decision):
    """Step a full episode feeding a fixed (deterministic) speaker action and the
    given listener decision; return the per-step structure list."""
    random.seed(SEED)
    np.random.seed(SEED)
    env = _make_env()
    env.seed(SEED)
    obs, infos = env.reset()
    rows = []
    done = False
    steps = 0
    while not done and steps < 1000:
        rows.append(_capture(obs, infos))
        sidx = int(infos[0].get("stimulus_idx", 0))
        speaker_action = {
            "decision": 0,
            "communication_channel": np.ones((1, MAX_SENTENCE_LENGTH)) * (sidx % VOCAB_SIZE),
        }
        listener_action = {
            "decision": listener_decision,
            "communication_channel": np.zeros((1, MAX_SENTENCE_LENGTH)),
        }
        obs, _reward, done, infos = env.step([speaker_action, listener_action])
        steps += 1
    return rows


def test_reproducible_same_seed_same_actions():
    assert _run_world(0) == _run_world(0)


def test_action_independent_structure():
    a = _run_world(0)
    b = _run_world(1)  # different listener decisions throughout
    assert len(a) == len(b) and len(a) > 0
    mismatches = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    assert not mismatches, (
        f"{len(mismatches)}/{len(a)} steps differ in structure under different "
        f"actions; first: A={a[mismatches[0]]} B={b[mismatches[0]]}"
    )
