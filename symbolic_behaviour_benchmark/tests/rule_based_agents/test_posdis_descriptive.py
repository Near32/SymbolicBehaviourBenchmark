"""
Tests for PositionallyDisentangledListenerAgent + SpeakerAgent in the
descriptive=True, nbr_distractors=0 configuration used by meta-rg-s2b.

Covers:
  - Listener produces both decision-0 (same class) and decision-1 (different)
    across an episode — guards against the argmax-of-1 bug.
  - Overall accuracy is high without vocabulary permutation (T-1 decode is exact).
  - Accuracy degrades under vocabulary permutation (known limitation; documents
    expected behaviour so a future fix can be verified against this baseline).
"""
from __future__ import annotations

import numpy as np
import pytest

from symbolic_behaviour_benchmark.envs import generate_receptive_constructive_test_env
from symbolic_behaviour_benchmark.rule_based_agents.positionally_disentangled_speaker_agent import (
    PositionallyDisentangledSpeakerAgent,
)
from symbolic_behaviour_benchmark.rule_based_agents.positionally_disentangled_listener_agent import (
    PositionallyDisentangledListenerAgent,
)
from symbolic_behaviour_benchmark.rule_based_agents.verbalize import HypothesisListenerAgent


# ── Helpers ────────────────────────────────────────────────────────────────────

_VOCAB_SIZE = 6
_MAX_SENT   = 3
_NBR_LAT    = 3
_NBR_COMM   = 1


def _int_comm_to_ohe(comm_int: np.ndarray, vocab_size: int, max_sentence_length: int) -> np.ndarray:
    """Convert integer tokens (1, L) → one-hot (1, L*(vocab_size+1))."""
    ohe = np.zeros((1, max_sentence_length * (vocab_size + 1)), dtype=np.float64)
    for i, tok in enumerate(comm_int[0]):
        ohe[0, i * (vocab_size + 1) + int(tok)] = 1.0
    return ohe


def _no_op(max_sentence_length: int) -> dict:
    return {
        "decision": 0,
        "communication_channel": np.zeros((1, max_sentence_length), dtype=np.int64),
    }


def _ensure_shape(action: dict, max_sentence_length: int) -> dict:
    comm = action.get("communication_channel", np.zeros((1, max_sentence_length), dtype=np.int64))
    if isinstance(comm, np.ndarray) and comm.ndim == 1:
        comm = comm.reshape(1, -1)
    return {**action, "communication_channel": comm}


def _build_env(seed: int = 0, use_permutations: bool = False) -> object:
    return generate_receptive_constructive_test_env(
        nbr_latents=_NBR_LAT,
        nbr_distractors=0,
        vocab_size=_VOCAB_SIZE,
        max_sentence_length=_MAX_SENT,
        min_nbr_values_per_latent=2,
        max_nbr_values_per_latent=5,
        nbr_communication_rounds=_NBR_COMM,
        descriptive=True,
        include_prompts=False,
        seed=seed,
        nbr_object_centric_samples=1,
        provide_listener_feedback=True,
        use_communication_channel_permutations=use_permutations,
    )


def _build_agents() -> tuple:
    common = dict(
        action_space=None,
        vocab_size=_VOCAB_SIZE,
        max_sentence_length=_MAX_SENT,
        nbr_communication_rounds=_NBR_COMM,
        nbr_latents=_NBR_LAT,
    )
    return PositionallyDisentangledSpeakerAgent(**common), PositionallyDisentangledListenerAgent(**common)


def _run_episode(env, speaker, listener) -> list[tuple[float, int]]:
    """
    Play one full episode with rb speaker and listener.
    Returns list of (reward, listener_decision) for every decision step.
    The listener is called on every round (including round_idx=0) so that its
    per_round_decision list is initialised correctly.
    """
    obs, infos = env.reset()
    speaker.reset()
    listener.reset()

    # Each entry: (reward, listener_decision, is_test)
    results: list[tuple[float, int, bool]] = []
    done = False
    last_decision: int | None = None
    last_is_test: bool = False

    while not done:
        round_idx = infos[0]["round_idx"]

        if round_idx == -1:
            if hasattr(listener, "observe_feedback"):
                listener.observe_feedback(infos[1])
            obs, rewards, done, infos = env.step([_no_op(_MAX_SENT)] * 2)
            continue

        is_speaker_round  = round_idx == 0
        is_listener_round = round_idx == _NBR_COMM

        if is_speaker_round:
            last_is_test = "test" in str(infos[0].get("mode", ""))

        # Speaker
        a0 = speaker.next_action(state=obs[0], infos=infos[0]) if is_speaker_round else _no_op(_MAX_SENT)
        a0 = _ensure_shape(a0, _MAX_SENT)

        # Listener — called every round for round-tracking; OHE only on non-speaker rounds
        listener_infos = dict(infos[1])
        if not is_speaker_round:
            listener_infos["communication_channel"] = _int_comm_to_ohe(
                obs[1]["communication_channel"], _VOCAB_SIZE, _MAX_SENT
            )
        a1_raw = listener.next_action(state=obs[1], infos=listener_infos)
        if is_listener_round:
            last_decision = int(np.asarray(a1_raw["decision"]).flat[0])
        a1 = _ensure_shape(a1_raw, _MAX_SENT)

        obs, rewards, done, infos = env.step([a0, a1])
        reward = float(rewards[0]) if hasattr(rewards, "__iter__") else float(rewards)

        if is_listener_round and last_decision is not None:
            results.append((reward, last_decision, last_is_test))
            if hasattr(listener, "update_from_reward"):
                listener.update_from_reward(reward)

    return results


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_listener_produces_both_decisions():
    """
    Without permutation the listener must output decision-1 on some games.
    If decision is always 0, the argmax-of-1 bug is still present.
    """
    env = _build_env(seed=0, use_permutations=False)
    speaker, listener = _build_agents()
    results = _run_episode(env, speaker, listener)
    decisions = [d for _, d, _ in results]
    assert len(decisions) > 0, "Episode produced no decision steps"
    assert 0 in decisions, "Listener never decided same-class (decision=0)"
    assert 1 in decisions, (
        "Listener always decided same-class (decision=0) over "
        f"{len(decisions)} games — argmax-of-1 bug may still be present"
    )


def test_accuracy_no_permutation():
    """
    Without vocabulary permutation, T-1 decoding is exact and the listener
    should achieve near-perfect accuracy.
    """
    env = _build_env(seed=42, use_permutations=False)
    speaker, listener = _build_agents()
    results = _run_episode(env, speaker, listener)
    n = len(results)
    correct = sum(1 for r, _, _ in results if r > 0)
    acc = correct / max(n, 1)
    assert acc >= 0.9, (
        f"Expected accuracy >= 0.9 without permutation, got {acc:.2f} over {n} games"
    )


def test_accuracy_degrades_with_permutation():
    """
    Vocabulary permutation scrambles token indices, breaking T-1 decoding.
    Accuracy with permutation must be strictly lower than without.
    This test documents a known limitation; it will pass once/if a
    permutation-aware listener is implemented.
    """
    env_no = _build_env(seed=7, use_permutations=False)
    env_p  = _build_env(seed=7, use_permutations=True)
    spk_no, lst_no = _build_agents()
    spk_p,  lst_p  = _build_agents()

    res_no = _run_episode(env_no, spk_no, lst_no)
    res_p  = _run_episode(env_p,  spk_p,  lst_p)

    acc_no = sum(r > 0 for r, _, _ in res_no) / max(len(res_no), 1)
    acc_p  = sum(r > 0 for r, _, _ in res_p)  / max(len(res_p),  1)
    assert acc_no > acc_p, (
        f"Expected accuracy to drop under permutation "
        f"(no_perm={acc_no:.2f}, perm={acc_p:.2f})"
    )


def test_zsct_accuracy_no_permutation():
    """
    Without vocabulary permutation, rb speaker + listener should reach 100% ZSCT.
    Runs across multiple seeds to confirm robustness.
    """
    zsct_correct = zsct_total = 0
    support_correct = support_total = 0
    for seed in range(5):
        env = _build_env(seed=seed, use_permutations=False)
        speaker, listener = _build_agents()
        results = _run_episode(env, speaker, listener)
        for reward, _, is_test in results:
            if is_test:
                zsct_total += 1
                zsct_correct += int(reward > 0)
            else:
                support_total += 1
                support_correct += int(reward > 0)

    support_acc = support_correct / max(support_total, 1)
    zsct_acc    = zsct_correct    / max(zsct_total, 1)
    print(f"\n  support: {support_correct}/{support_total} = {support_acc:.1%}")
    print(f"  ZSCT:    {zsct_correct}/{zsct_total} = {zsct_acc:.1%}")
    assert zsct_acc == 1.0, (
        f"Expected 100% ZSCT without permutation, got {zsct_acc:.1%} "
        f"({zsct_correct}/{zsct_total})"
    )


def _build_hypothesis_listener() -> HypothesisListenerAgent:
    return HypothesisListenerAgent(
        action_space=None,
        vocab_size=_VOCAB_SIZE,
        max_sentence_length=_MAX_SENT,
        nbr_communication_rounds=_NBR_COMM,
        nbr_latents=_NBR_LAT,
    )


def test_hypothesis_listener_no_permutation():
    """
    HypothesisListenerAgent with feedback must reach >=90% accuracy without
    permutation (same baseline as the fixed posdis listener).
    """
    env = _build_env(seed=42, use_permutations=False)
    speaker, _ = _build_agents()
    listener = _build_hypothesis_listener()
    results = _run_episode(env, speaker, listener)
    n = len(results)
    acc = sum(r > 0 for r, _, _ in results) / max(n, 1)
    assert acc >= 0.9, (
        f"Expected HypothesisListenerAgent accuracy >= 0.9 without permutation, "
        f"got {acc:.2f} over {n} games"
    )


def test_hypothesis_listener_improves_under_permutation():
    """
    With vocab permutation the hypothesis listener must exceed the fixed T-1
    baseline (~52%). The feedback round teaches correct token->value mappings
    across games within the episode.
    """
    zsct_correct = zsct_total = 0
    support_correct = support_total = 0
    for seed in range(5):
        env = _build_env(seed=seed, use_permutations=True)
        speaker, _ = _build_agents()
        listener = _build_hypothesis_listener()
        results = _run_episode(env, speaker, listener)
        for reward, _, is_test in results:
            if is_test:
                zsct_total += 1
                zsct_correct += int(reward > 0)
            else:
                support_total += 1
                support_correct += int(reward > 0)

    support_acc = support_correct / max(support_total, 1)
    zsct_acc    = zsct_correct    / max(zsct_total, 1)
    print(f"\n  support: {support_correct}/{support_total} = {support_acc:.1%}")
    print(f"  ZSCT:    {zsct_correct}/{zsct_total} = {zsct_acc:.1%}")
    assert support_acc > 0.55, (
        f"HypothesisListenerAgent support accuracy under permutation should exceed "
        f"~52% T-1 baseline, got {support_acc:.2f}"
    )


# ── Standalone runner ──────────────────────────────────────────────────────────

def test_zsct_accuracy_with_permutation():
    """
    With vocabulary permutation, rb speaker + listener ZSCT accuracy.
    Reports results; no hard assertion on value (documents current behaviour).
    """
    zsct_correct = zsct_total = 0
    support_correct = support_total = 0
    for seed in range(5):
        env = _build_env(seed=seed, use_permutations=True)
        speaker, listener = _build_agents()
        results = _run_episode(env, speaker, listener)
        for reward, _, is_test in results:
            if is_test:
                zsct_total += 1
                zsct_correct += int(reward > 0)
            else:
                support_total += 1
                support_correct += int(reward > 0)

    support_acc = support_correct / max(support_total, 1)
    zsct_acc    = zsct_correct    / max(zsct_total, 1)
    print(f"\n  support: {support_correct}/{support_total} = {support_acc:.1%}")
    print(f"  ZSCT:    {zsct_correct}/{zsct_total} = {zsct_acc:.1%}")


if __name__ == "__main__":
    tests = [
        test_listener_produces_both_decisions,
        test_accuracy_no_permutation,
        test_accuracy_degrades_with_permutation,
        test_zsct_accuracy_no_permutation,
        test_zsct_accuracy_with_permutation,
        test_hypothesis_listener_no_permutation,
        test_hypothesis_listener_improves_under_permutation,
    ]
    for fn in tests:
        print(f"Running {fn.__name__} ...", end=" ", flush=True)
        fn()
        print("PASS")
