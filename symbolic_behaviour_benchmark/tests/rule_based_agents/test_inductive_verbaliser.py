"""
Visual + functional tests for the inductive verbalization mode of
PosdisHypothesisTracker in the categorical stimulus domain.

Runs several warmup games with HypothesisListenerAgent (inductive=True) and
verifies that:
  - Output text contains the expected structural markers.
  - After the first warmup game, the sync-step section appears (value_map populated).
  - Before any feedback, the "No sync step data" fallback is used.
  - No T-1 numeric fallback text appears anywhere.
"""
from __future__ import annotations

import numpy as np
import pytest

from symbolic_behaviour_benchmark.envs import generate_receptive_constructive_test_env
from symbolic_behaviour_benchmark.rule_based_agents.positionally_disentangled_speaker_agent import (
    PositionallyDisentangledSpeakerAgent,
)
from symbolic_behaviour_benchmark.rule_based_agents.verbalize import (
    HypothesisListenerAgent,
    PosdisHypothesisTracker,
)

# ── Config ─────────────────────────────────────────────────────────────────────

_VOCAB   = 6
_SENT    = 3
_LAT     = 3
_COMM    = 1
_DOMAIN  = "categorical"
_SEED    = 0


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ohe(comm_int: np.ndarray, vocab_size: int, max_sent: int) -> np.ndarray:
    ohe = np.zeros((1, max_sent * (vocab_size + 1)), dtype=np.float64)
    for i, tok in enumerate(comm_int[0]):
        ohe[0, i * (vocab_size + 1) + int(tok)] = 1.0
    return ohe


def _no_op() -> dict:
    return {
        "decision": 0,
        "communication_channel": np.zeros((1, _SENT), dtype=np.int64),
    }


def _shape(action: dict) -> dict:
    comm = action.get("communication_channel", np.zeros((1, _SENT), dtype=np.int64))
    if isinstance(comm, np.ndarray) and comm.ndim == 1:
        comm = comm.reshape(1, -1)
    return {**action, "communication_channel": comm}


def _build_env(seed: int = _SEED):
    return generate_receptive_constructive_test_env(
        nbr_latents=_LAT,
        nbr_distractors=0,
        vocab_size=_VOCAB,
        max_sentence_length=_SENT,
        min_nbr_values_per_latent=2,
        max_nbr_values_per_latent=5,
        nbr_communication_rounds=_COMM,
        descriptive=True,
        include_prompts=False,
        seed=seed,
        nbr_object_centric_samples=1,
        provide_listener_feedback=True,
        domain=_DOMAIN,
    )


def _build_agents(inductive: bool = True):
    common = dict(
        action_space=None,
        vocab_size=_VOCAB,
        max_sentence_length=_SENT,
        nbr_communication_rounds=_COMM,
        nbr_latents=_LAT,
    )
    speaker  = PositionallyDisentangledSpeakerAgent(**common)
    listener = HypothesisListenerAgent(**common, slim=False, inductive=inductive)
    return speaker, listener


def _run_one_game(env, obs, infos, speaker, listener, verbalize_fn=None):
    """
    Play one game; return (reward, done, obs, infos, msg_tokens,
    listener_latents, decision, verbalization).

    verbalize_fn: optional callable(msg_tokens, listener_latents, decision) -> str,
        invoked at listener-decision time — BEFORE the feedback round fires.
        This matches real-system timing in game_loop.py's warmup loop.
    """
    n_steps = _COMM + 1 + 1  # speaker + listener + feedback
    final_reward  = 0.0
    done          = False
    msg_tokens    = []
    listener_latents = None
    decision      = 0
    verbalization = ""

    for _ in range(n_steps):
        rnd = infos[0]["round_idx"]

        if rnd == -1:
            if hasattr(listener, "observe_feedback"):
                listener.observe_feedback(infos[1])
            obs, _, done, infos = env.step([_no_op(), _no_op()])
            continue

        is_spk = rnd == 0
        is_lst = rnd == _COMM

        a0 = speaker.next_action(state=obs[0], infos=infos[0]) if is_spk else _no_op()

        li = dict(infos[1])
        if not is_spk:
            li["communication_channel"] = _ohe(obs[1]["communication_channel"], _VOCAB, _SENT)
        a1_raw = listener.next_action(state=obs[1], infos=li)

        if is_lst:
            msg_tokens       = [int(t) for t in np.asarray(obs[1]["communication_channel"]).flatten()]
            listener_latents = infos[1].get("listener_exp_latents")
            listener_exp_text = infos[1].get("listener_exp_text")
            decision         = int(np.asarray(a1_raw["decision"]).flat[0])
            # Verbalize BEFORE the feedback round fires — tracker holds only past-games' data.
            if verbalize_fn is not None:
                verbalization = verbalize_fn(msg_tokens, listener_latents, listener_exp_text, decision)

        a0 = _shape(a0)
        a1 = _shape(a1_raw)
        obs, rewards, done, infos = env.step([a0, a1])
        final_reward = float(rewards[0]) if hasattr(rewards, "__iter__") else float(rewards)

    return final_reward, done, obs, infos, msg_tokens, listener_latents, decision, verbalization


def _run_warmup(n_games: int = 5, seed: int = _SEED, inductive: bool = True):
    """
    Run n_games warmup games and collect (game_idx, verbalization, reward, decision).

    Verbalization is produced at listener-decision time (before the feedback round
    fires for that game), matching game_loop.py's warmup timing.
    """
    env = _build_env(seed=seed)
    speaker, listener = _build_agents(inductive=inductive)

    obs, infos = env.reset()
    speaker.reset()
    listener.reset()
    # Capture tracker AFTER reset: listener.reset() creates a fresh PosdisHypothesisTracker.
    tracker: PosdisHypothesisTracker = listener.tracker
    done = False

    records = []
    for game_idx in range(n_games):
        if done:
            break

        def _verb(msg_tokens, listener_latents, listener_exp_text, decision, _idx=game_idx):
            return tracker.verbalize_pre_result(
                game_idx=_idx,
                msg_tokens=msg_tokens,
                listener_latents=listener_latents,
                decision=decision,
                domain=_DOMAIN,
                o_centric=1,
                inductive=inductive,
                listener_exp_text=listener_exp_text,
            )

        reward, done, obs, infos, msg_tokens, listener_latents, decision, verb = \
            _run_one_game(env, obs, infos, speaker, listener, verbalize_fn=_verb)

        tracker.update_from_reward(msg_tokens, reward)
        records.append((game_idx, verb, reward, decision))

    return records


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_game0_no_sync_data():
    """
    Before any feedback round has fired, the inductive verbalizer should report
    that it has no sync-step data and default to Answer: 0.
    """
    records = _run_warmup(n_games=1)
    _, verb, _, _ = records[0]
    print(f"\n[game 0]\n{verb}\n")
    assert "No sync step data" in verb, f"Expected 'No sync step data' in game-0 output:\n{verb}"
    assert "Answer: 0" in verb


def test_no_t1_fallback_text():
    """
    The inductive mode must never contain T-1 numeric fallback annotations
    ('T-1 fallback') in any game.
    """
    records = _run_warmup(n_games=5)
    for game_idx, verb, _, _ in records:
        assert "T-1 fallback" not in verb, (
            f"Found 'T-1 fallback' in game {game_idx} inductive output:\n{verb}"
        )


def test_sync_step_section_appears_after_first_game():
    """
    From game 1 onward the 'From the last game syncing' header must appear, because
    at least one feedback round has been processed.
    """
    records = _run_warmup(n_games=5)
    for game_idx, verb, _, _ in records[1:]:
        assert "From the last game syncing" in verb, (
            f"Expected 'From the last game syncing' in game {game_idx}:\n{verb}"
        )


def test_inverse_prediction_section_present():
    """
    From game 1 onward the inverse-prediction section ('if the speaker were
    observing a similar stimulus') must appear.
    """
    records = _run_warmup(n_games=5)
    for game_idx, verb, _, _ in records[1:]:
        assert "if the speaker were observing" in verb, (
            f"Expected inverse-prediction section in game {game_idx}:\n{verb}"
        )


def test_answer_consistent_with_decision():
    """
    The 'Answer: N' token in the verbalization must match the decision value
    computed by the agent (for games where the agent has enough data to reason).
    """
    records = _run_warmup(n_games=5)
    for game_idx, verb, _, decision in records[1:]:
        assert f"Answer: {decision}" in verb, (
            f"Game {game_idx}: verbalization answer does not match decision={decision}:\n{verb}"
        )


def test_print_all_games(capsys):
    """
    Non-assertion test: prints the full verbalization for each game so the
    output can be inspected manually with pytest -s.
    """
    records = _run_warmup(n_games=5)
    for game_idx, verb, reward, decision in records:
        result = "CORRECT" if reward > 0 else "INCORRECT"
        print(f"\n{'='*70}")
        print(f"GAME {game_idx}  [{result}]  reward={reward:.1f}  decision={decision}")
        print(f"{'='*70}")
        print(verb)


# ── Standalone runner ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    records = _run_warmup(n_games=5)
    for game_idx, verb, reward, decision in records:
        result = "CORRECT" if reward > 0 else "INCORRECT"
        print(f"\n{'='*70}")
        print(f"GAME {game_idx}  [{result}]  reward={reward:.1f}  decision={decision}")
        print(f"{'='*70}")
        print(verb)
