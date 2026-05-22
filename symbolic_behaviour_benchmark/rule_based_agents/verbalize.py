"""
Hypothesis-tracking verbalization for the positionally-disentangled listener agent.

Used by the few_shot_discussion_cot prompt strategy to build natural-language
few-shot examples from rule-based warmup games before the tested LLM takes over.
"""
from __future__ import annotations

import numpy as np


class PosdisHypothesisTracker:
    """
    Accumulates symbol->feature/value hypotheses across warmup games and produces
    natural-language verbalizations that show the current hypothesis state before
    each game, then updates after the game result is known.

    Decision structure (descriptive=True, nbr_distractors=0):
      - Listener sees ONE object (its own stimulus) and makes a binary decision:
          0 = same latent class as the speaker's target
          1 = different latent class
      - listener_exp_latents[0] gives that single object's latents:
          shape (1, nbr_latents) for O=1, or (O, nbr_latents) for O>1
            (multiple object-centric views of the same base object).

    Create one instance per episode -- its state resets on instantiation.
    """

    def __init__(self) -> None:
        # (position, token) -> {"seen": int, "confirmed": int}
        self._hyps: dict[tuple[int, int], dict[str, int]] = {}
        # (position, token) -> {value: count}  -- learned from feedback rounds
        self._value_map: dict[tuple[int, int], dict[int, int]] = {}

    def verbalize_pre_result(
        self,
        game_idx: int,
        msg_tokens: list[int],
        listener_latents,           # ndarray or None
        decision: int,              # 0 = same class, 1 = different
        domain: str = "SCS",
        o_centric: int = 1,
    ) -> str:
        """
        Build the verbalization for this game using all prior hypotheses, WITHOUT
        the result line (reward unknown at call time) and WITHOUT updating the tracker.

        Call update_from_reward() after the game to commit the observations.
        """
        active  = [(i, t) for i, t in enumerate(msg_tokens) if t > 0]
        msg_s   = "[" + ", ".join(str(t) for t in msg_tokens) + "]"
        verdict = "same" if decision == 0 else "different"

        prior_str = self._format_prior(game_idx)

        if active:
            infer_parts = []
            for i, t in active:
                best_val = self.best_value(i, t)
                vm = self._value_map.get((i, t), {})
                if vm:
                    n_obs  = sum(vm.values())
                    source = f"[from {n_obs} feedback obs]"
                elif (i, t) in self._hyps:
                    s = self._hyps[(i, t)]
                    source = (
                        "[consistent, confirmed via T-1]"
                        if s["confirmed"] == s["seen"] and s["seen"] > 0
                        else "[consistent, tentative via T-1]"
                    )
                else:
                    source = "[new, T-1 fallback]"
                infer_parts.append(f"symbol {t} (pos {i}) -> value={best_val} {source}")
            infer_str = (
                "Speaker sent " + msg_s + ". Decoded: "
                + "; ".join(infer_parts) + ". "
            )
        else:
            infer_str = f"Speaker sent {msg_s} (all EoS -- no feature decoded). "

        match_str = self._format_match(active, listener_latents, verdict)

        domain_note = ""
        if domain in ("categorical", "pseudoword") and o_centric > 1:
            domain_note = (
                "(Note: the message encodes only core category labels, "
                "not adjective modifiers visible in multi-view descriptions.) "
            )

        return prior_str + infer_str + match_str + domain_note + f"Answer: {decision}"

    def update_from_reward(self, msg_tokens: list[int], reward: float) -> None:
        """Update hypothesis tracker after the game result is known."""
        was_correct = reward > 0
        for i, t in enumerate(msg_tokens):
            if t == 0:
                continue
            key = (i, t)
            if key not in self._hyps:
                self._hyps[key] = {"seen": 0, "confirmed": 0}
            self._hyps[key]["seen"] += 1
            if was_correct:
                self._hyps[key]["confirmed"] += 1

    def verbalize_and_update(
        self,
        game_idx: int,
        msg_tokens: list[int],
        listener_latents,           # ndarray or None
        decision: int,              # 0 = same class, 1 = different
        reward: float,              # > 0 means correct
        domain: str = "SCS",
        o_centric: int = 1,
    ) -> str:
        """
        Verbalize this game with all prior hypotheses, then update the tracker.

        Must be called AFTER running the game so that `reward` is known.
        Returns a string including the result line, intended for the discussion history.
        """
        was_correct = reward > 0
        text = self.verbalize_pre_result(
            game_idx, msg_tokens, listener_latents, decision, domain, o_centric
        )
        result_str = (
            "Result: correct -- inferences above are reinforced. "
            if was_correct else
            "Result: incorrect -- some symbol-to-value associations may be wrong; "
            "treat tentative hypotheses with extra caution. "
        )
        answer_tag = f"Answer: {decision}"
        text = text[: -len(answer_tag)] + result_str + answer_tag
        self.update_from_reward(msg_tokens, reward)
        return text

    def observe_feedback(self, msg_tokens: list[int], feedback_latents) -> None:
        """Update value map from the feedback round (speaker's target latents).

        Called unconditionally: the env sets listener_exp_latents = speaker_exp_latents
        during round_idx == -1, making this always-valid ground truth.
        """
        if feedback_latents is None:
            return
        try:
            obs_obj = np.asarray(feedback_latents)[0][0].flatten()
            for i, t in enumerate(msg_tokens):
                if t > 0 and i < len(obs_obj):
                    actual_value = int(obs_obj[i])
                    vm = self._value_map.setdefault((i, t), {})
                    vm[actual_value] = vm.get(actual_value, 0) + 1
        except Exception:
            pass

    def best_value(self, pos: int, tok: int) -> int:
        """Return the most-observed value for (pos, tok), or tok-1 as fallback."""
        vm = self._value_map.get((pos, tok), {})
        return max(vm, key=vm.get) if vm else tok - 1

    # -- Private helpers --------------------------------------------------------

    def _format_prior(self, game_idx: int) -> str:
        if not self._hyps and not self._value_map:
            return (
                f"Game {game_idx}: No prior games yet -- using T-1 fallback "
                "(symbol T at position i assumed to encode value T-1). "
            )
        parts = []
        all_keys = sorted(set(self._value_map.keys()) | set(self._hyps.keys()))
        for (pos, tok) in all_keys:
            vm  = self._value_map.get((pos, tok), {})
            hyp = self._hyps.get((pos, tok), {"seen": 0, "confirmed": 0})
            if vm:
                best_val = max(vm, key=vm.get)
                n_obs    = sum(vm.values())
                n_best   = vm[best_val]
                conf_str = (
                    f"confirmed from {n_obs} feedback obs"
                    if n_best == n_obs
                    else f"best of {n_best}/{n_obs} feedback obs"
                )
                parts.append(f"symbol {tok} at pos {pos} -> value {best_val} [{conf_str}]")
            else:
                seen, conf = hyp["seen"], hyp["confirmed"]
                status = (
                    "confirmed"
                    if conf == seen and conf > 0
                    else f"tentative -- {conf}/{seen} games correct"
                )
                parts.append(
                    f"symbol {tok} at pos {pos} -> value {tok - 1} [T-1 fallback, {status}]"
                )
        return (
            f"Game {game_idx}: Accumulated hypotheses ({game_idx} prior game(s)) -- "
            + "; ".join(parts) + ". "
        )

    def _format_match(
        self,
        active: list[tuple[int, int]],
        listener_latents,
        verdict: str,
    ) -> str:
        if listener_latents is not None:
            try:
                stimuli = np.asarray(listener_latents)[0]   # (nbr_views, nbr_latents)
                obs_obj = stimuli[0].flatten()               # base latent values
                decoded = {i: self.best_value(i, t) for i, t in active}
                n_match = sum(
                    int(obs_obj[i]) == decoded[i]
                    for i in decoded if i < len(obs_obj)
                )
                return (
                    f"My stimulus matches {n_match}/{len(decoded)} inferred values "
                    f"-> {verdict} latent class. "
                )
            except Exception:
                pass
        return f"Concluded {verdict} latent class. "


class HypothesisListenerAgent:
    """
    Listener that learns token->value mappings from the feedback round
    (round_idx == -1) instead of assuming the fixed T-1 formula.

    Starts identical to PositionallyDisentangledListenerAgent (T-1 fallback)
    and improves within the episode as feedback accumulates.

    Requires the caller to invoke observe_feedback(infos1) during the
    round_idx==-1 step so the tracker can update from the speaker's target.
    """

    def __init__(
        self,
        action_space: object,
        vocab_size: int,
        max_sentence_length: int,
        nbr_communication_rounds: int,
        nbr_latents: int,
    ) -> None:
        self.vocab_size = vocab_size
        self.max_sentence_length = max_sentence_length
        self.nbr_communication_rounds = nbr_communication_rounds
        self.nbr_latents = nbr_latents
        self.reset()

    def reset(self) -> None:
        self.round_idx = 0
        self.per_round_decision: list = []
        self.tracker = PosdisHypothesisTracker()
        self._last_msg_tokens: list[int] | None = None
        self._last_listener_latents = None
        self._last_decision: int | None = None

    def observe_feedback(self, infos1: dict) -> None:
        """Call during round_idx==-1: infos1['listener_exp_latents'] = speaker's target."""
        feedback_latents = infos1.get("listener_exp_latents")
        if self._last_msg_tokens is not None:
            self.tracker.observe_feedback(self._last_msg_tokens, feedback_latents)
        self._last_msg_tokens = None
        self._last_listener_latents = None
        self._last_decision = None

    def next_action(self, state: np.ndarray, infos: dict) -> dict:
        self.round_idx = infos["round_idx"]
        action_dict = {
            "communication_channel": np.zeros((1, self.max_sentence_length)),
            "decision": np.zeros((1, 1)),
        }

        if self.round_idx != 0:
            action_dict = self._reason(state=state, infos=infos, action_dict=action_dict)
            self.per_round_decision.append(action_dict["decision"])
        else:
            self.per_round_decision = []

        if self.round_idx == self.nbr_communication_rounds:
            import random
            final_decision = random.choice(self.per_round_decision)
            action_dict["decision"] = final_decision

        return action_dict

    def _reason(self, state: np.ndarray, infos: dict, action_dict: dict) -> dict:
        target_utterance_ohe = infos["communication_channel"]
        target_utterance_widx = np.reshape(
            target_utterance_ohe, (self.max_sentence_length, -1)
        )
        target_utterance_widx = (
            np.arange(self.vocab_size + 1) * target_utterance_widx
        ).max(axis=-1)

        pos_start = self.round_idx - 1
        pos_end = self.round_idx
        if self.round_idx == 1 and self.nbr_communication_rounds == 1:
            pos_start = 0
            pos_end = self.max_sentence_length

        round_utterance = target_utterance_widx[pos_start:pos_end].astype(int)

        listener_latents = infos.get("listener_exp_latents")
        try:
            obs_obj = np.asarray(listener_latents)[0][0].flatten()
        except Exception:
            obs_obj = np.array([])

        n_active = int((round_utterance > 0).sum())
        n_match = 0
        for rel_i, t in enumerate(round_utterance):
            if t == 0:
                continue
            abs_i = pos_start + rel_i
            if abs_i < len(obs_obj) and int(obs_obj[abs_i]) == self.tracker.best_value(abs_i, int(t)):
                n_match += 1

        decision = 0.0 if n_match >= max(n_active, 1) else 1.0
        action_dict["decision"][0, 0] = decision

        self._last_msg_tokens = list(target_utterance_widx.astype(int))
        self._last_listener_latents = listener_latents
        self._last_decision = int(decision)

        return action_dict
