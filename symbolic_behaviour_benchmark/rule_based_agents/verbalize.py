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
        # (position, token) -> last game index at which hypothesis was confirmed
        self._last_game: dict[tuple[int, int], int] = {}
        # set by verbalize_pre_result; read by update_from_reward / observe_feedback
        self._current_game_idx: int = 0
        # (position, token) -> human-readable label (categorical/pseudoword domains only)
        self._value_label: dict[tuple[int, int], str] = {}
        # (pos, tok) keys observed in the most recent observe_feedback call
        self._last_sync_keys: set[tuple[int, int]] = set()

    def verbalize_pre_result(
        self,
        game_idx: int,
        msg_tokens: list[int],
        listener_latents,           # ndarray or None
        decision: int,              # 0 = same class, 1 = different
        domain: str = "SCS",
        o_centric: int = 1,
        slim: bool = False,
        inductive: bool = False,
        listener_exp_text=None,     # list-of-lists of label strings (categorical/pseudoword)
        decoding: bool = False,
    ) -> str:
        """
        Build the verbalization for this game using all prior hypotheses, WITHOUT
        the result line (reward unknown at call time) and WITHOUT updating the tracker.

        Call update_from_reward() after the game to commit the observations.

        decoding=True: step-by-step DECODE-direction template (symbol->value), comparing the
            decoded message to the listener's own stimulus. Takes precedence over inductive/slim.
        inductive=True: step-by-step inverse-reasoning template (value->symbol prediction).
        slim=True: compact one-liner (ignored when inductive/decoding=True).
        listener_exp_text: when provided, used to display labelled stimulus values
            instead of raw integers in the inductive/decoding verbalization.
        """
        self._current_game_idx = game_idx

        if decoding:
            return self._verbalize_decoding(
                game_idx, msg_tokens, listener_latents, decision,
                domain, o_centric, listener_exp_text,
            )

        if inductive:
            return self._verbalize_inductive(
                game_idx, msg_tokens, listener_latents, decision,
                domain, o_centric, listener_exp_text,
            )

        if slim:
            return self._verbalize_slim(game_idx, msg_tokens, listener_latents, decision, domain, o_centric)

        active  = [(i, t) for i, t in enumerate(msg_tokens) if t > 0]
        msg_s   = "[" + ", ".join(str(t) for t in msg_tokens) + "]"
        verdict = "same" if decision == 0 else "different"

        prior_str = self._format_prior(game_idx)

        if active:
            infer_parts = []
            for i, t in active:
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
                infer_parts.append(f"symbol {t} (pos {i}) -> value={self.display_value(i, t)} {source}")
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
                self._last_game[key] = self._current_game_idx

    def verbalize_and_update(
        self,
        game_idx: int,
        msg_tokens: list[int],
        listener_latents,           # ndarray or None
        decision: int,              # 0 = same class, 1 = different
        reward: float,              # > 0 means correct
        domain: str = "SCS",
        o_centric: int = 1,
        slim: bool = False,
        inductive: bool = False,
        listener_exp_text=None,
        decoding: bool = False,
    ) -> str:
        """
        Verbalize this game with all prior hypotheses, then update the tracker.

        Must be called AFTER running the game so that `reward` is known.
        Returns a string including the result line, intended for the discussion history.

        decoding=True: step-by-step decode-direction template (takes precedence).
        inductive=True: step-by-step inverse-reasoning template.
        slim=True: compact one-liner (ignored when inductive/decoding=True).
        """
        was_correct = reward > 0
        text = self.verbalize_pre_result(
            game_idx, msg_tokens, listener_latents, decision,
            domain, o_centric, slim, inductive, listener_exp_text,
            decoding=decoding,
        )
        if slim:
            result_str = "Correct. " if was_correct else "Incorrect. "
        else:
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

    def observe_feedback(self, msg_tokens: list[int], feedback_latents, feedback_text=None) -> None:
        """Update value map from the feedback round (speaker's target latents).

        Called unconditionally: the env sets listener_exp_latents = speaker_exp_latents
        during round_idx == -1, making this always-valid ground truth.

        feedback_text: optional output of latent_class_to_text (list-of-lists of strings),
            used to store human-readable labels for categorical/pseudoword domains.
        """
        if feedback_latents is None:
            return
        self._last_sync_keys = set()
        try:
            obs_obj = np.asarray(feedback_latents)[0][0].flatten()
            # text_labels[0] is a flat list of label strings, one per latent position
            text_labels = feedback_text[0] if feedback_text is not None else None
            for i, t in enumerate(msg_tokens):
                if t > 0 and i < len(obs_obj):
                    actual_value = int(obs_obj[i])
                    vm = self._value_map.setdefault((i, t), {})
                    vm[actual_value] = vm.get(actual_value, 0) + 1
                    self._last_game[(i, t)] = self._current_game_idx
                    self._last_sync_keys.add((i, t))
                    if text_labels is not None and i < len(text_labels):
                        self._value_label[(i, t)] = str(text_labels[i])
        except Exception:
            pass

    def best_value(self, pos: int, tok: int) -> int:
        """Return the most-observed value for (pos, tok), or tok-1 as fallback."""
        vm = self._value_map.get((pos, tok), {})
        return max(vm, key=vm.get) if vm else tok - 1

    def display_value(self, pos: int, tok: int) -> str:
        """Return the human-readable label for (pos, tok) if known, else the integer value."""
        label = self._value_label.get((pos, tok))
        return label if label is not None else str(self.best_value(pos, tok))

    # -- Decision helpers (diagnostics for predict-vs-decode agreement) ----------

    def decode_decision(self, msg_tokens: list[int], listener_latents) -> int:
        """Decode-direction decision: decode each sent token to its learned value and
        match against the listener's observed stimulus. Mirrors HypothesisListenerAgent
        `_reason`'s value-map branch (must stay in sync). Returns 0 (same) / 1 (different).
        """
        active   = [(i, t) for i, t in enumerate(msg_tokens) if t > 0]
        n_active = len(active)
        if not self._value_map:
            return 0
        try:
            obs_obj = np.asarray(listener_latents)[0][0].flatten()
        except Exception:
            return 0
        n_match = 0
        for i, t in active:
            if i >= len(obs_obj):
                continue
            vm = self._value_map.get((i, t), {})
            if vm and int(obs_obj[i]) == max(vm, key=vm.get):
                n_match += 1
        return 0 if n_match >= max(n_active, 1) else 1

    def predict_decision(self, msg_tokens: list[int], listener_latents) -> int:
        """Predict-direction decision: from the inverse map (value->symbol), predict the
        message the speaker would send for the listener's stimulus and match against the
        actual message. Mirrors `_verbalize_inductive`'s decision (must stay in sync).
        Returns 0 (same) / 1 (different).
        """
        active   = [(i, t) for i, t in enumerate(msg_tokens) if t > 0]
        n_active = len(active)
        if not self._value_map:
            return 0
        try:
            obs_obj = np.asarray(listener_latents)[0][0].flatten()
        except Exception:
            return 0
        inv: dict[tuple[int, int], dict[int, int]] = {}
        for (pos, tok), val_counts in self._value_map.items():
            for val, cnt in val_counts.items():
                inv.setdefault((pos, val), {})[tok] = inv.get((pos, val), {}).get(tok, 0) + cnt
        predicted_msg: dict[int, int | None] = {}
        for i in range(len(msg_tokens)):
            if i >= len(obs_obj):
                continue
            v = int(obs_obj[i])
            predicted_msg[i] = (
                max(inv[(i, v)], key=inv[(i, v)].get) if (i, v) in inv else None
            )
        n_match = sum(
            1 for i, t in active
            if predicted_msg.get(i) is not None and predicted_msg[i] == t
        )
        return 0 if n_match >= max(n_active, 1) else 1

    # -- Private helpers --------------------------------------------------------

    def _verbalize_inductive(
        self,
        game_idx: int,
        msg_tokens: list[int],
        listener_latents,
        decision: int,
        domain: str = "SCS",
        o_centric: int = 1,
        listener_exp_text=None,
    ) -> str:
        """
        Inductive verbalization: reason from the inverse map (value->symbol) to
        predict what the speaker would send for our stimulus, then compare with
        the actual message.  No T-1 fallback: if data is absent the text says so
        and the answer defaults to 0.

        Template:
          Let's think step by step and leverage past games.
          From the last game syncing, we can learn that: symbol X at pos i -> value Y ; ...
          In the current game, if the speaker were observing a similar stimulus as
          ours, [A,B,C], then: at pos i, A -> symbol Z (from game #g) ; ...
          Since the speaker's message is [...], yield N/M matches, they are likely
          observing a [similar/different] stimulus. Answer: D
        """
        msg_s    = "[" + ", ".join(str(t) for t in msg_tokens) + "]"
        active   = [(i, t) for i, t in enumerate(msg_tokens) if t > 0]
        n_active = len(active)

        # 1. Preamble
        preamble = "Let's think step by step and leverage past games. "

        # 2. No sync-step data at all — bail out early
        if not self._value_map:
            return (
                preamble
                + "No sync step data yet -- cannot predict expected symbols. "
                + "Answer: 0"
            )

        # 3. Sync-step summary: only the associations from the most recent feedback round
        last_keys = sorted(self._last_sync_keys) if self._last_sync_keys else []
        if last_keys:
            sync_parts = [
                f"symbol {tok} at pos {pos} -> value {self.display_value(pos, tok)}"
                for (pos, tok) in last_keys
            ]
            sync_str = (
                "From the last game syncing, we can learn that: "
                + " ; ".join(sync_parts) + ". "
            )
        else:
            sync_str = "From the last game syncing, no new associations were recorded. "

        # 4. Build inverse map: (pos, value) -> {token: total_count}
        inv: dict[tuple[int, int], dict[int, int]] = {}
        for (pos, tok), val_counts in self._value_map.items():
            for val, cnt in val_counts.items():
                key = (pos, val)
                inv.setdefault(key, {})[tok] = inv.get(key, {}).get(tok, 0) + cnt

        # 5. Predict expected speaker message for listener's stimulus
        if listener_latents is not None:
            try:
                obs_obj      = np.asarray(listener_latents)[0][0].flatten()
                n_pos        = len(msg_tokens)
                listener_vals = [
                    int(obs_obj[i]) if i < len(obs_obj) else None
                    for i in range(n_pos)
                ]

                # Resolve human-readable labels for the listener's stimulus values.
                # Priority: listener_exp_text (direct from env) > _value_label (from feedback)
                # > integer fallback.
                try:
                    raw_labels = listener_exp_text[0] if listener_exp_text is not None else None
                except (TypeError, IndexError):
                    raw_labels = None

                def _stim_label(i: int, v: int | None) -> str:
                    if v is None:
                        return "?"
                    if raw_labels is not None and i < len(raw_labels):
                        return str(raw_labels[i])
                    return str(v)

                stim_s = "[" + ", ".join(
                    _stim_label(i, v) for i, v in enumerate(listener_vals)
                ) + "]"

                pred_parts: list[str] = []
                predicted_msg: dict[int, int | None] = {}
                for i, val in enumerate(listener_vals):
                    if val is None:
                        continue
                    label   = _stim_label(i, val)
                    inv_key = (i, val)
                    if inv_key in inv:
                        best_tok = max(inv[inv_key], key=inv[inv_key].get)
                        last_g   = self._last_game.get((i, best_tok), "?")
                        pred_parts.append(
                            f"at pos {i}, {label} -> symbol {best_tok} (from game #{last_g})"
                        )
                        predicted_msg[i] = best_tok
                    else:
                        pred_parts.append(
                            f"at pos {i}, {label} has not been observed yet"
                        )
                        predicted_msg[i] = None

                inverse_str = (
                    f"In the current game, if the speaker were observing a similar "
                    f"stimulus as ours, {stim_s}, then: "
                    + " ; ".join(pred_parts) + ". "
                )

                # 6. Compare actual message to predicted message
                n_match = sum(
                    1 for i, t in active
                    if predicted_msg.get(i) is not None and predicted_msg[i] == t
                )
                match_s = f"yield {n_match}/{n_active} matches"
                if n_match >= max(n_active, 1):
                    concl    = "they are likely observing a similar stimulus"
                    decision = 0
                else:
                    concl    = "they are likely observing a different stimulus"
                    decision = 1
                compare_str = (
                    f"Since the speaker's message is {msg_s}, "
                    f"{match_s}, {concl}. "
                )
            except Exception:
                inverse_str = "Could not build inverse prediction (latents unavailable). "
                compare_str = ""
                decision    = 0
        else:
            inverse_str = "Listener's stimulus is unknown -- cannot predict expected symbols. "
            compare_str = ""
            decision    = 0

        return preamble + sync_str + inverse_str + compare_str + f"Answer: {decision}"

    def _verbalize_decoding(
        self,
        game_idx: int,
        msg_tokens: list[int],
        listener_latents,
        decision: int,
        domain: str = "SCS",
        o_centric: int = 1,
        listener_exp_text=None,
    ) -> str:
        """
        Decode verbalization: reason in the FORWARD direction (symbol->value). Decode the
        speaker's actual message into values using what we synced, then check whether the
        decoded message matches the listener's own stimulus.  No T-1 fallback: tokens absent
        from the value map are "not observed yet" and count as non-matching.

        This is the decode-direction analog of `_verbalize_inductive` (which instead predicts
        what the speaker WOULD send for our stimulus).  The emitted Answer is the decode
        decision, consistent with HypothesisListenerAgent._reason.

        Template:
          Let's think step by step and leverage past games.
          From the last game syncing, we can learn that: symbol X at pos i -> value Y ; ...
          The speaker's message is [...]. Decoding each symbol from what we synced:
          at pos i, symbol T -> value V (from game #g) ; ...
          Comparing the decoded message to our own stimulus, [A,B,C], yield N/M matches;
          the message [matches/does not match] our own stimulus, so we are likely observing
          a [similar/different] stimulus. Answer: D
        """
        msg_s    = "[" + ", ".join(str(t) for t in msg_tokens) + "]"
        active   = [(i, t) for i, t in enumerate(msg_tokens) if t > 0]
        n_active = len(active)

        # 1. Preamble
        preamble = "Let's think step by step and leverage past games. "

        # 2. No sync-step data at all — bail out early
        if not self._value_map:
            return (
                preamble
                + "No sync step data yet -- cannot decode the message. "
                + "Answer: 0"
            )

        # 3. Sync-step summary: only the associations from the most recent feedback round
        last_keys = sorted(self._last_sync_keys) if self._last_sync_keys else []
        if last_keys:
            sync_parts = [
                f"symbol {tok} at pos {pos} -> value {self.display_value(pos, tok)}"
                for (pos, tok) in last_keys
            ]
            sync_str = (
                "From the last game syncing, we can learn that: "
                + " ; ".join(sync_parts) + ". "
            )
        else:
            sync_str = "From the last game syncing, no new associations were recorded. "

        # 4. Decode each active token of the actual message into a value
        if listener_latents is not None:
            try:
                obs_obj = np.asarray(listener_latents)[0][0].flatten()

                # Resolve human-readable labels for the listener's stimulus values.
                try:
                    raw_labels = listener_exp_text[0] if listener_exp_text is not None else None
                except (TypeError, IndexError):
                    raw_labels = None

                def _stim_label(i: int, v: int | None) -> str:
                    if v is None:
                        return "?"
                    if raw_labels is not None and i < len(raw_labels):
                        return str(raw_labels[i])
                    return str(v)

                decode_parts: list[str] = []
                decoded_vals: dict[int, int | None] = {}
                for i, t in active:
                    vm = self._value_map.get((i, t), {})
                    if vm:
                        best_val = max(vm, key=vm.get)
                        last_g   = self._last_game.get((i, t), "?")
                        decoded_vals[i] = best_val
                        decode_parts.append(
                            f"at pos {i}, symbol {t} -> value {self.display_value(i, t)} "
                            f"(from game #{last_g})"
                        )
                    else:
                        decoded_vals[i] = None
                        decode_parts.append(
                            f"at pos {i}, symbol {t} has not been observed yet"
                        )

                decode_str = (
                    f"The speaker's message is {msg_s}. "
                    f"Decoding each symbol from what we synced: "
                    + " ; ".join(decode_parts) + ". "
                )

                # Listener's own stimulus, for the comparison line
                n_pos       = len(msg_tokens)
                obs_vals    = [
                    int(obs_obj[i]) if i < len(obs_obj) else None for i in range(n_pos)
                ]
                stim_s = "[" + ", ".join(
                    _stim_label(i, v) for i, v in enumerate(obs_vals)
                ) + "]"

                # 5. Compare decoded message to the listener's own stimulus
                n_match = sum(
                    1 for i, _t in active
                    if decoded_vals.get(i) is not None
                    and i < len(obs_obj)
                    and decoded_vals[i] == int(obs_obj[i])
                )
                match_s = f"yield {n_match}/{n_active} matches"
                if n_match >= max(n_active, 1):
                    concl    = ("the message matches our own stimulus, so we are likely "
                                "observing a similar stimulus")
                    decision = 0
                else:
                    concl    = ("the message does not match our own stimulus, so we are "
                                "likely observing a different stimulus")
                    decision = 1
                compare_str = (
                    f"Comparing the decoded message to our own stimulus, {stim_s}, "
                    f"{match_s}; {concl}. "
                )
            except Exception:
                decode_str  = "Could not decode the message (latents unavailable). "
                compare_str = ""
                decision    = 0
        else:
            decode_str  = "Listener's stimulus is unknown -- cannot compare the decoded message. "
            compare_str = ""
            decision    = 0

        return preamble + sync_str + decode_str + compare_str + f"Answer: {decision}"

    def _verbalize_slim(
        self,
        game_idx: int,
        msg_tokens: list[int],
        listener_latents,
        decision: int,
        domain: str = "SCS",
        o_centric: int = 1,
    ) -> str:
        """Compact one-liner: confirmed evidence only, last game ref, no T-1 labels."""
        active  = [(i, t) for i, t in enumerate(msg_tokens) if t > 0]
        verdict = "same" if decision == 0 else "different"
        msg_s   = "[" + ", ".join(str(t) for t in msg_tokens) + "]"

        # Prior: only associations newly confirmed in the previous game
        prev = game_idx - 1
        prior_parts = []
        for (pos, tok) in sorted(
            key for key, last in self._last_game.items() if last == prev
        ):
            prior_parts.append(f"symbol {tok} pos {pos}-->{self.display_value(pos, tok)}")
        prior_str = (
            f"Game {game_idx}: from game {prev}: " + "; ".join(prior_parts) + ". "
            if prior_parts
            else f"Game {game_idx}: no new evidence from game {prev}. "
            if game_idx > 0
            else f"Game {game_idx}: no prior evidence. "
        )

        # Inference: only active tokens with confirmed evidence
        infer_parts = []
        for i, t in active:
            hyp  = self._hyps.get((i, t), {"seen": 0, "confirmed": 0})
            last = self._last_game.get((i, t), game_idx)
            if (i, t) in self._value_map:
                infer_parts.append(f"symbol {t} pos {i}-->{self.display_value(i, t)} (game {last})")
            elif hyp["confirmed"] > 0:
                infer_parts.append(f"symbol {t} pos {i}-->{self.display_value(i, t)} (game {last})")
        infer_str = (
            f"Sent {msg_s}: " + "; ".join(infer_parts) + ". "
            if infer_parts
            else f"Sent {msg_s}: no confirmed decoding. "
        )

        # Match
        match_str = self._format_match_slim(active, listener_latents, verdict)

        return prior_str + infer_str + match_str + f"Answer: {decision}"

    def _format_match_slim(
        self,
        active: list[tuple[int, int]],
        listener_latents,
        verdict: str,
    ) -> str:
        if listener_latents is not None:
            try:
                obs_obj = np.asarray(listener_latents)[0][0].flatten()
                decoded = {i: self.best_value(i, t) for i, t in active}
                n_match = sum(
                    int(obs_obj[i]) == decoded[i]
                    for i in decoded if i < len(obs_obj)
                )
                return f"Matched {n_match}/{len(decoded)}-->{verdict} class. "
            except Exception:
                pass
        return f"Concluded {verdict} class. "

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
                parts.append(f"symbol {tok} at pos {pos} -> value {self.display_value(pos, tok)} [{conf_str}]")
            else:
                seen, conf = hyp["seen"], hyp["confirmed"]
                status = (
                    "confirmed"
                    if conf == seen and conf > 0
                    else f"tentative -- {conf}/{seen} games correct"
                )
                parts.append(
                    f"symbol {tok} at pos {pos} -> value {self.display_value(pos, tok)} [T-1 fallback, {status}]"
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
        slim: bool = True,
        inductive: bool = False,
        decoding: bool = False,
    ) -> None:
        self.vocab_size = vocab_size
        self.max_sentence_length = max_sentence_length
        self.nbr_communication_rounds = nbr_communication_rounds
        self.nbr_latents = nbr_latents
        self.slim = slim
        self.inductive = inductive
        self.decoding = decoding
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
        feedback_text    = infos1.get("listener_exp_text")
        if self._last_msg_tokens is not None:
            self.tracker.observe_feedback(self._last_msg_tokens, feedback_latents, feedback_text)
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
        # Always capture tokens before any early return so observe_feedback()
        # can update _value_map from the feedback round.
        self._last_msg_tokens = list(target_utterance_widx.astype(int))
        self._last_listener_latents = listener_latents

        # The decode listener shares the inductive data regime (value-map only, no T-1
        # fallback) so a decode A/B is matched to the inductive baseline.
        value_map_only = self.inductive or self.decoding

        if value_map_only and not self.tracker._value_map:
            # No sync data yet: default to same class rather than T-1 guessing.
            action_dict["decision"][0, 0] = 0.0
            return action_dict

        n_match = 0
        for rel_i, t in enumerate(round_utterance):
            if t == 0:
                continue
            abs_i = pos_start + rel_i
            if abs_i >= len(obs_obj):
                continue
            if value_map_only:
                # No T-1 fallback: tokens absent from value_map count as non-matching.
                vm = self.tracker._value_map.get((abs_i, int(t)), {})
                if vm and int(obs_obj[abs_i]) == max(vm, key=vm.get):
                    n_match += 1
            else:
                if int(obs_obj[abs_i]) == self.tracker.best_value(abs_i, int(t)):
                    n_match += 1

        decision = 0.0 if n_match >= max(n_active, 1) else 1.0
        action_dict["decision"][0, 0] = decision
        self._last_decision = int(decision)

        return action_dict
