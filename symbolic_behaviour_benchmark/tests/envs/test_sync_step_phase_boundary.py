import gym
import symbolic_behaviour_benchmark
import numpy as np
import random

from symbolic_behaviour_benchmark.utils.utils import STR2BT, BT2STR


def test_sync_step_converted_at_phase_boundary(
    vocab_size=6,
    max_sentence_length=3,
    nbr_latents=3,
    min_nbr_values_per_latent=3,
    max_nbr_values_per_latent=4,
    nbr_object_centric_samples=1,
    nbr_distractors=0,
    sampling_strategy='component-focused-1shot',
):
    """
    Verify that the sync step appended at the end of the last training game is
    correctly converted from present-tense form to past-tense / compact form
    ("In game #N, this is the exact stimulus that your partner was observing")
    once the test phase begins at game #0.

    Before the fix, _update_listener_prompt used `game_id-1` to locate the sync
    step in the accumulated prompt.  When game_id resets to 0 at the phase
    boundary, game_id-1 == -1, which never matches "#N", so the replace was a
    silent no-op and the stale present-tense block remained in the prompt.
    """
    np.random.seed(1)
    random.seed(1)

    env = gym.make(
        "SymbolicBehaviourBenchmark-ReceptiveConstructiveTestEnv-v0",
        vocab_size=vocab_size,
        max_sentence_length=max_sentence_length,
        descriptive=True,
        nbr_communication_rounds=1,
        nbr_latents=nbr_latents,
        min_nbr_values_per_latent=min_nbr_values_per_latent,
        max_nbr_values_per_latent=max_nbr_values_per_latent,
        nbr_object_centric_samples=nbr_object_centric_samples,
        nbr_distractors=nbr_distractors,
        use_communication_channel_permutations=False,
        allow_listener_query=False,
        provide_listener_feedback=True,
        sampling_strategy=sampling_strategy,
        include_prompts=True,
    )

    env.reset()

    speaker_action = {'decision': 0, 'communication_channel': np.ones((1, max_sentence_length)) * 3}
    listener_action = {'decision': 0, 'communication_channel': np.ones((1, max_sentence_length)) * 2}

    STALE_PATTERN = "here is a special step where you are given"
    GOOD_PATTERN  = "this is the exact stimulus that your partner was observing"

    done = False
    first_test_step_checked = False
    while not done:
        output = env.step(action=[speaker_action, listener_action])
        _, _, done, infos = output

        mode = infos[1].get('mode', '')
        stimulus_idx = infos[1]['stimulus_idx']
        listener_prompt_str = BT2STR(infos[1]['prompt'])[0]

        # Only check once: at the very first step of the test phase
        if mode == 'test' and stimulus_idx == 0 and not first_test_step_checked:
            first_test_step_checked = True
            assert STALE_PATTERN not in listener_prompt_str, (
                "Sync step was NOT converted: stale present-tense block still present "
                "in listener prompt at the start of the test phase.\n\n"
                f"Offending fragment found in prompt:\n{listener_prompt_str}"
            )
            assert GOOD_PATTERN in listener_prompt_str, (
                "Sync step conversion produced unexpected output: "
                f"'{GOOD_PATTERN}' not found in listener prompt.\n\n"
                f"Prompt tail:\n{listener_prompt_str[-800:]}"
            )

    env.close()
    assert first_test_step_checked, (
        "Test phase never reached — increase nbr_object_centric_samples or check env config."
    )


if __name__ == "__main__":
    test_sync_step_converted_at_phase_boundary()
    print("PASSED")
