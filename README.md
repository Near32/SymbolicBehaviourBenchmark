# SymbolicBehaviourBenchmark

Suite of OpenAI Gym-compatible multi-agent reinforcement learning environment centered around referntial games to benchmark for behavioral traits pertaining to symbolic behaviours, as described in [Santoro et al., 2021, "Symbolic Behaviours in Artificial Intelligence"](https://arxiv.org/abs/2102.03406), primarily: exhibiting receptive, constructive, malleable, and separable behaviours.

![default_env](https://www.github.com/Near32/SymbolicBehaviourBenchmark/raw/main/resources/symbolic-continuous-stimulus-representation-binding-problem-Descriptive%20%2B%20Listener%20Feedback.drawio.png)

## Usage

`gym` must be installed. Environments can be created as follows, for instance, in order to test for receptivity and constructivity:

```python
import gym
import symbolic_behaviour_benchmark
env = gym.make(
        "SymbolicBehaviourBenchmark-ReceptiveConstructiveTestEnv-v0", 
        nbr_communication_rounds = 1,
        vocab_size = 6,
        max_sentence_length = 3,
        descriptive = True,
        nbr_latents = 3,
        min_nbr_values_per_latent = 2,
        max_nbr_values_per_latent = 5,
        nbr_object_centric_samples = 4,
        nbr_distractors = 0,
        use_communication_channel_permutations = True,
        allow_listener_query = False,
        provide_listener_feedback = True,
        sampling_strategy = "component-focused-4shots",
    )
```

## Installation

### Installing via cloning this repository

```bash
git clone https://www.github.com/Near32/SymbolicBehaviourBenchmark
pip install -e ./SymbolicBehaviourBenchmark/
```
