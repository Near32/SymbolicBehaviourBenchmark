# SymbolicBehaviourBenchmark

Suite of OpenAI Gym-compatible multi-agent reinforcement learning environment to benchmark for behavioral traits pertaining to symbolic behaviours (primarily: being receptive, constructive, malleable, and separable) using referential games.

![default_env](https://www.github.com/Near32/SymbolicBehaviourBenchmark/raw/main/resources/symbolic-behaviour-benchmark.png)

## Usage

`gym` must be installed. Environments can be created as follows, for instance, in order to test for receptivity and constructivity:

```python
>>> import gym
>>> import symbolic_behaviour_benchmark
>>> env = gym.make(
        "SymbolicBehaviourBenchmark-ReceptiveConstructiveTestEnv-v0", 
        vocab_size=10,
        max_sentence_length=5,
        nbr_latents=5,
        min_nbr_values_per_latent=3,
        max_nbr_values_per_latent=5,
        nbr_object_centric_samples=1,
        nbr_distractors=3,
        use_communication_channel_permutations=True,
        allow_listener_query=False,
    )
```

## Installation

### Installing via pip

This package is available in PyPi as `symbolic_behaviour_benchmark`

```bash
pip install symbolic_behaviour_benchmark
```

### Installing via cloning this repository

```bash
git clone https://www.github.com/Near32/SymbolicBehaviourBenchmark
cd SymbolicBehaviourBenchmark
pip install -e .
```
