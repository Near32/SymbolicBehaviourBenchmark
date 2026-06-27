# SymbolicBehaviourBenchmark for Language Models — S2B-LM

> **You are on the `S2B-LM` branch.** This is an adaptation of S2B designed for language model evaluation. For the standard reinforcement learning benchmark, see the [`main` branch](https://github.com/Near32/SymbolicBehaviourBenchmark/tree/main).

**SymbolicBehaviourBenchmark (S2B)** is a suite of OpenAI Gym-compatible multi-agent reinforcement learning environments built around referential games, designed to operationalise and empirically evaluate the symbolic behaviour traits introduced by [Santoro et al. (2021)](https://arxiv.org/abs/2102.03406) — namely *receptivity*, *constructivity*, *malleability*, and *separability*. The benchmark itself is proposed and described in:

> Denamganai, K., Missaoui, S., & Walker, J. A. (2022). **Meta-Referential Games to Learn Compositional Learning Behaviours**. *arXiv preprint arXiv:2207.08012*. [https://arxiv.org/abs/2207.08012](https://arxiv.org/abs/2207.08012)

![default_env](https://www.github.com/Near32/SymbolicBehaviourBenchmark/raw/main/resources/symbolic-continuous-stimulus-representation-binding-problem-Descriptive+ListenerFeedback.drawio.png)

---

## Overview

Human beings use compositionality to generalise from past experiences to novel ones — decomposing experiences into fundamental atomic components that can be recombined in novel ways. We refer to behaviours making use of this ability as **Compositional Learning Behaviours (CLBs)**. Building artificial agents capable of exhibiting CLBs is a central challenge towards agents that can collaborate with human beings.

S2B is designed to investigate agents' abilities to exhibit CLBs in a domain-agnostic setting. Taking inspiration from the language emergence and grounding framework of referential games, it proposes a meta-learning extension — **Meta-Referential Games** — as the vehicle for this investigation. The benchmark exposes fine-grained control over vocabulary size, sentence length, latent structure, distractor count, and sampling strategy, enabling systematic evaluation of the symbolic behaviour traits characterised by [Santoro et al. (2021)](https://arxiv.org/abs/2102.03406): *receptivity*, *constructivity*, *malleability*, and *separability*.

---

## S2B-LM: Adaptation for Language Models

The `S2B-LM` branch adapts S2B for the evaluation of language models, as described in:

> Denamganai, K. (2025). **On Compositional Learning Behaviours in Formal Mathematics**. *OpenReview*. [https://openreview.net/forum?id=M3kKajgMpY](https://openreview.net/forum?id=M3kKajgMpY)

Two key modifications distinguish S2B-LM from the base benchmark:

- **Removal of numerical processing as a confounding factor.** Stimulus domains are rendered in purely categorical, language-friendly terms — concretely, the **categorical domain** (`domain='categorical'`) and the **pseudoword domain** (`domain='pseudoword'`) replace raw integer-valued latents with human-readable category labels and pseudoword tokens respectively. This ensures that any emergent competency reflects genuine compositional grounding rather than numerical pattern matching.

- **Chain-of-thought scaffolding.** Verbalization utilities (`PosdisHypothesisTracker`, `HypothesisListenerAgent`) provide structured natural-language reasoning traces — including *inductive* (value→symbol prediction) and *decoding* (symbol→value inference) modes — to actively elicit CLBs from the language model rather than passively measuring them.

---

## Installation

### Via repository clone (this branch)

```bash
git clone -b S2B-LM https://www.github.com/Near32/SymbolicBehaviourBenchmark
pip install -e ./SymbolicBehaviourBenchmark/
```

---

## Usage

`gym` must be installed. The following example instantiates an environment configured to evaluate *receptivity* and *constructivity* using the categorical domain:

```python
import gym
import symbolic_behaviour_benchmark

env = gym.make(
    "SymbolicBehaviourBenchmark-ReceptiveConstructiveTestEnv-v0",
    nbr_communication_rounds=1,
    vocab_size=6,
    max_sentence_length=3,
    descriptive=True,
    nbr_latents=3,
    min_nbr_values_per_latent=2,
    max_nbr_values_per_latent=5,
    nbr_object_centric_samples=4,
    nbr_distractors=0,
    use_communication_channel_permutations=True,
    allow_listener_query=False,
    provide_listener_feedback=True,
    sampling_strategy="component-focused-4shots",
    domain="categorical",
)
```

---

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@inproceedings{denamganai2026on,
  title={On Compositional Learning Behaviours in Formal Mathematics},
  author={Kevin Yandoka Denamganai},
  booktitle={3rd AI for Math Workshop: Toward Self-Evolving Scientific Agents},
  year={2026},
  url={https://openreview.net/forum?id=M3kKajgMpY}
}
```
