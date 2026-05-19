"""PseudowordStimulusDataset — dynamically generated (CV)+ pseudowords as stimuli."""
import numpy as np

from symbolic_behaviour_benchmark.categorical_stimulus_dataset import CategoricalStimulusDataset
from symbolic_behaviour_benchmark.symbolic_continuous_stimulus_dataset import (
    SymbolicContinuousStimulusDataset,
)

VOWELS = list('aeiou')
CONSONANTS = list('bdgklmnprst')


class PseudowordStimulusDataset(CategoricalStimulusDataset):
    """Stimulus dataset using dynamically generated (CV)+ pseudowords.

    Each latent dimension gets a fresh set of pseudowords on every reset().
    Words are UPPERCASE and globally unique within an episode.
    Grammar: (CV)+ — strictly alternating consonant-vowel pairs.
    """

    def __init__(
        self,
        train=True,
        transform=None,
        sampling_strategy=None,
        split_strategy=None,
        nbr_latents=10,
        min_nbr_values_per_latent=2,
        max_nbr_values_per_latent=10,
        nbr_object_centric_samples=1,
        prototype=None,
        min_word_length=2,
        max_word_length=6,
    ):
        if nbr_object_centric_samples != 1:
            raise ValueError(
                f"domain='pseudoword' is incompatible with "
                f"nbr_object_centric_samples={nbr_object_centric_samples}. "
                "Pseudoword stimuli have exactly one instantiation per label, "
                "so object-centric sampling (O>1) is undefined. Set nbr_object_centric_samples=1."
            )
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        # Bypass CategoricalStimulusDataset.__init__ to skip registry-size check
        SymbolicContinuousStimulusDataset.__init__(
            self,
            train=train,
            transform=transform,
            sampling_strategy=sampling_strategy,
            split_strategy=split_strategy,
            nbr_latents=nbr_latents,
            min_nbr_values_per_latent=min_nbr_values_per_latent,
            max_nbr_values_per_latent=max_nbr_values_per_latent,
            nbr_object_centric_samples=nbr_object_centric_samples,
            prototype=prototype,
        )

    def _generate_pseudoword(self) -> str:
        min_pairs = max(1, self.min_word_length // 2)
        max_pairs = max(min_pairs, self.max_word_length // 2)
        n_pairs = np.random.randint(min_pairs, max_pairs + 1)
        chars = []
        for _ in range(n_pairs):
            chars.append(CONSONANTS[np.random.randint(len(CONSONANTS))])
            chars.append(VOWELS[np.random.randint(len(VOWELS))])
        return ''.join(chars)

    def reset(self):
        if self.prototype is None:
            self.latent_dims = {}
            self.latent_sizes = []
            self.dataset_size = 1
            used_words = set()  # lowercase generated form for collision detection

            for l_idx in range(self.nbr_latents):
                l_size = np.random.randint(
                    self.min_nbr_values_per_latent,
                    self.max_nbr_values_per_latent + 1,
                )
                words = []
                while len(words) < l_size:
                    w = self._generate_pseudoword()
                    if w not in used_words:
                        used_words.add(w)
                        words.append(w.upper())

                self.dataset_size *= l_size
                self.latent_sizes.append(l_size)
                self.latent_dims[l_idx] = {
                    'size': l_size,
                    'sections': {s: {'name': words[s]} for s in range(l_size)},
                    'nbr_fillers': 0,
                    'primitive': False,
                    'position': l_idx,
                    'remainder_use': 0,
                    'divider': 1,
                    'test_set_divider': self.test_set_divider,
                }

            self.dataset_size *= self.nbr_object_centric_samples  # always 1
            self.generate_object_centric_samples()

            self.latent_strides = [1]
            dims = [ld['size'] for ld in self.latent_dims.values()]
            for idx in range(self.nbr_latents):
                self.latent_strides.append(int(np.prod(dims[-idx - 1:])))
            self.latent_strides = list(reversed(self.latent_strides[:-1]))

            self.test_latents_mask = np.zeros((self.dataset_size, self.nbr_latents))
        else:
            self.latent_dims = self.prototype.latent_dims
            self.latent_sizes = self.prototype.latent_sizes
            self.dataset_size = self.prototype.dataset_size
            self.latent_strides = self.prototype.latent_strides
            self.test_latents_mask = self.prototype.test_latents_mask

        self.targets = np.zeros(self.dataset_size)
        for idx in range(self.dataset_size):
            self.targets[idx] = idx // self.nbr_object_centric_samples

        self.reset_sampling()
        self.reset_OC_classes()
