"""CategoricalStimulusDataset — replaces Gaussian floats with named category items.

Each latent dimension is assigned a distinct semantic category (e.g. vegetables,
colors, shapes). Stimuli are represented as integer indices (cast to float32) that
map to item names via latent_class_to_text().

Inherits from SymbolicContinuousStimulusDataset and overrides only the parts that
build and sample from the latent value space. All indexing, sampling-strategy, and
train/test split logic is reused unchanged.
"""
import random
from typing import Dict

import numpy as np

from symbolic_behaviour_benchmark.symbolic_continuous_stimulus_dataset import (
    SymbolicContinuousStimulusDataset,
)

CATEGORY_REGISTRY: Dict[str, list] = {
    "vegetables": ["carrot", "tomato", "potato", "onion", "broccoli",
                   "spinach", "cabbage", "pepper", "zucchini", "eggplant"],
    "fruits":     ["apple", "banana", "cherry", "grape", "mango",
                   "orange", "pear", "strawberry", "peach", "plum"],
    "colors":     ["red", "blue", "green", "yellow", "purple",
                   "orange", "pink", "brown", "cyan", "magenta"],
    "shapes":     ["circle", "triangle", "square", "pentagon", "hexagon",
                   "star", "diamond", "oval", "cross", "arrow"],
    "animals":    ["cat", "dog", "bird", "fish", "rabbit",
                   "horse", "elephant", "tiger", "lion", "wolf"],
    "countries":  ["france", "japan", "brazil", "canada", "india",
                   "egypt", "norway", "chile", "mexico", "poland"],
    "metals":     ["gold", "silver", "iron", "copper", "zinc",
                   "nickel", "platinum", "titanium", "lead", "tin"],
    "planets":    ["mercury", "venus", "earth", "mars", "jupiter",
                   "saturn", "uranus", "neptune", "pluto", "ceres"],
    "sports":     ["soccer", "tennis", "swimming", "cycling", "boxing",
                   "skiing", "golf", "hockey", "rugby", "baseball"],
    "instruments": ["piano", "guitar", "violin", "drums", "flute",
                    "trumpet", "cello", "harp", "oboe", "tuba"],
}


class CategoricalStimulusDataset(SymbolicContinuousStimulusDataset):
    """Stimulus dataset where each latent dimension is a named semantic category.

    Observations are integer indices (as float32) identifying which item within
    a dimension's category was selected.  Use latent_class_to_text() to recover
    the human-readable labels for prompt rendering.
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
    ):
        if nbr_object_centric_samples != 1:
            raise ValueError(
                f"domain='categorical' is incompatible with "
                f"nbr_object_centric_samples={nbr_object_centric_samples}. "
                "Categorical stimuli are deterministic (each label has exactly one "
                "instantiation), so object-centric sampling (O>1) is undefined. "
                "Set nbr_object_centric_samples=1."
            )
        if prototype is None and nbr_latents > len(CATEGORY_REGISTRY):
            raise ValueError(
                f"nbr_latents={nbr_latents} exceeds the number of available categories "
                f"({len(CATEGORY_REGISTRY)}). Add more categories to CATEGORY_REGISTRY "
                "or reduce nbr_latents."
            )
        super().__init__(
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

    # ── Overrides ──────────────────────────────────────────────────────────────

    def reset(self):
        if self.prototype is None:
            self.latent_dims = {}
            self.latent_sizes = []
            self.dataset_size = 1

            chosen_categories = random.sample(
                list(CATEGORY_REGISTRY.keys()), self.nbr_latents
            )

            for l_idx in range(self.nbr_latents):
                cat_name = chosen_categories[l_idx]
                l_size = np.random.randint(
                    low=self.min_nbr_values_per_latent,
                    high=self.max_nbr_values_per_latent + 1,
                )
                items = random.sample(CATEGORY_REGISTRY[cat_name], l_size)
                self.dataset_size *= l_size
                self.latent_sizes.append(l_size)

                self.latent_dims[l_idx] = {
                    'size': l_size,
                    'category_name': cat_name,
                    'sections': {s: {'name': items[s]} for s in range(l_size)},
                    # Fields required by reset_sampling() combinatorial split logic:
                    'nbr_fillers': 0,
                    'primitive': False,
                    'position': l_idx,
                    'remainder_use': 0,
                    'divider': 1,
                    'test_set_divider': self.test_set_divider,
                }

            self.dataset_size *= self.nbr_object_centric_samples  # always 1
            self.generate_object_centric_samples()  # no-op for categorical

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

    def generate_object_centric_samples(self):
        """No-op: categorical labels are deterministic, no pre-sampling needed."""
        pass

    def generate_object_centric_observations(
        self,
        latent_class: np.ndarray,
        object_centric_sample_idx: int = None,
    ) -> np.ndarray:
        """Return integer class indices as float32 (the index IS the observation)."""
        return latent_class.astype(np.float32)

    def generate_observations(
        self,
        latent_class: np.ndarray,
        sample: bool = True,
    ) -> np.ndarray:
        """Return integer class indices as float32 (sample flag ignored)."""
        return latent_class.astype(np.float32)

    # ── Text conversion ────────────────────────────────────────────────────────

    def latent_class_to_text(self, flat_arr: np.ndarray) -> list:
        """Convert a flat float32 array of integer-cast indices to category names.

        Args:
            flat_arr: 1-D array of float32 of length n * nbr_latents, where n is
                the number of stimuli (1 + nbr_distractors).

        Returns:
            List of n lists, each containing nbr_latents item-name strings.
            E.g. [['carrot', 'blue', 'triangle']] for n=1, nbr_latents=3.
        """
        nbr_latents = len(self.latent_dims)
        n = len(flat_arr) // nbr_latents
        result = []
        for i in range(n):
            group = flat_arr[i * nbr_latents:(i + 1) * nbr_latents]
            labels = [
                self.latent_dims[lidx]['sections'][int(v)]['name']
                for lidx, v in enumerate(group)
            ]
            result.append(labels)
        return result
