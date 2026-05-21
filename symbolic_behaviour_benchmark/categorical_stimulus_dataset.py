"""CategoricalStimulusDataset — replaces Gaussian floats with named category items.

Each latent dimension is assigned a distinct semantic category (e.g. vegetables,
colors, shapes). Stimuli are represented as integer indices (cast to float32) that
map to item names via latent_class_to_text().

Inherits from SymbolicContinuousStimulusDataset and overrides only the parts that
build and sample from the latent value space. All indexing, sampling-strategy, and
train/test split logic is reused unchanged.
"""
from typing import Dict

import numpy as np

from symbolic_behaviour_benchmark.symbolic_continuous_stimulus_dataset import (
    SymbolicContinuousStimulusDataset,
)

ADJECTIVE_POOL = [
    "big", "small", "large", "tiny", "huge", "old", "new", "young", "fast",
    "slow", "quick", "bright", "dark", "hard", "soft", "hot", "cold", "warm",
    "cool", "long", "short", "tall", "wide", "heavy", "light", "thick", "thin",
    "rough", "smooth", "sharp", "dull", "clean", "dirty", "sweet", "sour",
    "loud", "quiet", "deep", "shallow", "rich", "early", "late", "high", "low",
    "near", "far", "strong", "weak", "brave", "calm", "kind", "clever",
]

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
        self._adjectives = []
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

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _generate_adjectives(self, O: int) -> list:
        if O == 1:
            return ['']
        if O > len(ADJECTIVE_POOL):
            raise ValueError(
                f"nbr_object_centric_samples={O} exceeds ADJECTIVE_POOL size "
                f"({len(ADJECTIVE_POOL)}). Reduce O or extend ADJECTIVE_POOL."
            )
        indices = self._rng.choice(len(ADJECTIVE_POOL), size=O, replace=False)
        return [ADJECTIVE_POOL[i] for i in indices]

    # ── Overrides ──────────────────────────────────────────────────────────────

    def reset(self):
        if self.prototype is None:
            self.latent_dims = {}
            self.latent_sizes = []
            self.dataset_size = 1

            chosen_categories = self._rng.choice(
                list(CATEGORY_REGISTRY.keys()), size=self.nbr_latents, replace=False
            ).tolist()

            for l_idx in range(self.nbr_latents):
                cat_name = chosen_categories[l_idx]
                l_size = int(self._rng.integers(
                    self.min_nbr_values_per_latent,
                    self.max_nbr_values_per_latent + 1,
                ))
                items = self._rng.choice(CATEGORY_REGISTRY[cat_name], size=l_size, replace=False).tolist()
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

            self._adjectives = self._generate_adjectives(self.nbr_object_centric_samples)
            self.dataset_size *= self.nbr_object_centric_samples
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
            self._adjectives = self.prototype._adjectives

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
        O = self.nbr_object_centric_samples
        if O == 1 or object_centric_sample_idx is None:
            return latent_class.astype(np.float32)
        return (latent_class * O + int(object_centric_sample_idx)).astype(np.float32)

    def generate_observations(
        self,
        latent_class: np.ndarray,
        sample: bool = True,
    ) -> np.ndarray:
        """Return integer class indices as float32 (sample flag ignored)."""
        return latent_class.astype(np.float32)

    # ── Text conversion ────────────────────────────────────────────────────────

    def latent_class_to_text(self, flat_arr: np.ndarray) -> list:
        """Convert a flat float32 array of (possibly OC-encoded) indices to category names.

        For O=1: flat value is base_idx → category item name.
        For O>1: flat value is base_idx*O + oc_idx → item name with adjective randomly
            placed before or after (position re-randomised on each call).

        Args:
            flat_arr: 1-D array of float32 of length n * nbr_latents.

        Returns:
            List of n lists of strings, e.g. [['carrot', 'blue', 'triangle']] (O=1)
            or [['big carrot', 'blue old', 'smooth triangle']] (O>1, random position).
        """
        O = self.nbr_object_centric_samples
        nbr_latents = len(self.latent_dims)
        if len(flat_arr) % nbr_latents != 0:
            raise ValueError(
                f"flat_arr length {len(flat_arr)} is not divisible by "
                f"nbr_latents={nbr_latents}"
            )
        n = len(flat_arr) // nbr_latents
        result = []
        for i in range(n):
            group = flat_arr[i * nbr_latents:(i + 1) * nbr_latents]
            labels = []
            for lidx, v in enumerate(group):
                if O == 1:
                    base_idx, oc_idx = int(v), 0
                else:
                    base_idx, oc_idx = divmod(int(v), O)
                sections = self.latent_dims[lidx]['sections']
                if base_idx not in sections:
                    raise ValueError(
                        f"base_idx {base_idx} out of range for latent dim {lidx} "
                        f"(valid: 0–{len(sections)-1})"
                    )
                word = sections[base_idx]['name']
                adj = self._adjectives[oc_idx]
                if not adj:
                    labels.append(word)
                elif self._rng.integers(2) == 0:
                    labels.append(f"{adj} {word}")
                else:
                    labels.append(f"{word} {adj}")
            result.append(labels)
        return result
