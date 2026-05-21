"""PseudowordStimulusDataset — dynamically generated (CV)+ pseudowords as stimuli.

Object-centric sampling (O>1) is supported: a shared set of O suffix strings is
generated once per episode and appended to every base word. All latent values across
all dimensions use the same suffix set, so the suffix is purely an OC nuisance and
carries no categorical information.

Encoding for O>1: the float observation per latent = base_idx * O + oc_idx.
latent_class_to_text() decodes this back to base_word + suffix for the LM prompt.
"""
import math

import numpy as np

from symbolic_behaviour_benchmark.categorical_stimulus_dataset import CategoricalStimulusDataset
from symbolic_behaviour_benchmark.symbolic_continuous_stimulus_dataset import (
    SymbolicContinuousStimulusDataset,
)

VOWELS = list('aeiou')
CONSONANTS = list('bdgklmnprst')
ALPHABET = CONSONANTS + VOWELS  # 16 chars — used for suffix generation


class PseudowordStimulusDataset(CategoricalStimulusDataset):
    """Stimulus dataset using dynamically generated (CV)+ pseudowords.

    Each latent dimension gets a fresh set of pseudowords on every reset().
    Words are UPPERCASE and globally unique within an episode.
    Grammar: (CV)+ — strictly alternating consonant-vowel pairs.

    Object-centric samples (O>1): a shared set of O suffix strings is generated
    alongside the base words. Suffix length = ceil(log_16(O)), guaranteeing enough
    expressivity. The same suffixes apply to every latent value across every dimension,
    so they act as a shared nuisance rather than a category signal.
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
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        self._suffixes = []  # set during reset()
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

    # ── Word / suffix generation ───────────────────────────────────────────────

    def _generate_pseudoword(self) -> str:
        """Return a lowercase (CV)+ string (uppercasing happens at storage time)."""
        min_pairs = max(1, self.min_word_length // 2)
        max_pairs = max(min_pairs, self.max_word_length // 2)
        n_pairs = int(self._rng.integers(min_pairs, max_pairs + 1))
        chars = []
        for _ in range(n_pairs):
            chars.append(CONSONANTS[int(self._rng.integers(len(CONSONANTS)))])
            chars.append(VOWELS[int(self._rng.integers(len(VOWELS)))])
        return ''.join(chars)

    def _generate_suffixes(self, O: int) -> list:
        """Generate O distinct UPPERCASE suffix strings of minimum necessary length."""
        if O == 1:
            return ['']
        suffix_len = max(1, math.ceil(math.log(O) / math.log(len(ALPHABET))))
        used = set()
        suffixes = []
        while len(suffixes) < O:
            s = ''.join(
                ALPHABET[int(self._rng.integers(len(ALPHABET)))] for _ in range(suffix_len)
            ).upper()
            if s not in used:
                used.add(s)
                suffixes.append(s)
        return suffixes

    # ── Override ───────────────────────────────────────────────────────────────

    def reset(self):
        O = self.nbr_object_centric_samples
        if self.prototype is None:
            self.latent_dims = {}
            self.latent_sizes = []
            self.dataset_size = 1
            used_words = set()  # lowercase generated form for collision detection

            for l_idx in range(self.nbr_latents):
                l_size = int(self._rng.integers(
                    self.min_nbr_values_per_latent,
                    self.max_nbr_values_per_latent + 1,
                ))
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

            # Shared suffix set — same for every latent value across all dimensions
            self._suffixes = self._generate_suffixes(O)

            self.dataset_size *= O

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
            self._suffixes = self.prototype._suffixes  # share exact same suffix set

        self.targets = np.zeros(self.dataset_size)
        for idx in range(self.dataset_size):
            self.targets[idx] = idx // O

        self.reset_sampling()
        self.reset_OC_classes()

    def generate_object_centric_samples(self):
        """No-op: pseudoword OC samples are constructed on the fly via suffixes."""
        pass

    def generate_object_centric_observations(
        self,
        latent_class: np.ndarray,
        object_centric_sample_idx: int = None,
    ) -> np.ndarray:
        """Encode base word index and OC sample as a single float per latent.

        float_value = base_idx * O + oc_idx
        latent_class_to_text() decodes this back to base_word + suffix.
        """
        O = self.nbr_object_centric_samples
        if O == 1 or object_centric_sample_idx is None:
            return latent_class.astype(np.float32)
        return (latent_class * O + int(object_centric_sample_idx)).astype(np.float32)

    def generate_observations(
        self,
        latent_class: np.ndarray,
        sample: bool = True,
    ) -> np.ndarray:
        """Return base integer class indices (no OC encoding)."""
        return latent_class.astype(np.float32)

    # ── Text conversion ────────────────────────────────────────────────────────

    def latent_class_to_text(self, flat_arr: np.ndarray) -> list:
        """Convert encoded float observations to pseudoword+suffix strings.

        For O=1: flat value is base_idx → base_word (no suffix).
        For O>1: flat value is base_idx*O + oc_idx → base_word + suffix[oc_idx].

        Args:
            flat_arr: 1-D float32 array of length n * nbr_latents.

        Returns:
            List of n lists of strings, e.g. [['BANE', 'KODI', 'NILU']] (O=1)
            or [['BANEB', 'KODIE', 'NILUG']] (O>1, suffix appended).
        """
        O = self.nbr_object_centric_samples
        nbr_latents = len(self.latent_dims)
        if len(flat_arr) % nbr_latents != 0:
            raise ValueError(
                f"flat_arr length {len(flat_arr)} is not divisible by nbr_latents={nbr_latents}"
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
                labels.append(sections[base_idx]['name'] + self._suffixes[oc_idx])
            result.append(labels)
        return result
