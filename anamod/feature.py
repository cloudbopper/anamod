"""Feature class"""
import anytree
import numpy as np
import xxhash

from anamod import constants


# pylint: disable = too-many-instance-attributes
class Feature(anytree.Node):
    """Class representing feature/feature group"""
    # TODO: add function to visualize fields nicely
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.parent_name = kwargs.get(constants.PARENT_NAME, "")
        self.description = kwargs.get(constants.DESCRIPTION, "")
        self.idx = kwargs.get("idx", [])
        self.perturbable = kwargs.get("perturbable", True)  # TODO: Use or discard
        # TODO: (Verify) Could initialize the RNG right away, since cloudpickle should still be able to pickle it
        self._rng_seed = xxhash.xxh32_intdigest(name)
        self.rng = None  # RNG used for permuting this feature - see perturbations.py: 'feature.rng'
        # p-value attributes
        self.overall_pvalue = 1.
        self.ordering_pvalue = 1.
        self.window_pvalue = 1.
        self.window_ordering_pvalue = 1.
        # Effect size attributes
        self.overall_effect_size = 0.
        self.window_effect_size = 0.
        # Importance attributes
        self.important = False
        self.ordering_important = False
        self.window_important = False
        self.window_ordering_important = False
        # Miscellaneous attributes
        self.temporal_window = None

    @property
    def rng_seed(self):
        """Get RNG seed"""
        return self._rng_seed

    @rng_seed.setter
    def rng_seed(self, seed):
        """Set RNG seed"""
        self._rng_seed = seed

    def initialize_rng(self):
        """Initialize random number generator for feature (used for permutations)"""
        self.rng = np.random.default_rng(self._rng_seed)

    def uniquify(self, uniquifier):
        """Add uniquifying identifier to name"""
        assert uniquifier
        self.name = "{0}->{1}".format(uniquifier, self.name)

    @property
    def size(self):
        """Return size"""
        return len(self.idx)
