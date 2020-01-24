"""Classes for managing perturbations"""

from abc import ABC

import numpy as np


# pylint: disable = invalid-name
class PerturbationFunction(ABC):
    """Perturbation function base class"""
    def __init__(self, *args):
        pass

    def operate(self, X):
        """Operate on input"""


class Zeroing(PerturbationFunction):
    """Replace input values by zeros"""
    def operate(self, X):
        X[:] = 0
        return X


class Shuffling(PerturbationFunction):
    """Shuffles input values"""
    def __init__(self, rng, *args):
        super().__init__(*args)
        self._rng = rng

    def operate(self, X):
        self._rng.shuffle(X)
        return X


class PerturbationType(ABC):
    """Perturbation type base class"""
    # TODO: Across-instance vs within-across-instance


class PerturbationMechanism(ABC):
    """Performs perturbation"""
    def __init__(self, perturbation_fn, perturbation_type):
        self._perturbation_fn = perturbation_fn
        self._perturbation_type = perturbation_type  # TODO: use this field

    def perturb(self, X, feature, *args, **kwargs):
        """Perturb feature for input data"""


class PerturbMatrix(PerturbationMechanism):
    """Perturb input arranged as matrix of instances X features"""
    def perturb(self, X, feature, *args, **kwargs):
        X_hat = np.copy(X)
        perturbed_slice = self._perturbation_fn(feature.rng).operate(X_hat[:, feature.idx])
        if np.isscalar(feature.idx):
            # Basic indexing - view was perturbed, so no assignment needed
            assert perturbed_slice.base is X_hat
            return X_hat
        X_hat[:, feature.idx] = perturbed_slice
        return X_hat


class PerturbTensor(PerturbationMechanism):
    """Perturb input arranged as tensor of instances X features X time"""
    def perturb(self, X, feature, *args, **kwargs):
        timesteps = kwargs.get("timesteps", ...)
        X_hat = np.copy(X)
        perturbed_slice = self._perturbation_fn(X_hat[:, feature.idx, timesteps])
        if timesteps == ... and np.isscalar(feature.idx):
            # Basic indexing - view was perturbed, so no assignment needed
            assert perturbed_slice.base is X_hat
            return X_hat
        X_hat[:, feature.idx, timesteps] = perturbed_slice
        return X_hat
