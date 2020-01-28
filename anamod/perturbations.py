"""Classes for managing perturbations"""

from abc import ABC

import numpy as np

from anamod import constants


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


class PerturbationMechanism(ABC):
    """Performs perturbation"""
    def __init__(self, perturbation_fn, perturbation_type):
        assert issubclass(perturbation_fn, PerturbationFunction)
        self._perturbation_fn = perturbation_fn
        assert perturbation_type in {constants.ACROSS_INSTANCES, constants.WITHIN_INSTANCE}
        self._perturbation_type = perturbation_type

    def perturb(self, X, feature, *args, **kwargs):
        """Perturb feature for input data and given feature(s)"""
        size = feature.size
        if size == 0:
            return X  # No feature(s) to be perturbed
        if size == 1:
            idx = feature.idx[0]  # To enable fast view-based indexing for singleton features
        else:
            idx = feature.idx
        X_hat = np.copy(X)
        return self._perturb(X_hat, idx, feature.rng, *args, **kwargs)

    def _perturb(self, X_hat, idx, rng, *args, **kwargs):
        """Perturb feature for input data and given feature indices"""


class PerturbMatrix(PerturbationMechanism):
    """Perturb input arranged as matrix of instances X features"""
    def _perturb(self, X_hat, idx, rng, *args, **kwargs):
        perturbed_slice = self._perturbation_fn(rng).operate(X_hat[:, idx])
        if np.isscalar(idx):
            # Basic indexing - view was perturbed, so no assignment needed
            assert perturbed_slice.base is X_hat
            return X_hat
        X_hat[:, idx] = perturbed_slice
        return X_hat


class PerturbTensor(PerturbationMechanism):
    """Perturb input arranged as tensor of instances X features X time"""
    def _perturb(self, X_hat, idx, rng, *args, **kwargs):
        timesteps = kwargs.get("timesteps", ...)
        axis0 = slice(None)  # all sequences
        axis1 = idx  # features to be perturbed
        axis2 = timesteps  # timesteps to be perturbed
        if self._perturbation_type == constants.WITHIN_INSTANCE:
            X_hat = np.transpose(X_hat)
            axis0, axis2 = axis2, axis0  # swap sequence and timestep axis for within-instance shuffling
        perturbed_slice = self._perturbation_fn(rng).operate(X_hat[axis0, axis1, axis2])
        if timesteps == ... and np.isscalar(idx):
            # Basic indexing - view was perturbed, so no assignment needed
            X_hat = np.transpose(X_hat) if self._perturbation_type == constants.WITHIN_INSTANCE else X_hat
            assert perturbed_slice.base is X_hat
            return X_hat
        X_hat[axis0, axis1, axis2] = perturbed_slice
        return np.transpose(X_hat) if self._perturbation_type == constants.WITHIN_INSTANCE else X_hat


PERTURBATION_FUNCTIONS = {constants.ZEROING: Zeroing, constants.SHUFFLING: Shuffling}
PERTURBATION_MECHANISMS = {constants.HIERARCHICAL: PerturbMatrix, constants.TEMPORAL: PerturbTensor}
