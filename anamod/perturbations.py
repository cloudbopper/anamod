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


class InstancePermutation(PerturbationFunction):
    """Shuffles instances. This isn't truly a permutation since we want no duplicates for any instance"""
    def __init__(self, rng, num_instances, num_permutations, *args):
        """
        Generate num_permutations of range(num_instances) without any duplicates for any sample.
        """
        super().__init__(*args)
        self._num_permutations = num_permutations
        assert self._num_permutations < num_instances
        self._current_permutation_idx = -1
        self._permutations = np.empty((num_instances, self._num_permutations), dtype=np.int32)
        for idx in range(num_instances):
            self._permutations[idx, :] = rng.choice([pidx for pidx in range(num_instances) if pidx != idx],
                                                    size=self._num_permutations, replace=False)

    def operate(self, X):
        self._current_permutation_idx += 1
        assert self._current_permutation_idx < self._num_permutations, f"Permutation object is configured for a maximum of {self._num_permutations}"
        return X[self._permutations[:, self._current_permutation_idx]]


class TimestepPermutation(PerturbationFunction):
    """
    Permutes timesteps. This needs to be a separate from InstancePermutation since here we want a true permutation (with duplicates allowed).
    Otherwise small windows have too few perturbations and biased averages
    """
    def __init__(self, rng, *args):
        super().__init__(*args)
        self._rng = rng

    def operate(self, X):
        self._rng.shuffle(X)
        return X


class PerturbationMechanism(ABC):
    """Performs perturbation"""
    def __init__(self, perturbation_fn, perturbation_type,
                 rng, num_instances, num_permutations):
        # pylint: disable = too-many-arguments
        assert issubclass(perturbation_fn, PerturbationFunction)
        self._perturbation_fn = perturbation_fn(rng, num_instances, num_permutations)
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
        return self._perturb(X_hat, idx, *args, **kwargs)

    def _perturb(self, X_hat, idx, *args, **kwargs):
        """Perturb feature for input data and given feature indices"""


class PerturbMatrix(PerturbationMechanism):
    """Perturb input arranged as matrix of instances X features"""
    def _perturb(self, X_hat, idx, *args, **kwargs):
        perturbed_slice = self._perturbation_fn.operate(X_hat[:, idx])
        X_hat[:, idx] = perturbed_slice
        return X_hat


class PerturbTensor(PerturbationMechanism):
    """Perturb input arranged as tensor of instances X features X time"""
    def _perturb(self, X_hat, idx, *args, **kwargs):
        timesteps = kwargs.get("timesteps", ...)
        axis0 = slice(None)  # all sequences
        axis1 = idx  # features to be perturbed
        axis2 = timesteps  # timesteps to be perturbed
        if self._perturbation_type == constants.WITHIN_INSTANCE:
            X_hat = np.transpose(X_hat)
            axis0, axis2 = axis2, axis0  # swap sequence and timestep axis for within-instance shuffling
        perturbed_slice = self._perturbation_fn.operate(X_hat[axis0, axis1, axis2])
        if self._perturbation_type == constants.WITHIN_INSTANCE and timesteps == ... and np.isscalar(idx):
            # Basic indexing - view was perturbed, so no assignment needed
            X_hat = X_hat.base
            assert perturbed_slice.base is X_hat
            return X_hat
        X_hat[axis0, axis1, axis2] = perturbed_slice
        return np.transpose(X_hat) if self._perturbation_type == constants.WITHIN_INSTANCE else X_hat


PERTURBATION_FUNCTIONS = {constants.ACROSS_INSTANCES: {constants.ZEROING: Zeroing, constants.SHUFFLING: InstancePermutation},
                          constants.WITHIN_INSTANCE: {constants.ZEROING: Zeroing, constants.SHUFFLING: TimestepPermutation}}
PERTURBATION_MECHANISMS = {constants.HIERARCHICAL: PerturbMatrix, constants.TEMPORAL: PerturbTensor}
