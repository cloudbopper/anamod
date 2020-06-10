"""Generates model for simulation - perturbed version of ground truth polynomial"""

from functools import reduce

import numpy as np
import xxhash

from anamod import constants


class ModelWrapper():
    """Class implementing model API required by anamod"""
    def __init__(self, ground_truth_model, num_features, noise_type, noise_multiplier):
        self.ground_truth_model = ground_truth_model
        self.noisy = (noise_type != constants.NO_NOISE)
        if self.noisy:
            self.noise_type = noise_type
            self.irrelevant_features = self.get_irrelevant_features(num_features)
            self.rng = np.random.Generator(np.random.PCG64())
            self.rng_state = self.rng.__getstate__()
            self.noise_multiplier = noise_multiplier

    def get_irrelevant_features(self, num_features):
        """Return bit vector indicating irrelevance to model"""
        relevant_featureset = reduce(set.union, self.ground_truth_model.relevant_feature_map.keys(), set())
        irrelevant_features = [idx not in relevant_featureset for idx in range(num_features)]
        return irrelevant_features

    # pylint: disable = invalid-name
    def predict(self, X):
        """Perform prediction on input X (comprising one or more instances)"""
        if not self.noisy:
            return self.ground_truth_model.predict(X)
        noise = np.zeros(len(X))
        for idx, instance in enumerate(X):
            # The amount of noise is randomly chosen based on the instance
            # We want the noisy model to still be a function, i.e. a given instance should always return the same prediction
            # So we set the RNG state using the instance's hash value
            hashval = xxhash.xxh64_intdigest(instance)
            self.rng_state['state']['state'] = hashval
            self.rng_state['state']['inc'] = xxhash.xxh64_intdigest(str(hashval))
            self.rng.__setstate__(self.rng_state)
            if self.noise_type == constants.EPSILON_IRRELEVANT:
                # Add noise - small random non-zero coefficients for irrelevant features
                assert instance.ndim == 1, f"Noise type {constants.EPSILON_IRRELEVANT} only applicable for non-temporal feature representation"
                noise[idx] = np.dot(self.noise_multiplier * self.rng.uniform(-1, 1, size=(len(self.irrelevant_features))),
                                    self.irrelevant_features * instance)
            else:
                # Add noise - additive Gaussian, sampled for every instance/perturbed instance
                noise[idx] = self.rng.normal(0, self.noise_multiplier)
        return self.ground_truth_model.predict(X, noise=noise)
