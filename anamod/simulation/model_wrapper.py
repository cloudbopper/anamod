"""Generates model for simulation - perturbed version of ground truth polynomial"""

from functools import reduce

import numpy as np
from pyhashxx import hashxx

from anamod import constants


class ModelWrapper():
    """Class implementing model API required by anamod"""
    # pylint: disable = too-few-public-methods, unused-argument

    def __init__(self, ground_truth_model, num_features, noise_type, noise_multiplier):
        self.ground_truth_model = ground_truth_model
        self.noisy = (noise_type != constants.NO_NOISE)
        if self.noisy:
            self.noise_type = noise_type
            self.noise_multiplier = noise_multiplier
            self.irrelevant_features = self.get_irrelevant_features(num_features)
            self.rng = np.random.RandomState(constants.SEED)

    def get_irrelevant_features(self, num_features):
        """Return bit vector indicating irrelevance to model"""
        relevant_featureset = reduce(set.union, self.ground_truth_model.relevant_feature_map.keys(), set())
        irrelevant_features = [idx not in relevant_featureset for idx in range(num_features)]
        return irrelevant_features

    # pylint: disable = invalid-name
    def predict(self, X):
        """Perform prediction on input X (comprising one or more instances)"""
        prediction = self.ground_truth_model.predict(X)
        if not self.noisy:
            return prediction
        noise = np.zeros(len(X))
        for idx, instance in enumerate(X):
            # The amount of noise is randomly chosen based on the instance
            hashval = hashxx(instance.tobytes())
            self.rng.seed(hashval)
            if self.noise_type == constants.EPSILON_IRRELEVANT:
                # Add noise - small random non-zero coefficients for irrelevant features
                noise[idx] = np.dot(self.noise_multiplier * self.rng.uniform(-1, 1, size=len(self.irrelevant_features)),
                                    self.irrelevant_features * instance)
            else:
                # Add noise - additive Gaussian, sampled for every instance/perturbed instance
                noise[idx] = self.rng.normal(0, self.noise_multiplier)
        return prediction + noise

    def loss(self, targets, predictions):
        """Compute loss for prediction-target pair (comprising one or more instances)"""
        return self.ground_truth_model.loss(targets, predictions)
