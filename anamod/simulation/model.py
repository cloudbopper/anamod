"""Generates model for simulation - perturbed version of ground truth polynomial"""

import numpy as np
from pyhashxx import hashxx

from anamod import constants


class Model():
    """Class implementing model API required by anamod"""
    # pylint: disable = too-few-public-methods, unused-argument

    def __init__(self, model_fn, noise_multiplier, noise_type):
        """Args:
            model_fn: function that returns model prediction given input vector and noise
            noise_multiplier: factor that determines amount of noise
            noise_type: indicator for type of noise
        """
        self.model_fn = model_fn
        self.noise_multiplier = noise_multiplier
        self.noise_type = noise_type
        self.rng = np.random.RandomState(constants.SEED)

    def predict(self, target, static_data, temporal_data):
        """
        Predicts the model's output (loss, prediction) for the given target and instance.
        In general, at least one of static_data and temporal_data must be non-empty.
        In this case, the model only uses static_data.

        Args:
            target:         classification label or regression output (scalar value)
            static_data:    static data (vector)
            temporal_data:  temporal data (matrix, where number of rows are variable across instances)

        Returns:
            loss:           model's output loss
            prediction:     model's output prediction, only used for classifiers
        """
        hashval = hashxx(static_data.data.tobytes())
        self.rng.seed(hashval)
        if self.noise_type == constants.NO_NOISE:
            prediction = self.model_fn(static_data, [])
        elif self.noise_type == constants.EPSILON_IRRELEVANT:
            # Add noise - small random non-zero coefficients for irrelevant features
            noise = self.noise_multiplier * self.rng.uniform(-1, 1, static_data.size)
            prediction = self.model_fn(static_data, noise)
        elif self.noise_type == constants.ADDITIVE_GAUSSIAN:
            # Add noise - additive Gaussian, sampled for every instance/perturbed instance
            prediction = self.model_fn(static_data, self.rng.normal(0, self.noise_multiplier))
        else:
            raise NotImplementedError("Unknown noise type")
        loss = self.loss(prediction, target)
        return (loss, prediction)

    @staticmethod
    def loss(prediction, target):
        """Compute RMSE"""
        # Single-sample RMSE is just the absolute value
        return np.abs(prediction - target)
