"""Model-generating file"""

import os
import pickle

import numpy as np

from . import constants

class Model():
    """Class encapsulating model API required by mihifepe"""
    # pylint: disable = too-few-public-methods, unused-argument
    def __init__(self):
        config_filename = "%s/%s" % (os.path.dirname(os.path.abspath(__file__)), constants.GEN_MODEL_CONFIG_FILENAME)
        with open(config_filename, "rb") as config_file:
            model_filename = pickle.load(config_file)
            self.poly_coeff = np.load(model_filename)

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
        prediction = np.dot(static_data, self.poly_coeff)
        loss = np.sqrt(np.pow(prediction - target, 2)) # RMSE
        return (loss, prediction)


# pylint: disable = invalid-name
model = Model()
