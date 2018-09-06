"""Script to load model from file"""

import pickle

import numpy as np

from mihifepe.constants import ADDITIVE_GAUSSIAN, EPSILON_IRRELEVANT

# pylint: disable = invalid-name
config_filename = "GEN_MODEL_CONFIG_FILENAME_PLACEHOLDER" # This string gets replaced by name of config file during simulation
with open(config_filename, "rb") as config_file:
    model_filename = pickle.load(config_file)
    noise_multiplier = pickle.load(config_file)
    noise_type = pickle.load(config_file)
    poly_coeff = np.load(model_filename)

if noise_type == EPSILON_IRRELEVANT:
    from mihifepe.simulation.model import Model
    model = Model(poly_coeff, noise_multiplier)
elif noise_type == ADDITIVE_GAUSSIAN:
    from mihifepe.simulation.model_additive_gaussian_noise import Model
    model = Model(poly_coeff, noise_multiplier)
else:
    raise NotImplementedError("Unknown noise type")
