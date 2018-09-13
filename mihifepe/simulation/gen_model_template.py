"""Script to load model from file"""

import pickle

import numpy as np

from mihifepe.simulation import model

# pylint: disable = invalid-name
config_filename = "GEN_MODEL_CONFIG_FILENAME_PLACEHOLDER" # This string gets replaced by name of config file during simulation
with open(config_filename, "rb") as config_file:
    model_filename = pickle.load(config_file)
    noise_multiplier = pickle.load(config_file)
    noise_type = pickle.load(config_file)
    poly_coeff = np.load(model_filename)

model = model.Model(poly_coeff, noise_multiplier, noise_type)
