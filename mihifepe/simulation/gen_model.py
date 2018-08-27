"""Script to load model from file"""

import os
import pickle

import numpy as np

from mihifepe.constants import GEN_MODEL_CONFIG_FILENAME
from mihifepe.simulation.model import Model

# pylint: disable = invalid-name
config_filename = "%s/%s" % (os.path.dirname(os.path.abspath(__file__)), GEN_MODEL_CONFIG_FILENAME)
with open(config_filename, "rb") as config_file:
    model_filename = pickle.load(config_file)
    noise_multiplier = pickle.load(config_file)
    poly_coeff = np.load(model_filename)
model = Model(poly_coeff, noise_multiplier)
