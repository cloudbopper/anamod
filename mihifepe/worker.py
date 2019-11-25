"""
mihifepe worker pipeline
Given test data samples, a trained model and a set of feature groups, perturbs the features and
computes the effect on the model's output loss
"""

import argparse
import csv
import importlib
import logging
import os
import pickle
import sys

import h5py
import numpy as np

from mihifepe import constants
from mihifepe.feature import Feature


def main():
    """Main"""
    parser = argparse.ArgumentParser()
    parser.add_argument("args_filename", help="pickle file containing arguments"
                        " passed by master.py")
    cargs = parser.parse_args()
    with open(cargs.args_filename, "rb") as args_file:
        args = pickle.load(args_file)
    # np.random.seed(constants.SEED + args.task_idx)  # Enable if generating task-specific random numbers
    logging.basicConfig(level=logging.INFO, filename="%s/worker_%d.log" % (args.output_dir, args.task_idx),
                        format="%(asctime)s: %(message)s")
    logger = logging.getLogger(__name__)
    pipeline(args, logger)


def pipeline(args, logger):
    """Worker pipeline"""
    logger.info("Begin mihifepe worker pipeline")
    # Load features to perturb from file
    features = load_features(args.features_filename)
    # Load data
    records = load_data(args.data_filename)
    # Load model
    model = load_model(logger, args.model_generator_filename)
    # Perturb features
    targets, losses, predictions = perturb_features(args, logger, features, records, model)
    # Write outputs
    write_outputs(args, logger, targets, losses, predictions)
    logger.info("End mihifepe worker pipeline")


def load_features(features_filename):
    """
    Load features from file

    Args:
        features_filename: file containing list of features/feature groups to perturb

    Returns:
        list of features to perturb
    """
    features = []
    with open(features_filename, "r") as features_file:
        reader = csv.DictReader(features_file)
        for row in reader:
            node = Feature(row[constants.NODE_NAME], rng_seed=int(row[constants.RNG_SEED]),
                           static_indices=Feature.unpack_indices(row[constants.STATIC_INDICES]),
                           temporal_indices=Feature.unpack_indices(row[constants.TEMPORAL_INDICES]))
            node.initialize_rng()
            features.append(node)
    return features


def load_data(data_filename):
    """
    Load data from file.

    Args:
        data_filename: file in HDF5 format and specified structure (see mihifepe/spec.md) containing data samples

    Returns:
        data: HDF5 root group containing data
    """
    hdf5_root = h5py.File(data_filename, "r")
    return hdf5_root


def load_model(logger, gen_model_filename):
    """
    Load model object from model-generating python file.

    Args:
        gen_model_filename: name of standalone python file that provides model object

    Returns:
        model object.

    """
    logger.info("Begin loading model")
    if not os.path.exists(gen_model_filename):
        raise FileNotFoundError("Model-generating file not found")
    dirname, basename = os.path.split(gen_model_filename)
    sys.path.insert(0, dirname)
    module_name, _ = os.path.splitext(basename)
    module = importlib.import_module(module_name)
    model = getattr(module, "model")
    logger.info("End loading model")
    return model


def perturb_features(args, logger, features, hdf5_root, model):
    """
    Perturbs features and observes effect on model loss

    Args:
        args:       command-line arguments passed down from master
        logger:     logger
        features:   list of features to perturb
        hdf5_root:  HDF5 root object containing data
        model:      model object passed by client

    Returns:
        targets:        array of target values
                        (classification labels or regression outputs, always scalars)
        losses:         (feature_id-> loss))
                        mapping of feature names to loss vectors,
                        describing the losses of the model over the data with that feature perturbed
        predictions:    (feature_id-> prediction))
                        mapping of feature names to prediction vectors,
                        describing the predictions of the model over the data with that feature perturbed
    """
    logger.info("Begin perturbing features")
    perturber = Perturber(args, features, hdf5_root, model)
    for record_idx, _ in enumerate(perturber.record_ids):
        if record_idx % 100 == 0:
            logger.info("Begin processing record index %d of %d" % (record_idx + 1, perturber.num_records))
        # Perturb each feature for given record
        perturber.perturb_features_for_record(record_idx)
    logger.info("End perturbing features")
    return perturber.targets, perturber.losses, perturber.predictions


class Perturber():
    """Class to perform perturbations"""
    # pylint: disable = too-many-instance-attributes, len-as-condition
    # (Use len as it works with python lists as well as numpy arrays)
    def __init__(self, args, features, hdf5_root, model):
        self.args = args
        self.features = features
        self.model = model
        self.record_ids = hdf5_root[constants.RECORD_IDS].value
        self.targets = hdf5_root[constants.TARGETS].value
        self.static_dataset = hdf5_root[constants.STATIC].value
        self.temporal_grp = hdf5_root.get(constants.TEMPORAL)
        self.num_records = len(self.record_ids)
        self.static_data_input = bool(self.static_dataset.size)
        self.losses = {feature.name: np.zeros(self.num_records) for feature in self.features}
        self.predictions = {feature.name: np.zeros(self.num_records) for feature in self.features}

    def perturb_features_for_record(self, record_idx):
        """Perturbs all features for given record"""
        # Data
        target = self.targets[record_idx]
        static_data = self.static_dataset[record_idx] if self.static_data_input else []
        record_id = self.record_ids[record_idx]
        temporal_data = self.temporal_grp[record_id].value if self.temporal_grp else []
        # Perturb each feature
        for feature in self.features:
            tdata = self.perturb_temporal_data(feature, temporal_data)
            sdata = None
            if self.args.perturbation == constants.SHUFFLING:
                tvals = np.zeros((2, self.args.num_shuffling_trials))
                for trial in range(self.args.num_shuffling_trials):
                    sdata = self.perturb_static_data(feature, static_data)
                    tvals[:, trial] = self.model.predict(target, static_data=sdata, temporal_data=tdata)
                (loss, prediction) = np.average(tvals, axis=1)
            else:
                sdata = self.perturb_static_data(feature, static_data)
                (loss, prediction) = self.model.predict(target, static_data=sdata, temporal_data=tdata)
            # Update outputs
            self.losses[feature.name][record_idx] = loss
            self.predictions[feature.name][record_idx] = prediction

    def perturb_static_data(self, feature, static_data):
        """Perturb static data for given feature"""
        if len(static_data) == 0 or len(feature.static_indices) == 0:
            return static_data
        sdata = np.copy(static_data)
        if self.args.perturbation == constants.ZEROING:
            sdata[feature.static_indices] = 0
        elif self.args.perturbation == constants.SHUFFLING:
            replace_idx = feature.rng.randint(0, self.num_records)
            sdata[feature.static_indices] = self.static_dataset[replace_idx][feature.static_indices]
        return sdata

    def perturb_temporal_data(self, feature, temporal_data):
        """Perturb temporal data for given feature"""
        # TODO: verify that temporal data perturbation works.
        # temporal_data should be a list of vectors, but the perturbation below
        # treats it is a single vector
        # TODO: perturb_static_data and perturb_temporal_data are nearly identical - possibly merge.
        if len(temporal_data) == 0 or len(feature.temporal_indices) == 0:
            return temporal_data
        tdata = temporal_data
        if self.args.perturbation == constants.ZEROING:
            tdata = np.copy(temporal_data)
            tdata[feature.temporal_indices] = 0
        elif self.args.perturbation == constants.SHUFFLING:
            raise ValueError("Shuffling not valid for temporal data")
        return tdata


def write_outputs(args, logger, targets, losses, predictions):
    """Write outputs to results file"""
    logger.info("Begin writing outputs")
    results_filename = "%s/results_worker_%d.hdf5" % (args.output_dir, args.task_idx)
    root = h5py.File(results_filename, "w")

    def store_data(group, data):
        """Helper function to store data"""
        for feature_id, feature_data in data.items():
            group.create_dataset(feature_id, data=feature_data)

    store_data(root.create_group(constants.LOSSES), losses)
    store_data(root.create_group(constants.PREDICTIONS), predictions)
    if args.task_idx == 0:
        root.create_dataset(constants.TARGETS, data=targets)
    root.close()
    logger.info("End writing outputs")


if __name__ == "__main__":
    main()
