"""
anamod worker pipeline
Given test data samples, a trained model and a set of feature groups, perturbs the features and
computes the effect on the model's output loss
"""

import argparse
import csv
import pickle

import cloudpickle
import h5py
import numpy as np

from anamod import constants, utils
from anamod.feature import Feature


def main():
    """Main"""
    parser = argparse.ArgumentParser()
    parser.add_argument("args_filename", help="pickle file containing arguments"
                        " passed by master.py")
    cargs = parser.parse_args()
    with open(cargs.args_filename, "rb") as args_file:
        args = pickle.load(args_file)
    args.logger = utils.get_logger(__name__, "%s/worker_%d.log" % (args.output_dir, args.task_idx))
    pipeline(args)


def pipeline(args):
    """Worker pipeline"""
    args.logger.info("Begin anamod worker pipeline")
    # Load features to perturb from file
    features = load_features(args.features_filename)
    # Load data
    records = load_data(args.data_filename)
    # Load model
    model = load_model(args)
    # Perturb features
    targets, losses, predictions = perturb_features(args, features, records, model)
    # Write outputs
    write_outputs(args, targets, losses, predictions)
    args.logger.info("End anamod worker pipeline")


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
        data_filename: file in HDF5 format and specified structure (see anamod/spec.md) containing data samples

    Returns:
        data: HDF5 root group containing data
    """
    hdf5_root = h5py.File(data_filename, "r")
    return hdf5_root


def load_model(args):
    """
    Load model object from model-generating python file.

    Returns:
        model object.

    """
    args.logger.info("Begin loading model")
    with open(args.model_filename, "rb") as model_file:
        model = cloudpickle.load(model_file)
    args.logger.info("End loading model")
    return model


def perturb_features(args, features, hdf5_root, model):
    """
    Perturbs features and observes effect on model loss

    Args:
        args:       command-line arguments passed down from master
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
    args.logger.info("Begin perturbing features")
    perturber = Perturber(args, features, hdf5_root, model)
    for record_idx, _ in enumerate(perturber.record_ids):
        if record_idx % 100 == 0:
            args.logger.info("Begin processing record index %d of %d" % (record_idx + 1, perturber.num_records))
        # Perturb each feature for given record
        perturber.perturb_features_for_record(record_idx)
    args.logger.info("End perturbing features")
    return perturber.targets, perturber.losses, perturber.predictions


class Perturber():
    """Class to perform perturbations"""
    # pylint: disable = too-many-instance-attributes, len-as-condition
    # (Use len as it works with python lists as well as numpy arrays)
    def __init__(self, args, features, hdf5_root, model):
        self.args = args
        self.features = features
        self.model = model
        self.record_ids = hdf5_root[constants.RECORD_IDS][...]
        self.targets = hdf5_root[constants.TARGETS][...]
        self.static_dataset = hdf5_root[constants.STATIC][...]
        self.num_records = len(self.record_ids)
        self.static_data_input = bool(self.static_dataset.size)
        self.losses = {feature.name: np.zeros(self.num_records) for feature in self.features}
        self.predictions = {feature.name: np.zeros(self.num_records) for feature in self.features}

    def perturb_features_for_record(self, record_idx):
        """Perturbs all features for given record"""
        # TODO: revise handling of temporal vs. static data
        # The model doesn't know about the perturbation, it just takes the data input
        # We can also assume that the data format matches what model.predict expects,
        # so it shouldn't receive static/temporal data separately
        # Only the perturbation mechanism needs to be aware of static vs. temporal
        # so that it can apply the perturbation appropriately
        # Data
        target = self.targets[record_idx]
        static_data = self.static_dataset[record_idx] if self.static_data_input else []
        # Perturb each feature
        for feature in self.features:
            sdata = None
            if self.args.perturbation == constants.SHUFFLING:
                tvals = np.zeros((2, self.args.num_shuffling_trials))
                for trial in range(self.args.num_shuffling_trials):
                    sdata = self.perturb_static_data(feature, static_data)
                    pred = self.model.predict(sdata.reshape(1, -1))[0]
                    loss = self.model.loss(target, pred)
                    tvals[:, trial] = (pred, loss)
                (pred, loss) = np.average(tvals, axis=1)
            else:
                sdata = self.perturb_static_data(feature, static_data)
                pred = self.model.predict(sdata.reshape(1, -1))[0]
                loss = self.model.loss(target, pred)
            # Update outputs
            self.predictions[feature.name][record_idx] = pred
            self.losses[feature.name][record_idx] = loss

    def perturb_static_data(self, feature, static_data):
        """Perturb static data for given feature"""
        if len(static_data) == 0 or len(feature.static_indices) == 0:
            return static_data
        sdata = np.copy(static_data)
        if self.args.perturbation == constants.ZEROING:
            sdata[feature.static_indices] = 0
        elif self.args.perturbation == constants.SHUFFLING:
            replace_idx = feature.rng.integers(0, self.num_records)
            sdata[feature.static_indices] = self.static_dataset[replace_idx][feature.static_indices]
        return sdata

    def perturb_temporal_data(self, feature, temporal_data):
        """Perturb temporal data for given feature"""
        # FIXME: currently unused - temporal data will not be perturbed
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


def write_outputs(args, targets, losses, predictions):
    """Write outputs to results file"""
    args.logger.info("Begin writing outputs")
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
    args.logger.info("End writing outputs")


if __name__ == "__main__":
    main()
