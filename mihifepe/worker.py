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

import constants
from feature import Feature

def main():
    """Main"""
    parser = argparse.ArgumentParser()
    parser.add_argument("args_filename", help="pickle file containing arguments"
                        " passed by master.py")
    cargs = parser.parse_args()
    with open(cargs.args_filename, "r") as args_file:
        args = pickle.load(args_file)
    np.random.seed(constants.SEED)
    logging.basicConfig(level=logging.INFO, filename="%s/worker_%d.log" % (args.output_dir, args.task_idx),
                        format="%(asctime)s: %(message)s")
    logger = logging.getLogger(__name__)
    pipeline(args, logger)


def pipeline(args, logger):
    """Worker pipeline"""
    logger.info("Begin mihifepe worker pipeline")

    # categories = load_category_metadata(args, cur, logger)
    # map_codes(cur, logger, categories)
    # hierarchies = build_hierarchies(logger, categories)
    # codegroups = get_codegroups_from_file(args, logger, categories, hierarchies)
    # records, labels = load_data(args, cur, logger, categories)
    # test_fns, networks = build_networks(args, records, logger)
    # load_models(args, networks, logger)
    # get_num_int_codes(cur, categories)
    # get_embedding_functions(args, categories)
    # predictions, losses = get_predictions(logger, codegroups, records, test_fns)
    # write_outputs(args, labels, predictions, losses)

    # TODO: handle dynamic features
    # Load features to perturb from file
    features = load_features(args.features_filename)
    # Load data
    records = load_data(args.data_filename)
    # Load model
    model = load_model(logger, args.model_generator_filename)
    # Perturb features
    targets, losses, predictions = perturb_features(logger, features, records, model)
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
            node = Feature(row[constants.NODE_NAME], category=row[constants.CATEGORY],
                           static_indices=Feature.unpack_indices(row[constants.STATIC_INDICES]),
                           temporal_indices=Feature.unpack_indices(row[constants.TEMPORAL_INDICES]))
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
    records = hdf5_root[constants.RECORDS]
    return records


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
    basename, dirname = os.path.split(gen_model_filename)
    sys.path.insert(0, dirname)
    module = importlib.import_module(basename)
    model = getattr(module, "model")
    logger.info("End loading model")
    return model


def perturb_features(logger, features, records, model):
    """
    Perturbs features and observes effect on model loss

    Args:
        features:   list of features to perturb
        records:    HDF5 group containing data
        model:      model object passed by client

    Returns:
        targets:        (record -> target value)
                        mapping of record id's to target values
                        (classification labels or regression outputs, always scalars)
        losses:         (feature_id-> (record_id -> loss))
                        mapping of feature identifiers to a mapping of record id's to losses,
                        describing the losses of the model over the data with that feature perturbed
        predictions:    (feature_id -> (record_id -> prediction))
                        mapping feature identifiers to a mapping of record id's to predictions,
                        describing the predictions of the model over the data with that feature perturbed
    """
    # pylint: disable = too-many-locals
    logger.info("Begin perturbing features")
    num_records = len(records)
    targets = {}
    losses = {feature.identifier: {} for feature in features}
    predictions = {feature.identifier: {} for feature in features}
    for record_idx, record in enumerate(records):
        if record_idx % 100 == 0:
            logger.info("Begin processing record index %d of %d" % (record_idx, num_records))
        static_data = record[constants.STATIC]
        temporal_data = record[constants.TEMPORAL]
        target = record[constants.TARGET]
        targets[record.name] = target
        # Perturb each feature
        for feature in features:
            # TODO: configurable perturbations
            sdata = static_data
            tdata = temporal_data
            if static_data:
                sdata = np.copy(static_data)
                sdata[feature.static_indices] = 0
            if temporal_data:
                tdata = np.copy(temporal_data)
                tdata[feature.temporal_indices] = 0
            (loss, prediction) = model.predict(target, static_data=sdata, temporal_data=tdata)
            losses[feature.identifier][record.name] = loss
            predictions[feature.identifier][record.name] = prediction
    logger.info("End perturbing features")
    return targets, losses, predictions


def write_outputs(args, logger, targets, losses, predictions):
    """
    Write outputs to results file
    """
    logger.info("Begin writing outputs")
    record_ids = sorted(targets.keys()) # Sorting ensures record order is the same across workers
    results_filename = "%s/results_worker_%d.hdf5" % (args.output_dir, args.task_idx)
    root = h5py.File(results_filename, "w")

    def store_data(group, data):
        """Helper function to store data"""
        for feature_id, feature_data in data.items():
            group.create_dataset(feature_id, data=[feature_data[rid] for rid in record_ids])

    store_data(root.create_group(constants.LOSSES), losses)
    store_data(root.create_group(constants.PREDICTIONS), predictions)
    if args.task_idx == 0:
        root.create_dataset(constants.TARGETS, data=[targets[rid] for rid in record_ids])
    root.close()
    logger.info("End writing outputs")


if __name__ == "__main__":
    main()
