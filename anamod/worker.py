"""
anamod worker pipeline
Given test data samples, a trained model and a set of feature groups, perturbs the features and
computes the effect on the model's output loss
"""

import argparse
from collections import namedtuple
import importlib
import os
import pickle
import socket
import sys

import cloudpickle
import h5py
import numpy as np

from anamod import constants
from anamod.compute_p_values import compute_p_value
from anamod.losses import Loss
from anamod.perturbations import PERTURBATION_FUNCTIONS, PERTURBATION_MECHANISMS
from anamod.utils import get_logger, round_value

Inputs = namedtuple("Inputs", ["data", "targets", "model"])


def main():
    """Main"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-features_filename", required=True)
    parser.add_argument("-model_filename", required=True)
    parser.add_argument("-data_filename", required=True)
    parser.add_argument("-output_dir", required=True)
    parser.add_argument("-model_loader_filename", required=True)
    parser.add_argument("-analysis_type", required=True)
    parser.add_argument("-perturbation", required=True)
    parser.add_argument("-num_shuffling_trials", required=True, type=int)
    parser.add_argument("-worker_idx", required=True, type=int)
    parser.add_argument("-loss_function", required=True, type=str)
    parser.add_argument("-loss_target_values", required=True, type=str)
    args = parser.parse_args()
    args.logger = get_logger(__name__, "%s/worker_%d.log" % (args.output_dir, args.worker_idx))
    pipeline(args)


def pipeline(args):
    """Worker pipeline"""
    args.logger.info(f"Begin anamod worker pipeline on host {socket.gethostname()}")
    # Load features to perturb from file
    features = load_features(args.features_filename)
    # Load data
    data, targets = load_data(args.data_filename)
    # Load model
    model = load_model(args)
    inputs = Inputs(data, targets, model)
    # Baseline predictions/losses
    # FIXME: baseline may be computed in master and provided to all workers
    _, baseline_loss, loss_fn = compute_baseline(args, inputs)
    # Perturb features
    predictions, losses = perturb_features(args, inputs, features, loss_fn)
    compute_importances(features, losses, baseline_loss)
    # For important features, proceed with further analysis (temporal model analysis):
    if args.analysis_type == constants.TEMPORAL:
        temporal_analysis(args, inputs, features, baseline_loss, loss_fn)
    # Write outputs
    write_outputs(args, features, predictions)
    args.logger.info("End anamod worker pipeline")


def load_features(features_filename):
    """Load feature/feature groups to test from file"""
    with open(features_filename, "rb") as features_file:
        features = cloudpickle.load(features_file)
    for feature in features:
        feature.initialize_rng()
    return features


def load_data(data_filename):
    """Load data from HDF5 file"""
    data_root = h5py.File(data_filename, "r")
    data = data_root[constants.DATA][...]
    targets = data_root[constants.TARGETS][...]
    return data, targets


def load_model(args):
    """Load model object from model-generating python file"""
    args.logger.info("Begin loading model")
    dirname, filename = os.path.split(os.path.abspath(args.model_loader_filename))
    sys.path.insert(1, dirname)
    loader = importlib.import_module(os.path.splitext(filename)[0])
    model = loader.load_model(args.model_filename)
    args.logger.info("End loading model")
    return model


def compute_baseline(args, inputs):
    """Compute baseline prediction/loss"""
    data, targets, model = inputs
    pred = model.predict(data)
    if args.loss_target_values == constants.BASELINE_PREDICTIONS:
        targets = pred
    loss_fn = Loss(args.loss_function, targets).loss_fn
    baseline_loss = loss_fn(pred)
    return pred, baseline_loss, loss_fn


def get_perturbation_mechanism(args, perturbation_type=constants.ACROSS_INSTANCES):
    """Get appropriately configured object to perform perturbations"""
    perturbation_fn_class = PERTURBATION_FUNCTIONS[args.perturbation]
    perturbation_mechanism_class = PERTURBATION_MECHANISMS[args.analysis_type]
    return perturbation_mechanism_class(perturbation_fn_class, perturbation_type)


def perturb_features(args, inputs, features, loss_fn):
    """Perturb features"""
    # TODO: Perturbation modules should be provided as input so custom modules may be used
    args.logger.info("Begin perturbing features")
    predictions = {feature.name: np.zeros(len(inputs.targets)) for feature in features}
    losses = {feature.name: np.zeros(len(inputs.targets)) for feature in features}
    perturbation_mechanism = get_perturbation_mechanism(args)
    # Perturb each feature
    for feature in features:
        prediction, loss = perturb_feature(args, inputs, feature, perturbation_mechanism, loss_fn)
        predictions[feature.name] = prediction
        losses[feature.name] = loss
    args.logger.info("End perturbing features")
    return predictions, losses


def perturb_feature(args, inputs, feature, perturbation_mechanism, loss_fn, timesteps=...):
    """Perturb feature"""
    # pylint: disable = too-many-arguments
    data, targets, model = inputs
    if args.perturbation == constants.SHUFFLING:
        prediction = np.zeros(len(targets))
        loss = np.zeros(len(targets))
        for _ in range(args.num_shuffling_trials):
            data_perturbed = perturbation_mechanism.perturb(data, feature, timesteps=timesteps)
            pred = model.predict(data_perturbed)
            prediction += pred
            loss += loss_fn(pred)
        prediction /= args.num_shuffling_trials
        loss /= args.num_shuffling_trials
    else:
        data_perturbed = perturbation_mechanism.perturb(data, feature, timesteps=timesteps)
        prediction = model.predict(data_perturbed)
        loss = loss_fn(prediction)
    return prediction, loss


def search_window(args, inputs, feature, perturbation_mechanism, baseline_loss, loss_fn):
    """Search temporal window of importance for given feature"""
    # pylint: disable = too-many-arguments
    args.logger.info("Begin searching for temporal window for feature %s" % feature.name)
    # TODO: verify this will work if window spans the whole sequence
    T = inputs.data.shape[2]  # pylint: disable = invalid-name
    # Search left edge
    lbound, current, rbound = (0, T // 2, T)
    while current < rbound:
        # Search for the largest 'negative' window anchored on the left that results in a non-important p-value
        # The right edge of the resulting window is the left edge of the window of interest
        _, loss = perturb_feature(args, inputs, feature, perturbation_mechanism, loss_fn, range(0, current))
        important = compute_p_value(baseline_loss, loss) < constants.PVALUE_THRESHOLD
        if important:
            # Move pointer to the left, decrease negative window size
            rbound = current
            current = max(current // 2, lbound)
        else:
            # Move pointer to the right, increase negative window size
            lbound = current
            current = max(current + 1, (current + rbound) // 2)
    left = current - 1  # range(0, current) = 0, 1, ... current - 1
    # Search right edge
    lbound, current, rbound = (left, (left + T) // 2, T)
    while lbound < current:
        _, loss = perturb_feature(args, inputs, feature, perturbation_mechanism, loss_fn, range(current, T))
        important = compute_p_value(baseline_loss, loss) < constants.PVALUE_THRESHOLD
        if important:
            # Move pointer to the right, decrease negative window size
            lbound = current
            current = (current + rbound) // 2
        else:
            # Move pointer to the left, increase negative window size
            rbound = current
            current = (current + lbound) // 2
    right = current
    return left, right


def compute_importances(features, losses, baseline_loss):
    """Computes p-values indicating feature importances"""
    mean_baseline_loss = np.mean(baseline_loss)
    for feature in features:
        loss = losses[feature.name]
        mean_loss = np.mean(loss)
        pvalue_loss = compute_p_value(baseline_loss, loss)
        effect_size = mean_loss - mean_baseline_loss
        feature.effect_size = round_value(effect_size)
        feature.mean_loss = round_value(mean_loss)
        feature.pvalue_loss = round_value(pvalue_loss)
        feature.important = feature.pvalue_loss < constants.PVALUE_THRESHOLD


def temporal_analysis(args, inputs, features, baseline_loss, loss_fn):
    """Perform temporal analysis of important features"""
    # FIXME: hardcoded p-value threshold
    features = list(filter(lambda feature: feature.pvalue_loss < constants.PVALUE_THRESHOLD, features))  # select (overall) important features
    args.logger.info("Identified important features: %s; proceeding with temporal analysis" % ",".join([feature.name for feature in features]))
    perturbation_mechanism_within = get_perturbation_mechanism(args, perturbation_type=constants.WITHIN_INSTANCE)
    perturbation_mechanism_across = get_perturbation_mechanism(args, perturbation_type=constants.ACROSS_INSTANCES)
    for feature in features:
        # Test importance of feature ordering
        _, loss = perturb_feature(args, inputs, feature, perturbation_mechanism_within, loss_fn)
        feature.temporally_important = compute_p_value(baseline_loss, loss) < constants.PVALUE_THRESHOLD
        args.logger.info(f"Feature {feature.name}: ordering important: {feature.temporally_important}")
        # Test feature localization
        # TODO: figure out why within-instance perturbations to search window fail so haphazardly
        left, right = search_window(args, inputs, feature, perturbation_mechanism_across, baseline_loss, loss_fn)
        feature.temporal_window = (left, right)
        args.logger.info(f"Found window for feature {feature.name}: ({left}, {right})")


def write_outputs(args, features, predictions):
    """Write outputs to results file"""
    args.logger.info("Begin writing outputs")
    # Write features
    features_filename = constants.OUTPUT_FEATURES_FILENAME.format(args.output_dir, args.worker_idx)
    with open(features_filename, "wb") as features_file:
        cloudpickle.dump(features, features_file, protocol=pickle.DEFAULT_PROTOCOL)
    # TODO: Decide if all these are still necessary (only features and predictions used by callers)
    results_filename = constants.RESULTS_FILENAME.format(args.output_dir, args.worker_idx)
    root = h5py.File(results_filename, "w")

    def store_data(group, data):
        """Helper function to store data"""
        for feature_id, feature_data in data.items():
            group.create_dataset(feature_id, data=feature_data)

    store_data(root.create_group(constants.PREDICTIONS), predictions)
    root.close()
    args.logger.info("End writing outputs")


if __name__ == "__main__":
    main()
