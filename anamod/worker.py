"""
anamod worker pipeline
Given test data samples, a trained model and a set of feature groups, perturbs the features and
computes the effect on the model's output loss
"""

import argparse
from collections import namedtuple
import pickle

import cloudpickle
import h5py
import numpy as np

from anamod import constants
from anamod.compute_p_values import compute_p_value
from anamod.perturbations import PERTURBATION_FUNCTIONS, PERTURBATION_MECHANISMS
from anamod.utils import get_logger, round_value

Inputs = namedtuple("Inputs", ["data", "targets", "model"])


def main():
    """Main"""
    parser = argparse.ArgumentParser()
    parser.add_argument("args_filename", help="pickle file containing arguments"
                        " passed by master.py")
    cargs = parser.parse_args()
    with open(cargs.args_filename, "rb") as args_file:
        args = pickle.load(args_file)
    args.logger = get_logger(__name__, "%s/worker_%d.log" % (args.output_dir, args.task_idx))
    pipeline(args)


def pipeline(args):
    """Worker pipeline"""
    args.logger.info("Begin anamod worker pipeline")
    # Load features to perturb from file
    features = load_features(args.features_filename)
    # Load data
    data, targets = load_data(args.data_filename)
    # Load model
    model = load_model(args)
    inputs = Inputs(data, targets, model)
    # Baseline predictions/losses
    baseline_loss = compute_baseline(inputs)
    # Perturb features
    predictions, losses = perturb_features(args, inputs, features)
    compute_importances(features, losses, baseline_loss)
    # For important features, proceed with further analysis (temporal model analysis):
    if args.analysis_type == constants.TEMPORAL:
        temporal_analysis(args, inputs, features, baseline_loss)
    # Write outputs
    write_outputs(args, features, targets, predictions, losses)
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
    with open(args.model_filename, "rb") as model_file:
        model = cloudpickle.load(model_file)
    args.logger.info("End loading model")
    return model


def compute_baseline(inputs):
    """Compute baseline prediction/loss"""
    data, targets, model = inputs
    pred = model.predict(data)
    baseline_loss = model.loss(targets, pred)
    return baseline_loss


def get_perturbation_mechanism(args, perturbation_type=constants.ACROSS_INSTANCES):
    """Get appropriately configured object to perform perturbations"""
    perturbation_fn_class = PERTURBATION_FUNCTIONS[args.perturbation]
    perturbation_mechanism_class = PERTURBATION_MECHANISMS[args.analysis_type]
    return perturbation_mechanism_class(perturbation_fn_class, perturbation_type)


def perturb_features(args, inputs, features):
    """Perturb features"""
    # TODO: Perturbation modules should be provided as input so custom modules may be used
    args.logger.info("Begin perturbing features")
    predictions = {feature.name: np.zeros(len(inputs.targets)) for feature in features}
    losses = {feature.name: np.zeros(len(inputs.targets)) for feature in features}
    perturbation_mechanism = get_perturbation_mechanism(args)
    # Perturb each feature
    for feature in features:
        prediction, loss = perturb_feature(args, inputs, feature, perturbation_mechanism)
        predictions[feature.name] = prediction
        losses[feature.name] = loss
    args.logger.info("End perturbing features")
    return predictions, losses


def perturb_feature(args, inputs, feature, perturbation_mechanism, timesteps=...):
    """Perturb feature"""
    data, targets, model = inputs
    if args.perturbation == constants.SHUFFLING:
        prediction = np.zeros(len(targets))
        loss = np.zeros(len(targets))
        for _ in range(args.num_shuffling_trials):
            data_perturbed = perturbation_mechanism.perturb(data, feature, timesteps=timesteps)
            pred = model.predict(data_perturbed)
            prediction += pred
            loss += model.loss(targets, pred)
        prediction /= args.num_shuffling_trials
        loss /= args.num_shuffling_trials
    else:
        data_perturbed = perturbation_mechanism.perturb(data, feature, timesteps=timesteps)
        prediction = model.predict(data_perturbed)
        loss = model.loss(targets, prediction)
    return prediction, loss


def search_window(args, inputs, feature, perturbation_mechanism, baseline_loss):
    """Search temporal window of importance for given feature"""
    args.logger.info("Begin searching for temporal window for feature %s" % feature.name)
    # TODO: verify this will work if window spans the whole sequence
    T = inputs.data.shape[2]  # pylint: disable = invalid-name
    # Search left edge
    lbound, current, rbound = (0, T // 2, T)
    while current < rbound:
        # Search for the largest 'negative' window anchored on the left that results in a non-important p-value
        # The right edge of the resulting window is the left edge of the window of interest
        _, loss = perturb_feature(args, inputs, feature, perturbation_mechanism, range(0, current))
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
        _, loss = perturb_feature(args, inputs, feature, perturbation_mechanism, range(current, T))
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


def temporal_analysis(args, inputs, features, baseline_loss):
    """Perform temporal analysis of important features"""
    # FIXME: hardcoded p-value threshold
    features = list(filter(lambda feature: feature.pvalue_loss < constants.PVALUE_THRESHOLD, features))  # select (overall) important features
    args.logger.info("Identified important features: %s; proceeding with temporal analysis" % ",".join([feature.name for feature in features]))
    perturbation_mechanism_within = get_perturbation_mechanism(args, perturbation_type=constants.WITHIN_INSTANCE)
    perturbation_mechanism_across = get_perturbation_mechanism(args, perturbation_type=constants.ACROSS_INSTANCES)
    for feature in features:
        _, loss = perturb_feature(args, inputs, feature, perturbation_mechanism_within)
        feature.temporally_important = compute_p_value(baseline_loss, loss) < constants.PVALUE_THRESHOLD
        if feature.temporally_important:
            args.logger.info("Feature %s identified as temporally important" % feature.name)
            # TODO: figure out why within-instance perturbations to search window fail so haphazardly
            left, right = search_window(args, inputs, feature, perturbation_mechanism_across, baseline_loss)
            feature.temporal_window = (left, right)
            args.logger.info("Found window for feature %s: (%d, %d)" % (feature.name, left, right))
        else:
            args.logger.info("Feature %s identified as temporally unimportant" % feature.name)


def write_outputs(args, features, targets, predictions, losses):
    """Write outputs to results file"""
    args.logger.info("Begin writing outputs")
    # Write features
    features_filename = "%s/features_worker_%d.cpkl" % (args.output_dir, args.task_idx)
    with open(features_filename, "wb") as features_file:
        cloudpickle.dump(features, features_file)
    # TODO: Decide if all these are still necessary (only features and predictions used by callers)
    results_filename = "%s/results_worker_%d.hdf5" % (args.output_dir, args.task_idx)
    root = h5py.File(results_filename, "w")

    def store_data(group, data):
        """Helper function to store data"""
        for feature_id, feature_data in data.items():
            group.create_dataset(feature_id, data=feature_data)

    store_data(root.create_group(constants.PREDICTIONS), predictions)
    store_data(root.create_group(constants.LOSSES), losses)
    if args.task_idx == 0:
        root.create_dataset(constants.TARGETS, data=targets)  # TODO: remove
    root.close()
    args.logger.info("End writing outputs")


if __name__ == "__main__":
    main()
