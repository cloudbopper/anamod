"""
anamod worker pipeline
Given test data samples, a trained model and a set of feature groups, perturbs the features and
computes the effect on the model's output loss
"""

import argparse
import pickle

import cloudpickle
import h5py
import numpy as np

from anamod import constants
from anamod.compute_p_values import compute_p_value
from anamod.perturbations import PerturbMatrix, PerturbTensor, Zeroing, Shuffling
from anamod.utils import get_logger, round_value, round_vectordict

PERTURBATION_FUNCTIONS = {constants.ZEROING: Zeroing, constants.SHUFFLING: Shuffling}
PERTURBATION_MECHANISMS = {constants.HIERARCHICAL: PerturbMatrix, constants.TEMPORAL: PerturbTensor}


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
    data_root = load_data(args.data_filename)
    # Load model
    model = load_model(args)
    # Baseline predictions/losses
    baseline_loss = compute_baseline(data_root, model)
    # Perturb features
    targets, predictions, losses = perturb_features(args, features, data_root, model)
    compute_p_values(features, predictions, losses, baseline_loss)
    # TODO: For important features, proceed with further analysis (temporal model analysis)
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
    return data_root


def load_model(args):
    """Load model object from model-generating python file"""
    args.logger.info("Begin loading model")
    with open(args.model_filename, "rb") as model_file:
        model = cloudpickle.load(model_file)
    args.logger.info("End loading model")
    return model


def compute_baseline(data_root, model):
    """Compute baseline prediction/loss"""
    data = data_root[constants.DATA][...]
    y_true = data_root[constants.TARGETS]
    y_pred = model.predict(data)
    baseline_loss = model.loss(y_true, y_pred)
    return round_value(baseline_loss, 4)


def perturb_features(args, features, data_root, model, perturbation_type=constants.ACROSS_INSTANCES):
    """Perturb features"""
    # TODO: Perturbation modules should be provided as input so custom modules may be used
    # pylint: disable = invalid-name, too-many-locals
    args.logger.info("Begin perturbing features")
    X = data_root[constants.DATA]
    y_true = data_root[constants.TARGETS]
    # Select perturbation
    perturbation_fn_class = PERTURBATION_FUNCTIONS[args.perturbation]
    perturbation_mechanism_class = PERTURBATION_MECHANISMS[args.analysis_type]
    perturbation_mechanism = perturbation_mechanism_class(perturbation_fn_class, perturbation_type)
    predictions = {feature.name: np.zeros(len(y_true)) for feature in features}
    losses = {feature.name: np.zeros(len(y_true)) for feature in features}
    # Perturb each feature
    for feature in features:
        # FIXME: Having perturbation logic here (due to shuffles) hinders modularity/extensibility
        if args.perturbation == constants.SHUFFLING:
            for _ in range(args.num_shuffling_trials):
                X_perturbed = perturbation_mechanism.perturb(X, feature)
                y_pred = model.predict(X_perturbed)
                predictions[feature.name] += y_pred
                losses[feature.name] += model.loss(y_true, y_pred)
            predictions[feature.name] /= args.num_shuffling_trials
            losses[feature.name] /= args.num_shuffling_trials
        else:
            X_perturbed = perturbation_mechanism.perturb(X, feature)
            predictions[feature.name] = model.predict(X_perturbed)
            losses[feature.name] = model.loss(y_true, predictions[feature.name])
    args.logger.info("End perturbing features")
    return y_true, predictions, losses


def compute_p_values(features, predictions, losses, baseline_loss):
    """Computes p-values indicating feature importances"""
    losses = round_vectordict(losses)
    predictions = round_vectordict(predictions)
    mean_baseline_loss = np.mean(baseline_loss)
    for feature in features:
        loss = losses[feature.name]
        mean_loss = np.mean(loss)
        pvalue_loss = compute_p_value(baseline_loss, loss)
        effect_size = mean_loss - mean_baseline_loss
        feature.effect_size = round_value(effect_size)
        feature.mean_loss = round_value(mean_loss)
        feature.pvalue_loss = round_value(pvalue_loss)


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
        root.create_dataset(constants.TARGETS, data=targets)
    root.close()
    args.logger.info("End writing outputs")


if __name__ == "__main__":
    main()
