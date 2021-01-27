"""Synthesize data and run explainer"""
import argparse
import subprocess
import time
import timeit

import numpy as np
from synmod.constants import REGRESSOR
from synmod.aggregators import Slope

from anamod.baselines.explainers import AnamodExplainer, LimeExplainer, SageExplainer
from anamod.simulation.simulation import read_synthesized_inputs, read_intermediate_inputs

EXPLAINERS = {"anamod": AnamodExplainer, "lime": LimeExplainer, "sage": SageExplainer}
TRUE_SCORES_FILENAME = "true_scores.npy"
EXPLAINER_SCORES_FILENAME = "explainer_scores.npy"
EXPLAINER_RUNTIME_FILENAME = "explainer_runtime.npy"


def main():
    """Main"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-explainer", required=True)
    parser.add_argument("-model", required=True)
    parser.add_argument("-model_type", required=True)
    parser.add_argument("-output_dir", required=True)
    parser.add_argument("-seed", type=int, required=True)
    args, pass_arglist = parser.parse_known_args()
    pass_args = " ".join(pass_arglist)
    synthesized_features, data, model, targets = synthesize(args, pass_args)
    true_scores = get_true_scores(synthesized_features, data)
    explainer_scores, elapsed_time = explain(args, synthesized_features, data, model, targets)
    write_outputs(args, true_scores, explainer_scores, elapsed_time)


def synthesize(args, pass_args):
    """Synthesize data"""
    sim_cmd = (f"python -m anamod.simulation.simulation {pass_args} -output_dir {args.output_dir} -seed {args.seed}"
               f" -synthesize_only 1 -synthesis_dir {args.output_dir} -model_type {args.model_type}")
    subprocess.run(sim_cmd, check=True, shell=True)
    synthesized_features, data, _ = read_synthesized_inputs(args.output_dir)
    model, targets = read_intermediate_inputs(args.output_dir)
    return synthesized_features, data, model, targets


def get_true_scores(synthesized_features, data):
    """Get ground truth feature importance scores"""
    _, num_features, num_timesteps = data.shape
    true_scores = np.zeros((num_features, num_timesteps))
    for fidx, feature in enumerate(synthesized_features):
        left, right = feature.window
        # FIXME: change simulation to ensure all timesteps in window are relevant
        # Disable the conditional to show entire windows instead of just relevant timesteps
        if isinstance(feature.aggregation_fn, Slope):
            true_scores[fidx][[left, right]] = feature.effect_size
        else:
            true_scores[fidx][left: right + 1] = feature.effect_size
    return true_scores


def explain(args, synthesized_features, data, model, targets):
    """Pipeline"""
    start_time = get_time()
    # Configure explainer
    mode = "regression" if args.model_type == REGRESSOR else "classification"  # For LIME
    loss_fn = "mse" if args.model_type == REGRESSOR else "cross entropy"  # FOR SAGE
    # Tabular feature names
    num_timesteps = data.shape[2]
    feature_names_tabular = []
    for feature in synthesized_features:
        new_features = [f"{feature.name}_{tidx}" for tidx in range(num_timesteps)]
        feature_names_tabular.extend(new_features)
    kwargs = dict(model_type=args.model_type, mode=mode, loss_fn=loss_fn, targets=targets,
                  feature_names=feature_names_tabular, output_dir=args.output_dir)
    # Initialize explainer and explain model
    explainer_cls = EXPLAINERS[args.explainer]
    explainer = explainer_cls(model.predict, data, **kwargs)
    scores = explainer.explain()
    elapsed_time = get_time() - start_time
    return scores, elapsed_time


def write_outputs(args, true_scores, explainer_scores, elapsed_time):
    """Write outputs to file"""
    np.save(f"{args.output_dir}/{TRUE_SCORES_FILENAME}", true_scores)
    np.save(f"{args.output_dir}/{EXPLAINER_SCORES_FILENAME}", explainer_scores)
    np.save(f"{args.output_dir}/{EXPLAINER_RUNTIME_FILENAME}", elapsed_time)


def get_time():
    """Return CPU and wall clock time"""
    return np.array([time.process_time(), timeit.default_timer()])


if __name__ == "__main__":
    main()
