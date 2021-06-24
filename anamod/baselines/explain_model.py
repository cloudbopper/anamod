"""Synthesize data and explain model"""
import argparse
import os
import subprocess
import time
import timeit

import numpy as np
from synmod.constants import REGRESSOR

from anamod.baselines.explainers import TimeExplainer, PermutationTestExplainer, PermutationTestExplainerFDRControl
from anamod.baselines.explainers import LimeExplainer, SageExplainer, SageExplainerMeanImputer, SageExplainerZeroImputer
from anamod.baselines.explainers import OcclusionZeroExplainer, OcclusionUniformExplainer, CXPlainExplainer
from anamod.core.constants import QUADRATIC_LOSS, BINARY_CROSS_ENTROPY, SEED
from anamod.core.utils import get_logger
from anamod.simulation.simulation import read_synthesized_inputs, read_intermediate_inputs

EXPLAINERS = {"time": TimeExplainer, "perm": PermutationTestExplainer, "perm-fdr": PermutationTestExplainerFDRControl,
              "lime": LimeExplainer, "sage": SageExplainer, "sage-mean": SageExplainerMeanImputer, "sage-zero": SageExplainerZeroImputer,
              "occlusion-zero": OcclusionZeroExplainer, "occlusion-uniform": OcclusionUniformExplainer,
              "cxplain": CXPlainExplainer}
TRUE_SCORES_FILENAME = "true_scores.npy"
EXPLAINER_SCORES_FILENAME = "explainer_scores.npy"
EXPLAINER_RUNTIME_FILENAME = "explainer_runtime.npy"
EXPLAINER_EVALUATION_FILENAME = "explainer_evaluation.csv"


def main():
    """Main"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-explainer", required=True)
    parser.add_argument("-base_explainer_dir")
    parser.add_argument("-model", required=True)
    parser.add_argument("-model_type", required=True)
    parser.add_argument("-model_dir")
    parser.add_argument("-output_dir", required=True)
    parser.add_argument("-seed", type=int, required=True)
    args, pass_arglist = parser.parse_known_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.logger = get_logger(__name__, f"{args.output_dir}/explain_model.log")
    pass_args = " ".join(pass_arglist)
    synthesized_features, data, model, targets = synthesize(args, pass_args)
    true_scores = get_true_scores(synthesized_features, data)
    explainer_scores, elapsed_time = explain_model(args, synthesized_features, data, model, targets)
    write_outputs(args, true_scores, explainer_scores, elapsed_time)


def synthesize(args, pass_args):
    """Synthesize data"""
    model_dir = args.model_dir if args.model_dir else args.base_explainer_dir
    if model_dir is None:
        assert not args.explainer == "perm-fdr", "Explainer 'perm-fdr' requires prior execution of explainer 'perm'"
        model_dir = args.output_dir
        sim_cmd = (f"python -m anamod.simulation.simulation {pass_args} -output_dir {args.output_dir} -seed {args.seed}"
                   f" -synthesize_only 1 -synthesis_dir {args.output_dir} -model_type {args.model_type}")
        args.logger.info(f"Running synthesis cmd: {sim_cmd}")
        subprocess.run(sim_cmd, check=True, shell=True)
    synthesized_features, data, _ = read_synthesized_inputs(model_dir)
    model, targets = read_intermediate_inputs(model_dir)
    return synthesized_features, data, model, targets


def get_true_scores(synthesized_features, data):
    """Get ground truth feature importance scores"""
    _, num_features, num_timesteps = data.shape
    true_scores = np.zeros((num_features, num_timesteps))
    for fidx, feature in enumerate(synthesized_features):
        left, right = feature.window
        true_scores[fidx][left: right + 1] = feature.effect_size
    return true_scores


def explain_model(args, synthesized_features, data, model, targets):
    """Pipeline"""
    args.logger.info("Begin explaining model")
    start_time = get_time()
    # Configure explainer
    mode = "regression" if args.model_type == REGRESSOR else "classification"  # For LIME
    loss_fn = "mse" if args.model_type == REGRESSOR else "cross entropy"  # For SAGE
    loss_function = QUADRATIC_LOSS if args.model_type == REGRESSOR else BINARY_CROSS_ENTROPY  # For anamod
    # Tabular feature names
    num_timesteps = data.shape[2]
    feature_names_tabular = []
    for feature in synthesized_features:
        new_features = [f"{feature.name}_{tidx}" for tidx in range(num_timesteps)]
        feature_names_tabular.extend(new_features)
    if args.explainer == "perm-fdr":
        assert args.base_explainer_dir, "Explainer 'perm-fdr' requires prior execution of explainer 'perm'"
    kwargs = dict(model_type=args.model_type, mode=mode, loss_fn=loss_fn, targets=targets,
                  feature_names=feature_names_tabular, loss_function=loss_function,
                  base_explainer_dir=args.base_explainer_dir, output_dir=args.output_dir,
                  num_samples=10, rng_seed=SEED)
    # Initialize explainer and explain model
    explainer_cls = EXPLAINERS[args.explainer]
    explainer = explainer_cls(model.predict, data, **kwargs)
    scores = explainer.explain()
    elapsed_time = get_time() - start_time
    args.logger.info("End explaining model")
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
