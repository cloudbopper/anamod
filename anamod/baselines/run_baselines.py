"""Run baselines"""

import argparse
import configparser
from distutils.util import strtobool
import os
import subprocess

import numpy as np
import pandas as pd
from synmod.constants import CLASSIFIER, REGRESSOR

from anamod.baselines.explain_model import EXPLAINERS, TRUE_SCORES_FILENAME, EXPLAINER_SCORES_FILENAME, EXPLAINER_RUNTIME_FILENAME
from anamod.core.constants import DEFAULT
from anamod.core.utils import CondorJobWrapper, get_logger
from anamod.simulation.evaluation import get_precision_recall

SYNTHETIC = "SYNTHETIC"
LR = "LR"
RF = "RF"
NN = "NN"
CONFIG_BASELINES = "config.ini"


def main():
    """Parse arguments and run baselines"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-num_models", type=int, default=2, help="number of models to explain and average results over")
    parser.add_argument("-start_seed", type=int, default=100000, help="randomization seed for first model, incremented for"
                        " every subsequent model.")
    parser.add_argument("-condor", type=strtobool, default=True, help="Flag to enable condor for explaining models in parallel"
                        " (assumes shared filesystem")
    parser.add_argument("-config", type=str, default=DEFAULT, help="Section in config file"
                        " to use for simulation parameters")
    parser.add_argument("-model", default=SYNTHETIC, choices=[SYNTHETIC, LR, RF, NN],
                        help="Model to explain")
    parser.add_argument("-model_type", default=CLASSIFIER, choices=[CLASSIFIER, REGRESSOR],
                        help="Type of model to explain")
    parser.add_argument("-missing_models_only", type=strtobool, default=False,
                        help="Only run explainer for models where results are missing (due to failure/abort)")
    parser.add_argument("-evaluate_only", type=strtobool, default=False,
                        help="evaluate metrics assuming results are already generated")
    parser.add_argument("-anamod_scores_dir",
                        help="If provided, evaluate explainer w.r.t. top features returned by anamod")
    parser.add_argument("-explainer", choices=EXPLAINERS.keys(), required=True)
    parser.add_argument("-output_dir", required=True)
    args, pass_arglist = parser.parse_known_args()
    args.pass_args = " ".join(pass_arglist)

    args.output_dir = os.path.abspath(args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.logger = get_logger(__name__, f"{args.output_dir}/run_baselines.log")
    pipeline(args)


def pipeline(args):
    """Pipeline"""
    sconfig = load_config(args)
    if not args.evaluate_only:
        explain(args, sconfig)
    # Write metrics and explanations to file
    evaluate(args)


def load_config(args):
    """Load simulation configuration"""
    config_filename = f"{os.path.dirname(__file__)}/{CONFIG_BASELINES}"
    assert os.path.exists(config_filename)
    config = configparser.ConfigParser()
    config.read(config_filename)
    dconfig = config[args.config]
    assert "model_type" not in dconfig, "model_type specified on command-line"
    sconfig = ""
    for option, value in dconfig.items():
        sconfig += f"-{option} {value} "
    return sconfig


def explain(args, sconfig):
    """Synthesize data/model and explain"""
    jobs = []
    for ridx in range(args.num_models):
        job_dir = f"{args.output_dir}/{ridx}"
        if args.missing_models_only and os.path.exists(f"{job_dir}/{EXPLAINER_SCORES_FILENAME}"):
            continue
        seed = args.start_seed + ridx
        cmd = (f"python -m anamod.baselines.explain_model -explainer {args.explainer} -model {args.model} -model_type {args.model_type}"
               f" -output_dir {job_dir} -seed {seed} {sconfig} {args.pass_args}")
        args.logger.info(f"Running cmd (condor: {args.condor}): {cmd}")
        if args.condor:
            job = CondorJobWrapper(cmd, [], job_dir, shared_filesystem=True, avoid_bad_hosts=True, retry_arbitrary_failures=True)
            job.run()
            jobs.append(job)
        else:
            subprocess.run(cmd, shell=True, check=True)
    if args.condor:
        CondorJobWrapper.monitor(jobs)


METRICS = ["top_k_relevant_features_power", "top_k_relevant_features_fdr",
           "top_k_anamod_features_power", "top_k_anamod_features_fdr",
           "top_k_relevant_timesteps_power", "top_k_relevant_timesteps_fdr",
           "top_k_anamod_timesteps_power", "top_k_anamod_timesteps_fdr",
           "cpu_runtime", "user_runtime"]


def evaluate(args):
    """Evaluate explainer"""
    args.logger.info("Begin explainer evaluation")
    metrics = pd.DataFrame(data=np.zeros((args.num_models, len(METRICS))), columns=METRICS)
    for ridx in range(args.num_models):
        rdir = f"{args.output_dir}/{ridx}"
        runtime = np.load(f"{rdir}/{EXPLAINER_RUNTIME_FILENAME}")
        true_scores = np.load(f"{rdir}/{TRUE_SCORES_FILENAME}")
        explainer_scores = np.abs(np.load(f"{rdir}/{EXPLAINER_SCORES_FILENAME}"))  # absolute values for comparison since some scores may be negative
        assert explainer_scores.shape == true_scores.shape
        metrics.loc[ridx, ['top_k_relevant_features_power', 'top_k_relevant_features_fdr']] = evaluate_features(true_scores, explainer_scores)
        metrics.loc[ridx, ['top_k_relevant_timesteps_power', 'top_k_relevant_timesteps_fdr']] = evaluate_timesteps(true_scores, explainer_scores)
        if args.anamod_scores_dir:
            anamod_scores = np.load(f"{args.anamod_scores_dir}/{ridx}/{EXPLAINER_SCORES_FILENAME}")
            assert anamod_scores.shape == true_scores.shape
            metrics.loc[ridx, ['top_k_anamod_features_power', 'top_k_anamod_features_fdr']] = evaluate_features(true_scores, explainer_scores,
                                                                                                                    anamod_scores)
            metrics.loc[ridx, ['top_k_anamod_timesteps_power', 'top_k_anamod_timesteps_fdr']] = evaluate_timesteps(true_scores, explainer_scores,
                                                                                                                       anamod_scores)
        metrics.loc[ridx, ['cpu_runtime', 'user_runtime']] = runtime

    # TODO: visualize box/violin plot to examine high FDR cases
    print(f"Explainer: {args.explainer}")
    means = metrics.mean(axis=0)
    medians = metrics.median(axis=0)
    ametrics = {"Average power (top k features, k = min(num_relevant_features, num_important_features))": means['top_k_relevant_features_power'],
                "Average FDR (top k features, k = min(num_relevant_features, num_important_features))": means['top_k_relevant_features_fdr'],
                "Average power (top k features, k = num_important_features_anamod)": means['top_k_anamod_features_power'],
                "Average FDR (top k features, k = num_important_features_anamod)": means['top_k_anamod_features_fdr'],
                "Average power (top k timesteps, k = min(num_relevant_timesteps, num_important_timesteps))": means['top_k_relevant_timesteps_power'],
                "Average FDR (top k timesteps, k = min(num_relevant_timesteps, num_important_timesteps))": means['top_k_relevant_timesteps_fdr'],
                "Average power (top k timesteps, k = num_important_timesteps_anamod)": means['top_k_anamod_timesteps_power'],
                "Average FDR (top k timesteps, k = num_important_timesteps_anamod)": means['top_k_anamod_timesteps_fdr'],
                "Median CPU runtime (seconds)": medians['cpu_runtime'],
                "Median user runtime (seconds)": medians['user_runtime']}
    for key, value in ametrics.items():
        print(f"{key}: {value}")
    print(f"All metrics (CSV):\n{pd.DataFrame.from_dict([ametrics]).to_csv(index=False)}")
    args.logger.info("End explainer evaluation")


def evaluate_features(true_scores, explainer_scores, anamod_scores=None):
    """Evaluate features identified by explainer"""
    # Get power/FDR for relevant features
    num_features, _ = true_scores.shape
    true_features = np.any(true_scores, axis=1)
    explainer_num_timesteps_per_feature = np.sum(explainer_scores != 0, axis=1)  # pylint: disable = invalid-name
    explainer_scores_per_feature = np.zeros(num_features)
    for idx in range(num_features):
        if explainer_num_timesteps_per_feature[idx]:
            explainer_scores_per_feature[idx] = np.sum(explainer_scores[idx]) / explainer_num_timesteps_per_feature[idx]
    num_relevant_features = min(np.sum(true_features), np.sum(explainer_scores_per_feature != 0))
    if anamod_scores is not None:
        anamod_features = np.any(anamod_scores != 0, axis=1)
        num_relevant_features = min(num_relevant_features, np.sum(anamod_features))
    explainer_features = np.zeros(num_features)
    explainer_features[np.argsort(explainer_scores_per_feature)[::-1][:num_relevant_features]] = 1
    precision, recall = get_precision_recall(true_features, explainer_features)
    return recall, 1 - precision


def evaluate_timesteps(true_scores, explainer_scores, anamod_scores=None):
    """Evaluate timesteps identified by explainer"""
    # Identify relevant timesteps
    num_features, num_timesteps = true_scores.shape
    true_scores_tabular_bool = true_scores.flatten() != 0
    # Get power/FDR for relevant timesteps
    explainer_scores_tabular = explainer_scores.flatten()
    num_relevant_timesteps = min(sum(true_scores_tabular_bool), sum(explainer_scores_tabular != 0))
    if anamod_scores is not None:
        anamod_num_timesteps = np.sum(anamod_scores != 0)
        num_relevant_timesteps = min(num_relevant_timesteps, anamod_num_timesteps)
    explainer_scores_tabular_bool = np.zeros(num_features * num_timesteps)
    explainer_scores_tabular_bool[np.argsort(explainer_scores_tabular)[::-1][:num_relevant_timesteps]] = 1
    precision, recall = get_precision_recall(true_scores_tabular_bool, explainer_scores_tabular_bool)
    return recall, 1 - precision


if __name__ == "__main__":
    main()
