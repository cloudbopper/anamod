"""
anamod master pipeline
Given test samples, a trained model and a feature hierarchy,
computes the effect on the model's output loss after perturbing the features/feature groups
in the hierarchy.
"""

import csv
import importlib
import os
import sys
from unittest.mock import patch

import anytree
import numpy as np

from anamod import constants, utils
from anamod.fdr import hierarchical_fdr_control
from anamod.interactions import analyze_interactions
from anamod.pipelines import CondorPipeline, SerialPipeline


def main(args):
    """Parse arguments from command-line"""
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.rng = np.random.default_rng(args.seed)
    args.logger = utils.get_logger(__name__, "%s/anamod.log" % args.output_dir)
    validate_args(args)
    return pipeline(args)


def pipeline(args):
    """Master pipeline"""
    # FIXME: some outputs returned via return value (temporal analysis), other via output file (hierarchical analysis)
    args.logger.info("Begin anamod master pipeline with args: %s" % args)
    features = list(filter(lambda node: node.perturbable, anytree.PreOrderIter(args.feature_hierarchy)))  # flatten hierarchy
    # Perturb features
    analyzed_features, predictions = perturb_features(args, features)
    # TODO: Run these only for hierarchical feature importance analysis
    if args.analysis_type == constants.HIERARCHICAL:
        hierarchical_fdr(args, analyzed_features)
    # Analyze pairwise interactions
    if args.analyze_interactions:
        analyze_interactions(args, analyzed_features, predictions)
    args.logger.info("End anamod master pipeline")
    return reorder_features(analyzed_features, [feature.name for feature in features])  # Re-order analyzed features to match original order


def reorder_features(features, reordered_names):
    """Reorder features according to specified list of names"""
    reordered_features = [None] * len(features)
    name_to_feature_map = {feature.name: feature for feature in features}
    for idx, name in enumerate(reordered_names):
        reordered_features[idx] = name_to_feature_map[name]
    return reordered_features


def prepare_features(args, features):
    """Prepare features for perturbation by shuffling to balance load across workers"""
    reordered_names = [feature.name for feature in features]
    reordered_names.sort()  # For reproducibility across python versions
    args.rng.shuffle(reordered_names)  # To balance load across workers
    return reorder_features(features, reordered_names)


def perturb_features(args, features):
    """Perturb features, observe effect on model loss and aggregate results"""
    features = prepare_features(args, features)
    # Partition features, Launch workers, Aggregate results
    worker_pipeline = CondorPipeline(args, features) if args.condor else SerialPipeline(args, features)
    return worker_pipeline.run()


def hierarchical_fdr(args, features):
    """Performs hierarchical FDR control on results"""
    # Write FDR control input file
    input_filename = "%s/%s" % (args.output_dir, constants.PVALUES_FILENAME)
    with open(input_filename, "w", newline="") as outfile:
        writer = csv.writer(outfile, delimiter=",")
        writer.writerow([constants.NODE_NAME, constants.PARENT_NAME, constants.DESCRIPTION, constants.EFFECT_SIZE,
                         constants.MEAN_LOSS, constants.PVALUE_LOSSES])
        for node in features:
            name = node.name
            parent_name = node.parent.name if node.parent else ""
            writer.writerow([name, parent_name, node.description, node.effect_size, node.mean_loss, node.pvalue_loss])
    # Run FDR control
    output_dir = "%s/%s" % (args.output_dir, constants.HIERARCHICAL_FDR_DIR)
    cmd = ("python -m anamod.fdr.hierarchical_fdr_control -output_dir %s -procedure yekutieli "
           "-rectangle_leaves 1 %s" % (output_dir, input_filename))
    args.logger.info("Running cmd: %s" % cmd)
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        hierarchical_fdr_control.main()
    # TODO: update feature importance attributes based on results of hierarchical FDR control
    # Better yet, pass features directly to hierarchical FDR control and update


def validate_args(args):
    """Validate arguments"""
    if args.analyze_interactions and args.perturbation == constants.SHUFFLING:
        raise ValueError("Interaction analysis is not supported with shuffling perturbations")
    if args.condor:
        try:
            importlib.import_module("htcondor")
        except ModuleNotFoundError:
            raise ModuleNotFoundError("htcondor module not found. "
                                      "Use 'pip install htcondor' to install htcondor on a compatible platform, or "
                                      "disable condor")
