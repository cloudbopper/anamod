"""
anamod master pipeline
Given test samples, a trained model and a feature hierarchy,
computes the effect on the model's output loss after perturbing the features/feature groups
in the hierarchy.
"""

import argparse
import csv
from distutils.util import strtobool
import os
import sys
from unittest.mock import patch

import anytree
import h5py
import numpy as np

from anamod import constants, utils
from anamod.fdr import hierarchical_fdr_control
from anamod.feature import Feature
from anamod.interactions import analyze_interactions
from anamod.pipelines import CondorPipeline, SerialPipeline


def main():
    """Parse arguments"""
    parser = argparse.ArgumentParser("python anamod")
    # Required arguments
    required = parser.add_argument_group("Required parameters")
    required.add_argument("-output_dir", help="Output directory", required=True)
    required.add_argument("-data_filename", help="Test data in HDF5 format", required=True)
    required.add_argument("-model_filename", help="File containing model, pickled using cloudpickle", required=True)
    # Optional common arguments
    common = parser.add_argument_group("Common optional parameters")
    common.add_argument("-analysis_type", help="Type of model analysis to perform",
                        default=constants.HIERARCHICAL, choices=[constants.TEMPORAL, constants.HIERARCHICAL])
    common.add_argument("-perturbation", default=constants.SHUFFLING, choices=[constants.ZEROING, constants.SHUFFLING],
                        help="type of perturbation to perform (default %s)" % constants.SHUFFLING)
    common.add_argument("-num_shuffling_trials", type=int, default=50, help="Number of shuffling trials to average over, "
                        "when shuffling perturbations are selected")
    common.add_argument("-compile_results_only", help="only compile results (assuming they already exist), "
                        "skipping actually launching jobs", type=strtobool, default=False)
    # Hierarchical feature importance analysis arguments
    hierarchical = parser.add_argument_group("Hierarchical feature analysis arguments")
    hierarchical.add_argument("-hierarchy_filename", help="Feature hierarchy in CSV format", default="")
    hierarchical.add_argument("-analyze_interactions", help="flag to enable testing of interaction significance. By default,"
                              " only pairwise interactions between leaf features identified as important by hierarchical FDR"
                              " are tested. To enable testing of all pairwise interactions, also use -analyze_all_pairwise_interactions",
                              type=strtobool, default=False)
    hierarchical.add_argument("-analyze_all_pairwise_interactions", help="analyze all pairwise interactions between leaf features,"
                              " instead of just pairwise interactions of leaf features identified by hierarchical FDR",
                              type=strtobool, default=False)
    # Condor arguments
    condor = parser.add_argument_group("Condor parameters")
    condor.add_argument("-condor", help="Use condor for parallelization", type=strtobool, default=False)
    condor.add_argument("-condor_cleanup", type=strtobool, default=True, help="remove intermediate condor files"
                        " after completion (typically for debugging). Enabled by default to remove"
                        " space and clutter, and to avoid condor file issues")
    condor.add_argument("-features_per_worker", type=int, default=10, help="worker load")
    condor.add_argument("-eviction_timeout", type=int, default=14400, help="time in seconds to allow condor jobs"
                        " to run before evicting and restarting them on another condor node")
    condor.add_argument("-idle_timeout", type=int, default=3600, help="time in seconds to allow condor jobs"
                        " to stay idle before removing them from condor and attempting them on the master node.")
    condor.add_argument("-memory_requirement", type=int, default=16, help="memory requirement in GB, minimum 1, default 16")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.rng = np.random.default_rng(constants.SEED)
    args.logger = utils.get_logger(__name__, "%s/anamod.log" % args.output_dir)
    validate_args(args)
    return pipeline(args)


def pipeline(args):
    """Master pipeline"""
    args.logger.info("Begin anamod master pipeline with args: %s" % args)
    if args.analysis_type == constants.HIERARCHICAL:
        # Load hierarchy from file
        hierarchy_root = load_hierarchy(args.hierarchy_filename)
        # Prepare features
        features = list(anytree.PreOrderIter(hierarchy_root))  # flatten hierarchy
    else:
        # TODO: get number of features more efficiently
        data_root = h5py.File(args.data_filename, "r")
        data = data_root[constants.DATA]
        num_features = data.shape[1]  # data is instances X features X timesteps
        features = [Feature(str(idx), idx=[idx]) for idx in range(num_features)]
    # Prepare features for perturbation
    prepare_features(args, features)
    # Perturb features
    features, _, _, predictions = perturb_features(args, features)
    # TODO: Run these only for hierarchical feature importance analysis
    if args.analysis_type == constants.HIERARCHICAL:
        hierarchical_fdr(args, features)
    # Analyze pairwise interactions
    if args.analyze_interactions:
        analyze_interactions(args, features, predictions)
    args.logger.info("End anamod master pipeline")
    return features  # FIXME: some outputs returned via return value (temporal analysis), other via output file (hierarchical analysis)


def load_hierarchy(hierarchy_filename):
    """
    Load hierarchy from CSV.

    Args:
        hierarchy_filename: CSV specifying hierarchy in required format (see anamod/spec.md)

    Returns:
        anytree node representing root of hierarchy

    """
    root = None
    nodes = {}
    # Construct nodes
    with open(hierarchy_filename) as hierarchy_file:
        reader = csv.DictReader(hierarchy_file)
        for row in reader:
            node = Feature(row[constants.NODE_NAME],
                           parent_name=row[constants.PARENT_NAME], description=row[constants.DESCRIPTION],
                           idx=Feature.unpack_indices(row[constants.INDICES]))
            assert node.name not in nodes, "Node name must be unique across all features: %s" % node.name
            nodes[node.name] = node
    # Construct tree
    for node in nodes.values():
        if not node.parent_name:
            assert not root, "Invalid tree structure: %s and %s both have no parent" % (root.node_name, node.node_name)
            root = node
        else:
            assert node.parent_name in nodes, "Invalid tree structure: no parent named %s" % node.parent_name
            node.parent = nodes[node.parent_name]
    assert root, "Invalid tree structure: root node missing (every node has a parent)"
    # Checks
    all_idx = set()
    for node in anytree.PostOrderIter(root):
        if node.is_leaf:
            assert node.idx, "Leaf node %s must have at least one index" % node.name
            assert not all_idx.intersection(node.idx), "Leaf node %s has index overlap with other leaf nodes" % node.name
            all_idx.update(node.idx)
        else:
            # Ensure non-leaf nodes have empty initial indices
            assert not node.idx, "Non-leaf node %s has non-empty initial indices" % node.name
    # Populate data structures
    for node in anytree.PostOrderIter(root):
        for child in node.children:
            node.idx += child.idx
    return root


def prepare_features(args, features):
    """Prepare features for perturbation"""
    features.sort(key=lambda node: node.name)  # For reproducibility across python versions
    args.rng.shuffle(features)  # To balance load across workers


def perturb_features(args, feature_nodes):
    """
    Perturb features, observe effect on model loss and aggregate results

    Args:
        args: Command-line arguments
        feature_nodes: flattened feature hierarchy comprising nodes for base features/feature groups

    Returns:
        Aggregated results from workers
    """
    # Partition features, Launch workers, Aggregate results
    worker_pipeline = SerialPipeline(args, feature_nodes)
    if args.condor:
        worker_pipeline = CondorPipeline(args, feature_nodes)
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
    if args.analysis_type == constants.HIERARCHICAL:
        assert args.hierarchy_filename, "Hierarchy filename required for hierarchical feature importance analysis"


if __name__ == "__main__":
    main()
