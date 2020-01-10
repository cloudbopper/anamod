"""
anamod master pipeline
Given test samples, a trained model and a feature hierarchy,
computes the effect on the model's output loss after perturbing the features/feature groups
in the hierarchy.
"""

import argparse
import csv
import os
import sys
from unittest.mock import patch

import anytree
import numpy as np

from anamod.compute_p_values import compute_p_value
from anamod import constants, utils
from anamod.fdr import hierarchical_fdr_control
from anamod.feature import Feature
from anamod.interactions import analyze_interactions
from anamod.pipelines import CondorPipeline, SerialPipeline, round_vectordict


def main():
    """Parse arguments"""
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument("-model_generator_filename", help="python script that generates model "
                        "object for subsequent callbacks to model.predict", required=True)
    parser.add_argument("-hierarchy_filename", help="Feature hierarchy in CSV format", required=True)
    parser.add_argument("-data_filename", help="Test data in HDF5 format", required=True)
    parser.add_argument("-output_dir", help="Output directory", required=True)
    # Optional arguments
    parser.add_argument("-perturbation", default=constants.ZEROING, choices=[constants.ZEROING, constants.SHUFFLING],
                        help="type of perturbation to perform:\n"
                        "%s (default): works on both static and temporal data\n"
                        "%s: works only on static data" % (constants.ZEROING, constants.SHUFFLING))
    parser.add_argument("-num_shuffling_trials", type=int, default=500, help="Number of shuffling trials to average over, "
                        "when shuffling perturbations are selected")
    parser.add_argument("-condor", dest="condor", action="store_true",
                        help="Enable parallelization using condor (default disabled)")
    parser.add_argument("-no-condor", dest="condor", action="store_false", help="Disable parallelization using condor")
    parser.set_defaults(condor=False)
    parser.add_argument("-features_per_worker", type=int, default=10, help="worker load")
    parser.add_argument("-eviction_timeout", type=int, default=14400, help="time in seconds to allow condor jobs"
                        " to run before evicting and restarting them on another condor node")
    parser.add_argument("-idle_timeout", type=int, default=3600, help="time in seconds to allow condor jobs"
                        " to stay idle before removing them from condor and attempting them on the master node.")
    parser.add_argument("-memory_requirement", type=int, default=16, help="memory requirement in GB, minimum 1, default 16")
    parser.add_argument("-compile_results_only", help="only compile results (assuming they already exist), "
                        "skipping actually launching jobs", action="store_true")
    parser.add_argument("-model_type", default=constants.REGRESSION,
                        help="Model type (default: regression). Note: this variable is currently unused.",
                        choices=[constants.BINARY_CLASSIFIER, constants.CLASSIFIER, constants.REGRESSION],)
    parser.add_argument("-analyze_interactions", help="flag to enable testing of interaction significance. By default,"
                        " only pairwise interactions between leaf features identified as important by hierarchical FDR"
                        " are tested. To enable testing of all pairwise interactions, also use -analyze_all_pairwise_interactions",
                        action="store_true")
    parser.add_argument("-analyze_all_pairwise_interactions", help="analyze all pairwise interactions between leaf features,"
                        " instead of just pairwise interactions of leaf features identified by hierarchical FDR",
                        action="store_true")
    parser.add_argument("-no-condor-cleanup", action="store_false", help="disable removal of intermediate condor files"
                        " after completion (typically for debugging). By default these files will be cleared to remove"
                        " space and clutter, and to avoid condor file issues", dest="cleanup")
    parser.set_defaults(cleanup=True)

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.rng = np.random.default_rng(constants.SEED)
    logger = utils.get_logger(__name__, "%s/master.log" % args.output_dir)
    validate_args(args)
    pipeline(args, logger)


def pipeline(args, logger):
    """Master pipeline"""
    logger.info("Begin anamod master pipeline with args: %s" % args)
    # Load hierarchy from file
    hierarchy_root = load_hierarchy(args.hierarchy_filename)
    # Flatten hierarchy to allow partitioning across workers
    feature_nodes = flatten_hierarchy(args, hierarchy_root)
    # Perturb features
    _, losses, predictions = perturb_features(args, logger, feature_nodes)
    # Compute p-values
    compute_p_values(args, hierarchy_root, losses, predictions)
    # Run hierarchical FDR
    hierarchical_fdr(args, logger)
    # Analyze pairwise interactions
    if args.analyze_interactions:
        analyze_interactions(args, logger, feature_nodes, predictions)
    logger.info("End anamod master pipeline")


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
                           static_indices=Feature.unpack_indices(row[constants.STATIC_INDICES]),
                           temporal_indices=Feature.unpack_indices(row[constants.TEMPORAL_INDICES]))
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
    all_static_indices = set()
    all_temporal_indices = set()
    for node in anytree.PostOrderIter(root):
        if node.is_leaf:
            assert node.static_indices or node.temporal_indices, "Leaf node %s must have at least one index of either type" % node.name
            assert not all_static_indices.intersection(node.static_indices), "Leaf node %s has static index overlap with other leaf nodes" % node.name
            assert not all_temporal_indices.intersection(node.temporal_indices), \
                    "Leaf node %s has temporal index overlap with other leaf nodes" % node.name
        else:
            # Ensure non-leaf nodes have empty initial indices
            assert not node.static_indices, "Non-leaf node %s has non-empty initial indices" % node.name
            assert not node.temporal_indices, "Non-leaf node %s has non-empty initial indices" % node.name
    # Populate data structures
    for node in anytree.PostOrderIter(root):
        for child in node.children:
            node.static_indices += child.static_indices
            node.temporal_indices += child.temporal_indices
    return root


def flatten_hierarchy(args, hierarchy_root):
    """
    Flatten hierarchy to allow partitioning across workers

    Args:
        hierarchy_root: root of feature hierarchy

    Returns:
        Flattened hierarchy comprising list of features/feature groups
    """
    nodes = list(anytree.PreOrderIter(hierarchy_root))
    nodes.append(Feature(constants.BASELINE, description="No perturbation"))  # Baseline corresponds to no perturbation
    nodes.sort(key=lambda node: node.name)  # For reproducibility across python versions
    args.rng.shuffle(nodes)  # To balance load across workers
    return nodes


def perturb_features(args, logger, feature_nodes):
    """
    Perturb features, observe effect on model loss and aggregate results

    Args:
        args: Command-line arguments
        feature_nodes: flattened feature hierarchy comprising nodes for base features/feature groups

    Returns:
        Aggregated results from workers
    """
    # Partition features, Launch workers, Aggregate results
    worker_pipeline = SerialPipeline(args, logger, feature_nodes)
    if args.condor:
        worker_pipeline = CondorPipeline(args, logger, feature_nodes)
    return worker_pipeline.run()


def compute_p_values(args, hierarchy_root, losses, predictions):
    """Evaluates and compares different feature erasures"""
    # pylint: disable = too-many-locals
    losses = round_vectordict(losses)
    predictions = round_vectordict(predictions)
    outfile = open("%s/%s" % (args.output_dir, constants.PVALUES_FILENAME), "w", newline="")
    writer = csv.writer(outfile, delimiter=",")
    writer.writerow([constants.NODE_NAME, constants.PARENT_NAME, constants.DESCRIPTION, constants.EFFECT_SIZE,
                     constants.MEAN_LOSS, constants.PVALUE_LOSSES])
    baseline_loss = losses[constants.BASELINE]
    mean_baseline_loss = np.mean(baseline_loss)
    for node in anytree.PreOrderIter(hierarchy_root):
        name = node.name
        parent_name = node.parent.name if node.parent else ""
        loss = losses[node.name]
        mean_loss = np.mean(loss)
        pvalue_loss = compute_p_value(baseline_loss, loss)
        effect_size = mean_loss - mean_baseline_loss
        writer.writerow([name, parent_name, node.description, effect_size, mean_loss, pvalue_loss])
    outfile.close()


def hierarchical_fdr(args, logger):
    """Performs hierarchical FDR control on results"""
    input_filename = "%s/%s" % (args.output_dir, constants.PVALUES_FILENAME)
    output_dir = "%s/%s" % (args.output_dir, constants.HIERARCHICAL_FDR_DIR)
    cmd = ("python -m anamod.fdr.hierarchical_fdr_control -output_dir %s -procedure yekutieli "
           "-rectangle_leaves %s" % (output_dir, input_filename))
    logger.info("Running cmd: %s" % cmd)
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        hierarchical_fdr_control.main()


def validate_args(args):
    """Validate arguments"""
    if args.analyze_interactions and args.perturbation == constants.SHUFFLING:
        raise ValueError("Interaction analysis is not supported with shuffling perturbations")


if __name__ == "__main__":
    main()
