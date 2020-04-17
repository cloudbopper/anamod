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
from anamod.feature import Feature
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
    args.logger.info("Begin anamod master pipeline with args: %s" % args)
    hierarchy_root = load_hierarchy(args.hierarchy_filename)
    features = list(anytree.PreOrderIter(hierarchy_root))  # flatten hierarchy
    if args.analysis_type == constants.TEMPORAL:
        features = features[1:]  # Remove dummy root node; TODO: get rid of hierarchy file altogether to avoid this
    # Prepare features for perturbation
    prepare_features(args, features)
    # Perturb features
    features, predictions = perturb_features(args, features)
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
    worker_pipeline = CondorPipeline(args, feature_nodes) if args.condor else SerialPipeline(args, feature_nodes)
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
    if args.condor:
        try:
            importlib.import_module("htcondor")
        except ModuleNotFoundError:
            raise ModuleNotFoundError("htcondor module not found. "
                                      "Use 'pip install htcondor' to install htcondor on a compatible platform, or "
                                      "disable condor by passing command-line argument -condor 0'")
