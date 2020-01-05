"""Tests pairwise interactions given output of anamod"""

import copy
import csv
import itertools
import sys
import time
from unittest.mock import patch

import anytree
from anytree.importer import JsonImporter
import numpy as np

from anamod.compute_p_values import compute_p_value
from anamod import constants
from anamod.fdr import hierarchical_fdr_control
from anamod.feature import Feature
from anamod.pipelines import CondorPipeline, SerialPipeline, round_vector


def analyze_interactions(args, logger, feature_nodes, cached_predictions):
    """Analyzes pairwise interactions among (relevant) features"""
    if args.condor:
        time.sleep(5)  # To allow file changes from preceding analysis to propagate
    logger.info("Begin analyzing interactions")
    # Identify relevant features and feature pairs
    relevant_feature_nodes = get_relevant_features(args, feature_nodes)
    # Identify potential interactions to test
    potential_interactions = itertools.combinations(relevant_feature_nodes, 2)
    # TODO: Remove interactions already tested
    # Transform into nodes for testing
    interaction_groups = get_interaction_groups(args, potential_interactions)
    # Perturb interaction nodes
    interaction_predictions = perturb_interactions(args, logger, interaction_groups)
    # Compute p-values
    compute_p_values(args, interaction_groups, interaction_predictions, cached_predictions)
    # Perform BH procedure on interaction p-values
    bh_procedure(args, logger)
    logger.info("End analyzing interactions")


def bh_procedure(args, logger):
    """Performs BH procedure on interaction p-values"""
    # TODO: Directly use BH procedure
    input_filename = "%s/%s" % (args.output_dir, constants.INTERACTIONS_PVALUES_FILENAME)
    output_dir = "%s/%s" % (args.output_dir, constants.INTERACTIONS_FDR_DIR)
    cmd = ("python -m anamod.fdr.hierarchical_fdr_control -output_dir %s -procedure yekutieli "
           "-rectangle_leaves %s" % (output_dir, input_filename))
    logger.info("Running cmd: %s" % cmd)
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        hierarchical_fdr_control.main()


def compute_p_values(args, interaction_groups, interaction_predictions, cached_predictions):
    """Computes p-values for assessing interaction significance"""
    # TODO: handle non-identity transfer function
    outfile = open("%s/%s" % (args.output_dir, constants.INTERACTIONS_PVALUES_FILENAME), "w", newline="")
    writer = csv.writer(outfile, delimiter=",")
    # Construct two-level hierarchy with dummy root node and interactions as its children
    # Leverages existing hierarchical FDR code to perform BH procedure on interactions
    # TODO: Directly use BH procedure, maybe?
    # TODO: Since we're using outputs and not losses here, the p-values schema is misleading
    writer.writerow([constants.NODE_NAME, constants.PARENT_NAME, constants.DESCRIPTION, constants.EFFECT_SIZE,
                     constants.MEAN_LOSS, constants.PVALUE_LOSSES])
    writer.writerow([constants.DUMMY_ROOT, "", "", "", "", 0.])
    baseline_prediction = cached_predictions[constants.BASELINE]
    redo_predictions = interaction_predictions if args.perturbation == constants.SHUFFLING else cached_predictions
    for cached_node, redo_node, parent_node in interaction_groups:
        lhs = round_vector(interaction_predictions[parent_node.name])
        rhs = round_vector(cached_predictions[cached_node.name] + redo_predictions[redo_node.name] - baseline_prediction)
        pvalue = compute_p_value(lhs, rhs, alternative=constants.TWOSIDED)
        effect_size = np.mean(lhs - rhs)  # TODO: confirm sign
        # TODO: Add description?
        writer.writerow([parent_node.name, constants.DUMMY_ROOT, "", effect_size, "", pvalue])
    outfile.close()


def perturb_interactions(args, logger, interaction_groups):
    """Perturb interactions, observe effect on model loss and aggregate results"""
    logger.info("Begin perturbing interactions")
    interaction_nodes = []
    for _, redo_node, parent_node in interaction_groups:
        interaction_nodes.append(parent_node)
        if args.perturbation == constants.SHUFFLING:
            interaction_nodes.append(redo_node)
    worker_pipeline = SerialPipeline(args, logger, interaction_nodes)
    if args.condor:
        worker_pipeline = CondorPipeline(args, logger, interaction_nodes)
    _, _, interaction_predictions = worker_pipeline.run()
    logger.info("End perturbing interactions")
    return interaction_predictions


def get_interaction_groups(args, potential_interactions):
    """Transform into nodes for testing"""
    interaction_groups = []
    for left, right in potential_interactions:
        name = left.name + " + " + right.name
        parent_node = Feature(name, static_indices=left.static_indices + right.static_indices,
                              temporal_indices=left.temporal_indices + right.temporal_indices)
        if Feature.size(left) >= Feature.size(right):
            cached_node = left
            redo_node = right
        else:
            cached_node = right
            redo_node = left
        if args.perturbation == constants.SHUFFLING:  # Set attributes on redo_node
            redo_node = copy.deepcopy(redo_node)
            redo_node.uniquify(parent_node.name)
            redo_node.rng_seed = cached_node.rng_seed
        parent_node.rng_seed = cached_node.rng_seed  # Set attribute on parent_node
        interaction_groups.append((cached_node, redo_node, parent_node))
    return interaction_groups


def get_relevant_features(args, feature_nodes):
    """Identify relevant features and feature pairs"""
    candidate_nodes = [node for node in feature_nodes if node.is_leaf and node.name != constants.BASELINE]
    if not args.analyze_all_pairwise_interactions:
        # Get list of nodes outputted by hierarchical FDR procedure on features
        fdr_tree_filename = "%s/%s/%s.json" % (args.output_dir, constants.HIERARCHICAL_FDR_DIR,
                                               constants.HIERARCHICAL_FDR_OUTPUTS)
        with open(fdr_tree_filename, "r") as fdr_tree_file:
            fdr_tree = JsonImporter().read(fdr_tree_file)
            fdr_tree_node_map = {node.name: node for node in anytree.PreOrderIter(fdr_tree)}
            candidate_nodes = [node for node in candidate_nodes if fdr_tree_node_map[node.name].rejected]
    return candidate_nodes
