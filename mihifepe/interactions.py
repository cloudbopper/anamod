"""Tests pairwise interactions given output of mihifepe"""

import csv
import itertools
import subprocess

import anytree
from anytree.importer import JsonImporter
from mihifepe.compute_p_values import compute_p_value
from mihifepe import constants
from mihifepe.feature import Feature
from mihifepe.pipelines import CondorPipeline, SerialPipeline, round_vectordict, round_vector


def analyze_interactions(args, logger, feature_nodes, predictions):
    """Analyzes pairwise interactions among (relevant) features"""
    logger.info("Begin analyzing interactions")
    # TODO: if args.analyze_all_pairwise_interactions:
    # Load post-hierarchical FDR tree
    fdr_tree_node_map = get_fdr_tree_node_map(args)
    # Identify relevant features and feature pairs
    relevant_feature_nodes, _ = get_relevant_features(feature_nodes, fdr_tree_node_map)
    # Identify potential interactions to test
    potential_interactions = itertools.combinations(relevant_feature_nodes, 2)
    # TODO: Remove interactions already tested
    # Transform into nodes for testing
    interaction_nodes = get_interaction_nodes(potential_interactions)
    # Perturb interaction nodes
    interaction_predictions = perturb_interaction_nodes(args, logger, interaction_nodes)
    # Compute p-values
    compute_p_values(args, interaction_predictions, predictions)
    # Perform BH procedure on interaction p-values
    bh_procedure(args, logger)
    logger.info("End analyzing interactions")


def bh_procedure(args, logger):
    """Performs BH procedure on interaction p-values"""
    # TODO: Directly use BH procedure
    input_filename = "%s/%s" % (args.output_dir, constants.INTERACTIONS_PVALUES_FILENAME)
    output_dir = "%s/%s" % (args.output_dir, constants.INTERACTIONS_FDR_DIR)
    cmd = ("python -m mihifepe.fdr.hierarchical_fdr_control -output_dir %s -procedure yekutieli "
           "-rectangle_leaves %s" % (output_dir, input_filename))
    logger.info("Running cmd: %s" % cmd)
    subprocess.check_call(cmd, shell=True)


def compute_p_values(args, interaction_predictions, predictions):
    """Computes p-values for assessing interaction significance"""
    # TODO: handle non-identity transfer function
    interaction_predictions = round_vectordict(interaction_predictions)
    outfile = open("%s/%s" % (args.output_dir, constants.INTERACTIONS_PVALUES_FILENAME), "w", newline="")
    writer = csv.writer(outfile, delimiter=",")
    # Construct two-level hierarchy with dummy root node and interactions as its children
    # Leverages existing hierarchical FDR code to perform BH procedure on interactions
    # TODO: Directly use BH procedure
    # TODO: Verify this approach works for shuffling perturbations, since shuffling randomization may
    # be different when A and B perturbed together vs. when A and B perturbed separately
    writer.writerow([constants.NODE_NAME, constants.PARENT_NAME, constants.DESCRIPTION, constants.EFFECT_SIZE,
                     constants.MEAN_LOSS, constants.PVALUE_LOSSES])
    writer.writerow([constants.DUMMY_ROOT, "", "", "", "", 0.])
    baseline_prediction = predictions[constants.BASELINE]
    for name in interaction_predictions.keys():
        left, right = name.split(" + ")
        lhs = interaction_predictions[name]
        rhs = round_vector(predictions[left] + predictions[right] - baseline_prediction)
        pvalue = compute_p_value(lhs, rhs, alternative=constants.TWOSIDED)
        writer.writerow([name, constants.DUMMY_ROOT, "", "", "", pvalue])
    outfile.close()


def perturb_interaction_nodes(args, logger, interaction_nodes):
    """Perturb interaction nodes, observe effect on model loss and aggregate results"""
    logger.info("Begin perturbing interaction nodes")
    worker_pipeline = SerialPipeline(args, logger, interaction_nodes)
    if args.condor:
        worker_pipeline = CondorPipeline(args, logger, interaction_nodes)
    _, _, interaction_predictions = worker_pipeline.run()
    logger.info("End perturbing interaction nodes")
    return interaction_predictions


def get_interaction_nodes(potential_interactions):
    """Transform into nodes for testing"""
    interaction_nodes = []
    for left, right in potential_interactions:
        name = left.name + " + " + right.name
        node = Feature(name, static_indices=left.static_indices + right.static_indices,
                       temporal_indices=left.temporal_indices + right.temporal_indices)
        interaction_nodes.append(node)
    return interaction_nodes


def get_relevant_features(feature_nodes, fdr_tree_node_map):
    """Identify relevant features and feature pairs"""
    relevant_feature_nodes = []
    relevant_pair_map = {}  # map of relevant feature pair names to nodes
    for node in feature_nodes:
        name = node.name
        if name == constants.BASELINE or not fdr_tree_node_map[name].rejected:
            continue
        if node.is_leaf:
            relevant_feature_nodes.append(node)
        elif len(node.children) == 2:
            relevant_pair_map[name] = node
    return relevant_feature_nodes, relevant_pair_map


def get_fdr_tree_node_map(args):
    """Get list of nodes outputted by hierarchical FDR procedure on features"""
    fdr_tree_filename = "%s/%s/%s.json" % (args.output_dir, constants.HIERARCHICAL_FDR_DIR, constants.HIERARCHICAL_FDR_OUTPUTS)
    with open(fdr_tree_filename, "r") as fdr_tree_file:
        fdr_tree = JsonImporter().read(fdr_tree_file)
        fdr_tree_node_map = {node.name: node for node in anytree.PreOrderIter(fdr_tree)}
    return fdr_tree_node_map
