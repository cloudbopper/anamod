"""Generates simulated data and model to test anamod algorithm"""

import argparse
from collections import namedtuple
import copy
import csv
import os
import pickle
import sys
from unittest.mock import patch

import anytree
from anytree.importer import JsonImporter
import cloudpickle
import h5py
import numpy as np
from scipy.cluster.hierarchy import linkage
from sklearn.metrics import precision_recall_fscore_support
import synmod.master

from anamod import constants, master, utils
from anamod.fdr import hierarchical_fdr_control
from anamod.simulation.model_wrapper import ModelWrapper

# TODO maybe: write arguments to separate readme.txt for documenting runs

# Simulation results object
HierarchicalResults = namedtuple(constants.SIMULATION_RESULTS, [constants.FDR, constants.POWER,
                                                                constants.BASE_FEATURES_FDR, constants.BASE_FEATURES_POWER,
                                                                constants.INTERACTIONS_FDR, constants.INTERACTIONS_POWER])

TemporalResults = namedtuple(constants.SIMULATION_RESULTS, [constants.FDR, constants.POWER,
                                                            constants.TEMPORAL_FDR, constants.TEMPORAL_POWER,
                                                            constants.AVERAGE_WINDOW_FDR, constants.AVERAGE_WINDOW_POWER])


def main():
    """Main"""
    parser = argparse.ArgumentParser("python anamod.simulation")
    # Required arguments
    required = parser.add_argument_group("Required parameters")
    required.add_argument("-output_dir", help="Name of output directory")
    # Optional common arguments
    common = parser.add_argument_group("Optional common parameters")
    common.add_argument("-analysis_type", help="Type of model analysis to perform",
                        default=constants.HIERARCHICAL, choices=[constants.TEMPORAL, constants.HIERARCHICAL])
    common.add_argument("-seed", type=int, default=constants.SEED)
    common.add_argument("-num_instances", type=int, default=10000)
    common.add_argument("-num_features", type=int, default=100)
    common.add_argument("-fraction_relevant_features", type=float, default=.05)
    common.add_argument("-num_interactions", type=int, default=0, help="number of interaction pairs in model")
    common.add_argument("-exclude_interaction_only_features", help="exclude interaction-only features in model"
                        " in addition to linear + interaction features (default included)", action="store_false",
                        dest="include_interaction_only_features")
    common.set_defaults(include_interaction_only_features=True)
    # Hierarchical feature importance analysis arguments
    hierarchical = parser.add_argument_group("Hierarchical feature analysis arguments")
    hierarchical.add_argument("-noise_multiplier", type=float, default=.05,
                              help="Multiplicative factor for noise added to polynomial computation for irrelevant features")
    hierarchical.add_argument("-noise_type", choices=[constants.ADDITIVE_GAUSSIAN, constants.EPSILON_IRRELEVANT, constants.NO_NOISE],
                              default=constants.EPSILON_IRRELEVANT)
    hierarchical.add_argument("-hierarchy_type", help="Choice of hierarchy to generate", default=constants.CLUSTER_FROM_DATA,
                              choices=[constants.CLUSTER_FROM_DATA, constants.RANDOM])
    hierarchical.add_argument("-contiguous_node_names", action="store_true", help="enable to change node names in hierarchy "
                              "to be contiguous for better visualization (but creating mismatch between node names and features indices)")
    hierarchical.add_argument("-analyze_interactions", help="enable analyzing interactions", action="store_true")
    hierarchical.add_argument("-perturbation", default=constants.SHUFFLING, choices=[constants.ZEROING, constants.SHUFFLING])
    hierarchical.add_argument("-num_shuffling_trials", type=int, default=100, help="Number of shuffling trials to average over, "
                              "when shuffling perturbations are selected")
    # Temporal model analysis arguments
    temporal = parser.add_argument_group("Temporal model analysis arguments")
    temporal.add_argument("-sequence_length", help="sequence length for temporal models", default=20)
    temporal.add_argument("-model_type", default=constants.REGRESSION)
    temporal.add_argument("-sequences_independent_of_windows", action="store_true", dest="window_independent")
    temporal.add_argument("-sequences_dependent_on_windows", action="store_false", dest="window_independent")
    temporal.set_defaults(window_independent=False)

    args, pass_args = parser.parse_known_args()
    pass_args = " ".join(pass_args)
    if not args.output_dir:
        args.output_dir = ("sim_outputs_inst_%d_feat_%d_noise_%.3f_relfraction_%.3f_pert_%s_shufftrials_%d" %
                           (args.num_instances, args.num_features, args.noise_multiplier,
                            args.fraction_relevant_features, args.perturbation, args.num_shuffling_trials))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.rng = np.random.default_rng(args.seed)
    args.logger = utils.get_logger(__name__, "%s/simulation.log" % args.output_dir)
    if args.analysis_type == constants.TEMPORAL:
        args.noise_type = constants.NO_NOISE
    return pipeline(args, pass_args)


def pipeline(args, pass_args):
    """Simulation pipeline"""
    args.logger.info("Begin anamod simulation with args: %s" % args)
    features, data, model = run_synmod(args)
    targets = model.predict(data, labels=True) if args.model_type == constants.CLASSIFIER else model.predict(data)
    data_filename = write_data(args, data, targets)
    model_filename = write_model(args, model)
    if args.analysis_type == constants.HIERARCHICAL:
        # Generate hierarchy using clustering (test data also used for clustering)
        hierarchy_root, feature_id_map = gen_hierarchy(args, data)
        # Update hierarchy descriptions for future visualization
        update_hierarchy_relevance(hierarchy_root, model.relevant_feature_map, features)
        # Write hierarchy to file
        hierarchy_filename = write_hierarchy(args, hierarchy_root)
        # Invoke feature importance algorithm
        run_anamod(args, pass_args, data_filename, model_filename, hierarchy_filename)
        # Compare anamod outputs with ground truth outputs
        compare_with_ground_truth(args, hierarchy_root)
        # Evaluate anamod outputs - power/FDR for all nodes/outer nodes/base features
        results = evaluate_hierarchical(args, model.relevant_feature_map, feature_id_map)
    else:
        # Temporal model analysis
        # FIXME: should have similar mode of parsing outputs for both analyses
        analyzed_features = run_anamod(args, pass_args, data_filename, model_filename)
        results = evaluate_temporal(args, model, analyzed_features)
    args.logger.info("Results:\n%s" % str(results))
    write_results(args, results)
    args.logger.info("End anamod simulation")
    return results


def run_synmod(args):
    """Synthesize data and model"""
    pass_args = copy.copy(args)
    if args.analysis_type == constants.HIERARCHICAL:
        pass_args.synthesis_type = constants.STATIC
    else:
        pass_args.synthesis_type = constants.TEMPORAL
    return synmod.master.pipeline(pass_args)


def gen_hierarchy(args, clustering_data):
    """
    Generate hierarchy over features

    Args:
        args: Command-line arguments
        clustering_data: Data potentially used to cluster features
                         (depending on hierarchy generation method)

    Returns:
        hierarchy_root: root fo resulting hierarchy over features
    """
    # Generate hierarchy
    hierarchy_root = None
    if args.hierarchy_type == constants.CLUSTER_FROM_DATA:
        clusters = cluster_data(args, clustering_data)
        hierarchy_root = gen_hierarchy_from_clusters(args, clusters)
    elif args.hierarchy_type == constants.RANDOM:
        hierarchy_root = gen_random_hierarchy(args)
    else:
        raise NotImplementedError("Need valid hierarchy type")
    # Improve visualization - contiguous feature names
    feature_id_map = {}  # mapping from visual feature ids to original ids
    if args.contiguous_node_names:
        for idx, node in enumerate(anytree.PostOrderIter(hierarchy_root)):
            node.vidx = idx
            if node.is_leaf:
                node.min_child_vidx = idx
                node.max_child_vidx = idx
                node.num_base_features = 1
                node.name = str(idx)
                feature_id_map[idx] = int(node.idx)
            else:
                node.min_child_vidx = min([child.min_child_vidx for child in node.children])
                node.max_child_vidx = max([child.vidx for child in node.children])
                node.num_base_features = sum([child.num_base_features for child in node.children])
                node.name = "[%d-%d] (size: %d)" % (node.min_child_vidx, node.max_child_vidx, node.num_base_features)
    return hierarchy_root, feature_id_map


def gen_random_hierarchy(args):
    """Generates balanced random hierarchy"""
    args.logger.info("Begin generating hierarchy")
    nodes = [anytree.Node(str(idx), idx=str(idx)) for idx in range(args.num_features)]
    args.rng.shuffle(nodes)
    node_count = len(nodes)
    while len(nodes) > 1:
        parents = []
        for left_idx in range(0, len(nodes), 2):
            parent = anytree.Node(str(node_count))
            node_count += 1
            nodes[left_idx].parent = parent
            right_idx = left_idx + 1
            if right_idx < len(nodes):
                nodes[right_idx].parent = parent
            parents.append(parent)
        nodes = parents
    hierarchy_root = nodes[0]
    args.logger.info("End generating hierarchy")
    return hierarchy_root


def cluster_data(args, data):
    """Cluster data using hierarchical clustering with Hamming distance"""
    # Cluster data
    args.logger.info("Begin clustering data")
    clusters = linkage(data.transpose(), metric="hamming", method="complete")
    args.logger.info("End clustering data")
    return clusters


def gen_hierarchy_from_clusters(args, clusters):
    """
    Organize clusters into hierarchy

    Args:
        clusters: linkage matrix (num_features-1 X 4)
                  rows indicate successive clustering iterations
                  columns, respectively: 1st cluster index, 2nd cluster index, distance, sample count
    Returns:
        hierarchy_root: root of resulting hierarchy over features
    """
    # Generate hierarchy from clusters
    nodes = [anytree.Node(str(idx), idx=str(idx)) for idx in range(args.num_features)]
    for idx, cluster in enumerate(clusters):
        cluster_idx = idx + args.num_features
        left_idx, right_idx, _, _ = cluster
        left_idx = int(left_idx)
        right_idx = int(right_idx)
        cluster_node = anytree.Node(str(cluster_idx))
        nodes[left_idx].parent = cluster_node
        nodes[right_idx].parent = cluster_node
        nodes.append(cluster_node)
    hierarchy_root = nodes[-1]
    return hierarchy_root


def update_hierarchy_relevance(hierarchy_root, relevant_feature_map, features):
    """
    Add feature relevance information to nodes of hierarchy:
    their probabilty of being enabled,
    their polynomial coefficient
    """
    probs = [feature.prob for feature in features]
    relevant_features = set()
    for key in relevant_feature_map:
        relevant_features.update(key)
    for node in anytree.PostOrderIter(hierarchy_root):
        node.description = constants.IRRELEVANT
        if node.is_leaf:
            idx = int(node.idx)
            node.poly_coeff = 0.0
            node.bin_prob = probs[idx]
            coeff = relevant_feature_map.get(frozenset([idx]))
            if coeff:
                node.poly_coeff = coeff
                node.description = ("%s feature:\nPolynomial coefficient: %f\nBinomial probability: %f"
                                    % (constants.RELEVANT, coeff, probs[idx]))
            elif idx in relevant_features:
                node.description = ("%s feature\n(Interaction-only)\nBinomial probability: %f"
                                    % (constants.RELEVANT, probs[idx]))
        else:
            for child in node.children:
                if child.description != constants.IRRELEVANT:
                    node.description = constants.RELEVANT


def write_data(args, data, targets):
    """Write data in HDF5 format"""
    data_filename = "%s/%s" % (args.output_dir, "data.hdf5")
    root = h5py.File(data_filename, "w")
    record_ids = [str(idx).encode("utf8") for idx in range(args.num_instances)]
    root.create_dataset(constants.RECORD_IDS, data=record_ids)
    root.create_dataset(constants.TARGETS, data=targets)
    root.create_dataset(constants.DATA, data=data)
    root.close()
    return data_filename


def write_hierarchy(args, hierarchy_root):
    """
    Write hierarchy in CSV format.

    Columns:    *name*:             feature name, must be unique across features
                *parent_name*:      name of parent if it exists, else '' (root node)
                *description*:      node description
                *idx*:              [only required for leaf nodes] list of tab-separated indices corresponding to the indices
                                    of these features in the data
    """
    hierarchy_filename = "%s/%s" % (args.output_dir, "hierarchy.csv")
    with open(hierarchy_filename, "w", newline="") as hierarchy_file:
        writer = csv.writer(hierarchy_file, delimiter=",")
        writer.writerow([constants.NODE_NAME, constants.PARENT_NAME,
                         constants.DESCRIPTION, constants.INDICES])
        for node in anytree.PreOrderIter(hierarchy_root):
            idx = node.idx if node.is_leaf else ""
            parent_name = node.parent.name if node.parent else ""
            writer.writerow([node.name, parent_name, node.description, idx])
    return hierarchy_filename


def write_model(args, model):
    """
    Pickle and write model to file in output directory.
    """
    # Create wrapper around ground-truth model
    model_wrapper = ModelWrapper(model, args.num_features, args.noise_type, args.noise_multiplier)
    # Pickle and write to file
    model_filename = "%s/%s" % (args.output_dir, constants.MODEL_FILENAME)
    with open(model_filename, "wb") as model_file:
        cloudpickle.dump(model_wrapper, model_file)
    return model_filename


def run_anamod(args, pass_args, data_filename, model_filename, hierarchy_filename=""):
    """Run analysis algorithms"""
    args.logger.info("Begin running anamod")
    analyze_interactions = "-analyze_interactions" if args.analyze_interactions else ""
    hierarchical_analysis_options = ("-hierarchy_filename {0} -perturbation {1} {2}".format
                                     (hierarchy_filename, args.perturbation, analyze_interactions))
    temporal_analysis_options = ""
    analysis_options = hierarchical_analysis_options if args.analysis_type == constants.HIERARCHICAL else temporal_analysis_options
    args.logger.info("Passing the following arguments to anamod.master without parsing: %s" % pass_args)
    memory_requirement = 1 + (os.stat(data_filename).st_size // (2 ** 30))  # Compute approximate memory requirement in GB
    cmd = ("python -m anamod.master -analysis_type {0} -output_dir {1} -num_shuffling_trials {2} -data_filename {3} "
           "-model_filename {4} -memory_requirement {5} {6} {7}"
           .format(args.analysis_type,
                   args.output_dir,
                   args.num_shuffling_trials,
                   data_filename,
                   model_filename,
                   memory_requirement,
                   analysis_options,
                   pass_args))
    args.logger.info("Running cmd: %s" % cmd)
    nargs = cmd.split()[2:]
    with patch.object(sys, 'argv', nargs):
        features = master.main()
    args.logger.info("End running anamod")
    return features


def compare_with_ground_truth(args, hierarchy_root):
    """Compare results from anamod with ground truth results"""
    # Generate ground truth results
    # Write hierarchical FDR input file for ground truth values
    args.logger.info("Compare anamod results to ground truth")
    input_filename = "%s/ground_truth_pvalues.csv" % args.output_dir
    with open(input_filename, "w", newline="") as input_file:
        writer = csv.writer(input_file)
        writer.writerow([constants.NODE_NAME, constants.PARENT_NAME, constants.PVALUE_LOSSES, constants.DESCRIPTION])
        for node in anytree.PostOrderIter(hierarchy_root):
            parent_name = node.parent.name if node.parent else ""
            # Decide p-values based on rough heuristic for relevance
            node.pvalue = 1.0
            if node.description != constants.IRRELEVANT:
                if node.is_leaf:
                    node.pvalue = 0.001
                    if node.poly_coeff:
                        node.pvalue = min(node.pvalue, 1e-10 / (node.poly_coeff * node.bin_prob) ** 3)
                else:
                    node.pvalue = 0.999 * min([child.pvalue for child in node.children])
            writer.writerow([node.name, parent_name, node.pvalue, node.description])
    # Generate hierarchical FDR results for ground truth values
    ground_truth_dir = "%s/ground_truth_fdr" % args.output_dir
    cmd = ("python -m anamod.fdr.hierarchical_fdr_control -output_dir %s -procedure yekutieli "
           "-rectangle_leaves %s" % (ground_truth_dir, input_filename))
    args.logger.info("Running cmd: %s" % cmd)
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        hierarchical_fdr_control.main()
    # Compare results
    ground_truth_outputs_filename = "%s/%s.png" % (ground_truth_dir, constants.TREE)
    args.logger.info("Ground truth results: %s" % ground_truth_outputs_filename)
    anamod_outputs_filename = "%s/%s/%s.png" % (args.output_dir, constants.HIERARCHICAL_FDR_DIR, constants.TREE)
    args.logger.info("anamod results: %s" % anamod_outputs_filename)


def evaluate_hierarchical(args, relevant_feature_map, feature_id_map):
    """
    Evaluate hierarchical analysis results - obtain power/FDR measures for all nodes/base features/interactions
    """
    # pylint: disable = too-many-locals
    def get_relevant_rejected(nodes, leaves=False):
        """Get set of relevant and rejected nodes"""
        if leaves:
            nodes = [node for node in nodes if node.is_leaf]
        relevant = [0 if node.description == constants.IRRELEVANT else 1 for node in nodes]
        rejected = [1 if node.rejected else 0 for node in nodes]
        return relevant, rejected

    tree_filename = "%s/%s/%s.json" % (args.output_dir, constants.HIERARCHICAL_FDR_DIR, constants.HIERARCHICAL_FDR_OUTPUTS)
    with open(tree_filename, "r") as tree_file:
        tree = JsonImporter().read(tree_file)
        nodes = list(anytree.PreOrderIter(tree))
        # All nodes FDR/power
        relevant, rejected = get_relevant_rejected(nodes)
        precision, recall, _, _ = precision_recall_fscore_support(relevant, rejected, average="binary")
        # Base features FDR/power
        bf_relevant, bf_rejected = get_relevant_rejected(nodes, leaves=True)
        bf_precision, bf_recall, _, _ = precision_recall_fscore_support(bf_relevant, bf_rejected, average="binary")
        # Interactions FDR/power
        interaction_precision, interaction_recall = get_precision_recall_interactions(args, relevant_feature_map, feature_id_map)

        return HierarchicalResults(1 - precision, recall, 1 - bf_precision, bf_recall, 1 - interaction_precision, interaction_recall)


def evaluate_temporal(args, model, features):
    """Evaluate results of temporal model analysis - obtain power/FDR measures for importance, temporal importance and windows"""
    # pylint: disable = protected-access, too-many-locals

    def init_vectors():
        """Initialize vectors indicating importances"""
        important = [False for idx, _ in enumerate(features)]
        temporally_important = [False for idx, _ in enumerate(features)]
        windows = np.zeros((len(features), args.sequence_length))
        return important, temporally_important, windows

    # Populate importance vectors (ground truth and inferred)
    features = sorted(features, key=lambda feature: feature.idx[0])  # To ensure features are ordered by their index in the feature vector
    important, temporally_important, windows = init_vectors()
    inferred_important, inferred_temporally_important, inferred_windows = init_vectors()
    for idx, feature in enumerate(features):
        assert idx == feature.idx[0]
        # Ground truth values
        if model.relevant_feature_map.get(frozenset({idx})):
            important[idx] = True
            left, right = model._operation._windows[idx]
            if right - left + 1 < args.sequence_length:
                temporally_important[idx] = True
                windows[idx][left: right + 1] = 1  # algorithm doesn't test for window unless feature is identified as temporally important
        # Inferred values
        inferred_important[idx] = feature.important
        inferred_temporally_important[idx] = feature.temporally_important
        if inferred_temporally_important[idx]:
            left, right = feature.temporal_window
            inferred_windows[idx][left: right + 1] = 1
    # Get scores
    imp_precision, imp_recall, _, _ = precision_recall_fscore_support(important, inferred_important, average="binary")
    timp_precision, timp_recall, _, _ = precision_recall_fscore_support(temporally_important, inferred_temporally_important, average="binary")
    avg_window_precision, avg_window_recall = (0, 0)
    num_windows = 0
    for idx, _ in enumerate(features):
        if not inferred_temporally_important[idx]:
            continue  # Don't include features unless identified as temporally relevant
        window_precision, window_recall, _, _ = precision_recall_fscore_support(windows[idx], inferred_windows[idx], average="binary")
        avg_window_precision += window_precision
        avg_window_recall += window_recall
        num_windows += 1
    avg_window_precision /= num_windows
    avg_window_recall /= num_windows
    return TemporalResults(1 - imp_precision, imp_recall, 1 - timp_precision, timp_recall, 1 - avg_window_precision, avg_window_recall)


def get_precision_recall_interactions(args, relevant_feature_map, feature_id_map):
    """Computes precision (1 - FDR) and recall (power) for detecting interactions"""
    # pylint: disable = invalid-name, too-many-locals
    # The set of all possible interactions might be very big, so don't construct label vector for all
    # possible interactions - compute precision/recall from basics
    # TODO: alter to handle higher-order interactions
    if not args.analyze_interactions:
        return (0.0, 0.0)
    true_interactions = {key for key in relevant_feature_map.keys() if len(key) > 1}
    tree_filename = "%s/%s/%s.json" % (args.output_dir, constants.INTERACTIONS_FDR_DIR, constants.HIERARCHICAL_FDR_OUTPUTS)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    tested = set()
    with open(tree_filename, "r") as tree_file:
        tree = JsonImporter().read(tree_file)
        # Two-level tree with tested interactions on level 2
        for node in tree.children:
            pair = frozenset({int(idx) for idx in node.name.split(" + ")})
            if feature_id_map:
                pair = frozenset({feature_id_map[visual_id] for visual_id in pair})
            tested.add(pair)
            if node.rejected:
                if relevant_feature_map.get(pair):
                    tp += 1
                else:
                    fp += 1
            else:
                if relevant_feature_map.get(pair):
                    fn += 1
                else:
                    tn += 1
    if not tp > 0:
        return (0.0, 0.0)
    missed = true_interactions.difference(tested)
    fn += len(missed)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall


def write_results(args, results):
    """Write results to pickle file"""
    results_filename = "%s/%s" % (args.output_dir, constants.SIMULATION_RESULTS_FILENAME)
    with open(results_filename, "wb") as results_file:
        pickle.dump(results._asdict(), results_file)


if __name__ == "__main__":
    main()
