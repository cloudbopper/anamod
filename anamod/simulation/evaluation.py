"""Evaluate simulation results"""

import csv
import sys
from unittest.mock import patch
import warnings

import anytree
from anytree.importer import JsonImporter
import numpy as np
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

from anamod import constants
from anamod.constants import FDR, POWER, BASE_FEATURES_FDR, BASE_FEATURES_POWER, INTERACTIONS_FDR, INTERACTIONS_POWER
from anamod.constants import ORDERING_ALL_IMPORTANT_FDR, ORDERING_ALL_IMPORTANT_POWER
from anamod.constants import ORDERING_IDENTIFIED_IMPORTANT_FDR, ORDERING_IDENTIFIED_IMPORTANT_POWER
from anamod.constants import AVERAGE_WINDOW_FDR, AVERAGE_WINDOW_POWER, WINDOW_OVERLAP
from anamod.fdr import hierarchical_fdr_control


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
           "-rectangle_leaves 1 %s" % (ground_truth_dir, input_filename))
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
        return {FDR: 1 - precision, POWER: recall,
                BASE_FEATURES_FDR: 1 - bf_precision, BASE_FEATURES_POWER: bf_recall,
                INTERACTIONS_FDR: 1 - interaction_precision, INTERACTIONS_POWER: interaction_recall}


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
            left, right = model._aggregator._windows[idx]
            windows[idx][left: right + 1] = 1
            if right - left + 1 < args.sequence_length:
                temporally_important[idx] = True  # FIXME: will fail for ordering-relevant functions e.g. slope, weighted avg
            # TODO: Add flag for ordering relevance within window (asserted when aggregation function ordering-relevant )
        # Inferred values
        inferred_important[idx] = feature.important
        inferred_temporally_important[idx] = feature.temporally_important
        if feature.temporal_window is not None:
            left, right = feature.temporal_window
            inferred_windows[idx][left: right + 1] = 1
    # Get scores
    if not any(important or inferred_important):
        imp_precision, imp_recall = (1.0, 1.0)
    else:
        imp_precision, imp_recall, _, _ = precision_recall_fscore_support(important, inferred_important, average="binary", zero_division=0)
    if not any(temporally_important or inferred_temporally_important):
        timp_precision, timp_recall = (1.0, 1.0)
    else:
        timp_precision, timp_recall, _, _ = precision_recall_fscore_support(temporally_important, inferred_temporally_important,
                                                                            average="binary", zero_division=0)
    ordering_precision, ordering_recall = (timp_precision, timp_recall / imp_recall if imp_recall != 0 else 0)
    window_results = {}
    for idx, feature in enumerate(features):
        if model.relevant_feature_map.get(frozenset({idx})) is None and feature.temporal_window is None:
            continue  # Ignore irrelevant features that weren't misclassified
        window_precision, window_recall, _, _ = precision_recall_fscore_support(windows[idx], inferred_windows[idx],
                                                                                average="binary", zero_division=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Avoid warning if the two vectors have no common values
            window_overlap = balanced_accuracy_score(windows[idx], inferred_windows[idx])
        window_results[idx] = {"precision": window_precision, "recall": window_recall, "overlap": window_overlap}
    avg_window_precision = np.mean([result["precision"] for result in window_results.values()]) if window_results else 0.
    avg_window_recall = np.mean([result["recall"] for result in window_results.values()]) if window_results else 0.
    window_overlaps = {idx: result["overlap"] for idx, result in window_results.items()}
    # TODO: Replace 'temporally important' by 'ordering'
    return {FDR: 1 - imp_precision, POWER: imp_recall,
            ORDERING_ALL_IMPORTANT_FDR: 1 - timp_precision, ORDERING_ALL_IMPORTANT_POWER: timp_recall,
            ORDERING_IDENTIFIED_IMPORTANT_FDR: 1 - ordering_precision, ORDERING_IDENTIFIED_IMPORTANT_POWER: ordering_recall,
            AVERAGE_WINDOW_FDR: 1 - avg_window_precision, AVERAGE_WINDOW_POWER: avg_window_recall,
            WINDOW_OVERLAP: window_overlaps}


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
