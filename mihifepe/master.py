"""
mihifepe master pipeline
Given test samples, a trained model and a feature hierarchy,
computes the effect on the model's output loss after perturbing the features/feature groups
in the hierarchy.
"""

import argparse
import csv
import logging
import os
import subprocess

import anytree
import numpy as np
from sklearn.metrics import roc_auc_score

from mihifepe import constants
from mihifepe.compute_p_values import compute_p_value
from mihifepe.feature import Feature
from mihifepe.pipelines import SerialPipeline, CondorPipeline


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
                        help="Model type - output includes perturbed AUROCs for binary classifiers",
                        choices=[constants.BINARY_CLASSIFIER, constants.CLASSIFIER, constants.REGRESSION],)

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    np.random.seed(constants.SEED)
    logging.basicConfig(level=logging.INFO, filename="%s/master.log" % args.output_dir,
                        format="%(asctime)s: %(message)s")
    logger = logging.getLogger(__name__)
    pipeline(args, logger)


def pipeline(args, logger):
    """Master pipeline"""
    logger.info("Begin mihifepe master pipeline with args: %s" % args)
    # Load hierarchy from file
    hierarchy_root = load_hierarchy(args.hierarchy_filename)
    # Flatten hierarchy to allow partitioning across workers
    feature_nodes = flatten_hierarchy(hierarchy_root)
    # Perturb features
    targets, losses, predictions = perturb_features(args, logger, feature_nodes)
    # Compute p-values
    losses, predictions = round_vectors(losses, predictions)
    compute_p_values(args, hierarchy_root, targets, losses, predictions)
    # Run hierarchical FDR
    hierarchical_fdr(args, logger)
    logger.info("End mihifepe master pipeline")


def load_hierarchy(hierarchy_filename):
    """
    Load hierarchy from CSV.

    Args:
        hierarchy_filename: CSV specifying hierarchy in required format (see mihifepe/spec.md)

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


def flatten_hierarchy(hierarchy_root):
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
    np.random.shuffle(nodes)  # To balance load across workers
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


def round_vectors(losses, predictions):
    """Round to 4 decimals to avoid floating-point errors"""

    def round_vectordict(vectordict):
        """Round dictionary of vectors"""
        return {key: np.around(value, decimals=4) for (key, value) in vectordict.items()}

    losses = round_vectordict(losses)
    predictions = round_vectordict(predictions)
    return losses, predictions


def compute_p_values(args, hierarchy_root, targets, losses, predictions):
    """Evaluates and compares different feature erasures"""
    # pylint: disable = too-many-locals
    outfile = open("%s/%s" % (args.output_dir, constants.PVALUES_FILENAME), "w", newline="")
    writer = csv.writer(outfile, delimiter=",")
    writer.writerow([constants.NODE_NAME, constants.PARENT_NAME, constants.DESCRIPTION, constants.EFFECT_SIZE,
                     constants.MEAN_LOSS, constants.PVALUE_LOSSES])
    baseline_loss = losses[constants.BASELINE]
    for node in anytree.PreOrderIter(hierarchy_root):
        name = node.name
        parent_name = node.parent.name if node.parent else ""
        loss = losses[node.name]
        mean_loss = np.mean(loss)
        pvalue_loss = compute_p_value(baseline_loss, loss)
        # Compute AUROC depending on whether task is binary classification or not:
        auroc = ""
        if args.model_type == constants.BINARY_CLASSIFIER:
            prediction = predictions[node.name]
            auroc = roc_auc_score(targets, prediction)
        writer.writerow([name, parent_name, node.description, auroc, mean_loss, pvalue_loss])
    outfile.close()


def hierarchical_fdr(args, logger):
    """Performs hierarchical FDR control on results"""
    input_filename = "%s/%s" % (args.output_dir, constants.PVALUES_FILENAME)
    output_dir = "%s/%s" % (args.output_dir, constants.HIERARCHICAL_FDR_DIR)
    cmd = ("python -m mihifepe.fdr.hierarchical_fdr_control -output_dir %s -procedure yekutieli "
           "-rectangle_leaves %s" % (output_dir, input_filename))
    logger.info("Running cmd: %s" % cmd)
    subprocess.check_call(cmd, shell=True)


if __name__ == "__main__":
    main()
