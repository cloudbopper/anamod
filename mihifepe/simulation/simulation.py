"""Generates simulated data and model to test mihifepe algorithm"""

import argparse
from collections import namedtuple
import csv
import functools
import itertools
import logging
import os
import pickle
import subprocess

import anytree
from anytree.importer import JsonImporter
import h5py
import numpy as np
from scipy.cluster.hierarchy import linkage
import sympy
from sympy.utilities.lambdify import lambdify
from sklearn.metrics import precision_recall_fscore_support

from mihifepe import constants

# TODO maybe: write arguments to separate readme.txt for documentating runs

# Simulation results object
Results = namedtuple(constants.SIMULATION_RESULTS, [constants.FDR, constants.POWER, constants.OUTER_NODES_FDR,
                                                    constants.OUTER_NODES_POWER, constants.BASE_FEATURES_FDR, constants.BASE_FEATURES_POWER])


def main():
    """Main"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", type=int, default=constants.SEED)
    parser.add_argument("-num_instances", type=int, default=10000)
    parser.add_argument("-num_features", type=int, default=100)
    parser.add_argument("-output_dir", help="Name of output directory")
    parser.add_argument("-fraction_relevant_features", type=float, default=.05)
    parser.add_argument("-noise_multiplier", type=float, default=.05,
                        help="Multiplicative factor for noise added to polynomial computation for irrelevant features")
    parser.add_argument("-noise_type", choices=[constants.ADDITIVE_GAUSSIAN, constants.EPSILON_IRRELEVANT],
                        default=constants.EPSILON_IRRELEVANT)
    parser.add_argument("-hierarchy_type", help="Choice of hierarchy to generate", default=constants.CLUSTER_FROM_DATA,
                        choices=[constants.CLUSTER_FROM_DATA, constants.RANDOM])
    parser.add_argument("-clustering_instance_count", type=int, help="If provided, uses this number of instances to "
                        "cluster the data to generate a hierarchy, allowing the hierarchy to remain same across multiple "
                        "sets of instances", default=0)
    parser.add_argument("-num_interactions", type=int, default=0, help="number of interaction pairs")
    # Arguments passed to mihifepe
    parser.add_argument("-perturbation", default=constants.SHUFFLING, choices=[constants.ZEROING, constants.SHUFFLING])
    parser.add_argument("-num_shuffling_trials", type=int, default=100, help="Number of shuffling trials to average over, "
                        "when shuffling perturbations are selected")
    parser.add_argument("-condor", dest="condor", action="store_true",
                        help="Enable parallelization using condor (default disabled)")
    parser.add_argument("-no-condor", dest="condor", action="store_false", help="Disable parallelization using condor")
    parser.set_defaults(condor=False)
    parser.add_argument("-features_per_worker", type=int, default=10, help="worker load")
    parser.add_argument("-eviction_timeout", type=int, default=7200)
    parser.add_argument("-idle_timeout", type=int, default=7200)

    args = parser.parse_args()
    if not args.output_dir:
        args.output_dir = ("sim_outputs_inst_%d_feat_%d_noise_%.3f_relfraction_%.3f_pert_%s_shufftrials_%d" %
                           (args.num_instances, args.num_features, args.noise_multiplier,
                            args.fraction_relevant_features, args.perturbation, args.num_shuffling_trials))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    np.random.seed(args.seed)
    logging.basicConfig(level=logging.INFO, filename="%s/simulation.log" % args.output_dir,
                        format="%(asctime)s: %(message)s")
    logger = logging.getLogger(__name__)
    args.logger = logger

    pipeline(args)


def pipeline(args):
    """Simulation pipeline"""
    # TODO: Features other than binary
    args.logger.info("Begin mihifepe simulation with args: %s" % args)
    # Synthesize polynomial that generates ground truth
    sym_vars, relevant_feature_map, polynomial_fn = gen_polynomial(args)
    # Synthesize data
    probs, test_data, clustering_data = synthesize_data(args)
    # Generate hierarchy using clustering
    hierarchy_root = gen_hierarchy(args, clustering_data)
    # Update hierarchy descriptions for future visualization
    update_hierarchy_relevance(hierarchy_root, relevant_feature_map, probs)
    # Generate targets (ground truth)
    targets = gen_targets(polynomial_fn, test_data)
    # Write outputs - data, gen_model.py, hierarchy
    data_filename = write_data(args, test_data, targets)
    hierarchy_filename = write_hierarchy(args, hierarchy_root)
    gen_model_filename = write_model(args, sym_vars)
    # Invoke feature importance algorithm
    run_mihifepe(args, data_filename, hierarchy_filename, gen_model_filename)
    # Compare mihifepe outputs with ground truth outputs
    compare_results(args, hierarchy_root)
    # Evaluate mihifepe outputs - power/FDR for all nodes/outer nodes/base features
    results = evaluate(args.output_dir)
    args.logger.info("Results:\n%s" % str(results))
    args.logger.info("End mihifepe simulation")


def synthesize_data(args):
    """Synthesize data"""
    # TODO: Correlations between features
    args.logger.info("Begin generating data")
    probs = np.random.uniform(size=args.num_features)
    data = np.random.binomial(1, probs, size=(max(args.num_instances, args.clustering_instance_count), args.num_features))
    test_data = data
    clustering_data = data
    if args.clustering_instance_count:
        clustering_data = data[:args.clustering_instance_count, :]
        if args.clustering_instance_count > args.num_instances:
            test_data = data[:args.num_instances, :]
    args.logger.info("End generating data")
    return probs, test_data, clustering_data


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
    for idx, node in enumerate(anytree.PostOrderIter(hierarchy_root)):
        node.idx = idx
        if node.is_leaf:
            node.min_child_idx = idx
            node.max_child_idx = idx
            node.num_base_features = 1
            node.name = str(idx)
        else:
            node.min_child_idx = min([child.min_child_idx for child in node.children])
            node.max_child_idx = max([child.idx for child in node.children])
            node.num_base_features = sum([child.num_base_features for child in node.children])
            node.name = "[%d-%d] (size: %d)" % (node.min_child_idx, node.max_child_idx, node.num_base_features)
    return hierarchy_root


def gen_random_hierarchy(args):
    """Generates balanced random hierarchy"""
    args.logger.info("Begin generating hierarchy")
    nodes = [anytree.Node(str(idx), static_indices=str(idx)) for idx in range(args.num_features)]
    np.random.shuffle(nodes)
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
    nodes = [anytree.Node(str(idx), static_indices=str(idx)) for idx in range(args.num_features)]
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


def gen_polynomial(args):
    """Generate polynomial which decides the ground truth and noisy model"""
    # Note: using sympy to build function appears to be 1.5-2x slower than erstwhile raw numpy implementation (for linear terms)
    sym_features = sympy.symbols(["x%d" % x for x in range(args.num_features)])
    relevant_feature_map = {} # map of relevant feature tuples to coefficients
    num_relevant_features = max(1, round(args.num_features * args.fraction_relevant_features))
    # Generate polynomial expression
    # Order one terms
    sym_polynomial_fn = update_linear_terms(args, num_relevant_features, relevant_feature_map, sym_features)
    irrelevant_features = np.array([0 if (x,) in relevant_feature_map else 1 for x in range(args.num_features)])
    # Pairwise interactions
    sym_polynomial_fn = update_quadratic_terms(args, num_relevant_features, relevant_feature_map, sym_features, sym_polynomial_fn)
    # Generate model expression
    polynomial_fn = lambdify([sym_features], sym_polynomial_fn, "numpy")
    if args.noise_type == constants.EPSILON_IRRELEVANT:
        sym_noise = sympy.symbols(["noise%d" % x for x in range(args.num_features)])
        sym_model_fn = sym_polynomial_fn + (sym_noise * irrelevant_features).dot(sym_features)
    elif args.noise_type == constants.ADDITIVE_GAUSSIAN:
        sym_noise = sympy.symbols("noise")
        sym_model_fn = sym_polynomial_fn + sym_noise
    else:
        raise NotImplementedError("Unknown noise type")
    sym_vars = (sym_features, sym_noise, sym_model_fn)
    return sym_vars, relevant_feature_map, polynomial_fn


def update_linear_terms(args, num_relevant_features, relevant_feature_map, sym_features):
    """Order one terms for polynomial"""
    coefficients = np.zeros(args.num_features)
    coefficients[:num_relevant_features] = 1
    np.random.shuffle(coefficients)
    relevant_ids = [idx for idx in range(args.num_features) if coefficients[idx]]
    coefficients = np.multiply(coefficients, np.random.uniform(size=args.num_features))
    for relevant_id in relevant_ids:
        relevant_feature_map[(relevant_id,)] = coefficients[relevant_id]
    sym_polynomial_fn = coefficients.dot(sym_features)
    return sym_polynomial_fn


def update_quadratic_terms(args, num_relevant_features, relevant_feature_map, sym_features, sym_polynomial_fn):
    """Quadratic (pairwise interaction) terms for polynomial"""
    # Some interactions between individually relevant features
    # Some interactions between individually irrelevant features
    # Some pairwise interactions
    # Some higher-order interactions
    num_interactions = min(args.num_interactions, num_relevant_features * (num_relevant_features - 1) / 2)
    if not num_interactions:
        return sym_polynomial_fn
    relevant_ids = [x[0] for x in relevant_feature_map.keys()]
    potential_pairs = list(itertools.combinations(relevant_ids, 2))
    potential_pairs_arr = np.empty(len(potential_pairs), dtype=np.object)
    potential_pairs_arr[:] = potential_pairs
    interaction_pairs = np.random.choice(potential_pairs_arr, size=num_interactions, replace=False)
    for interaction_pair in interaction_pairs:
        coefficient = np.random.uniform()
        relevant_feature_map[tuple(interaction_pair)] = coefficient
        sym_polynomial_fn += coefficient * functools.reduce(lambda sym_x, y: sym_x * sym_features[y], interaction_pair, 1)
    return sym_polynomial_fn


def update_hierarchy_relevance(hierarchy_root, relevant_feature_map, probs):
    """
    Add feature relevance information to nodes of hierarchy:
    their probabilty of being enabled,
    their polynomial coefficient
    """
    for node in anytree.PostOrderIter(hierarchy_root):
        node.description = constants.IRRELEVANT
        if node.is_leaf:
            idx = int(node.static_indices)
            coeff = relevant_feature_map.get((idx,))
            if coeff:
                node.bin_prob = probs[idx]
                node.poly_coeff = coeff
                node.description = ("%s feature:\nPolynomial coefficient: %f\nBinomial probability: %f"
                                    % (constants.RELEVANT, coeff, probs[idx]))
        else:
            for child in node.children:
                if child.description != constants.IRRELEVANT:
                    node.description = constants.RELEVANT


def gen_targets(polynomial_fn, data):
    """Generate targets (ground truth) from polynomial"""
    return [polynomial_fn(instance) for instance in data]


def write_data(args, data, targets):
    """
    Write data in HDF5 format.

    Groups:     /records
                /records/<record_id>

    Datasets:   /records/<record_id>/temporal (2-D array)
                /records/<record_id>/static (1-D array)
                /records/<record_id>/target (scalar)
    """
    data_filename = "%s/%s" % (args.output_dir, "data.hdf5")
    root = h5py.File(data_filename, "w")
    record_ids = [str(idx).encode("utf8") for idx in range(args.num_instances)]
    root.create_dataset(constants.RECORD_IDS, data=record_ids)
    root.create_dataset(constants.TARGETS, data=targets)
    root.create_dataset(constants.STATIC, data=data)
    # Temporal data: not used here, but add fields for demonstration
    temporal = root.create_group(constants.TEMPORAL)
    for record_id in record_ids:
        temporal.create_dataset(record_id, data=[])
    root.close()
    return data_filename


def write_hierarchy(args, hierarchy_root):
    """
    Write hierarchy in CSV format.

    Columns:    *name*:             feature name, must be unique across features
                *parent_name*:      name of parent if it exists, else '' (root node)
                *description*:      node description
                *static_indices*:   [only required for leaf nodes] list of tab-separated indices corresponding to the indices
                                    of these features in the static data
                *temporal_indices*: [only required for leaf nodes] list of tab-separated indices corresponding to the indices
                                    of these features in the temporal data
    """
    hierarchy_filename = "%s/%s" % (args.output_dir, "hierarchy.csv")
    with open(hierarchy_filename, "w", newline="") as hierarchy_file:
        writer = csv.writer(hierarchy_file, delimiter=",")
        writer.writerow([constants.NODE_NAME, constants.PARENT_NAME,
                         constants.DESCRIPTION, constants.STATIC_INDICES, constants.TEMPORAL_INDICES])
        for node in anytree.PreOrderIter(hierarchy_root):
            static_indices = node.static_indices if node.is_leaf else ""
            parent_name = node.parent.name if node.parent else ""
            writer.writerow([node.name, parent_name, node.description, static_indices, ""])
    return hierarchy_filename


def write_model(args, sym_vars):
    """
    Write model to file in output directory.
    Write model_filename to config file in script directory.
    gen_model.py uses config file to load model.
    """
    # Write model to file
    model_filename = "%s/%s" % (args.output_dir, constants.MODEL_FILENAME)
    with open(model_filename, "wb") as model_file:
        pickle.dump(sym_vars, model_file)
    # Write model_filename to config
    gen_model_config_filename = "%s/%s" % (args.output_dir, constants.GEN_MODEL_CONFIG_FILENAME)
    with open(gen_model_config_filename, "wb") as gen_model_config_file:
        pickle.dump(model_filename, gen_model_config_file)
        pickle.dump(args.noise_multiplier, gen_model_config_file)
        pickle.dump(args.noise_type, gen_model_config_file)
    # Write gen_model.py to output_dir
    gen_model_filename = "%s/%s" % (args.output_dir, constants.GEN_MODEL_FILENAME)
    gen_model_template_filename = "%s/%s" % (os.path.dirname(os.path.abspath(__file__)), constants.GEN_MODEL_TEMPLATE_FILENAME)
    gen_model_file = open(gen_model_filename, "w")
    with open(gen_model_template_filename, "r") as gen_model_template_file:
        for line in gen_model_template_file:
            line = line.replace(constants.GEN_MODEL_CONFIG_FILENAME_PLACEHOLDER, gen_model_config_filename)
            gen_model_file.write(line)
    gen_model_file.close()
    return gen_model_filename


def run_mihifepe(args, data_filename, hierarchy_filename, gen_model_filename):
    """Run mihifepe algorithm"""
    args.logger.info("Begin running mihifepe")
    condor_val = "-condor" if args.condor else "-no-condor"
    # Compute approximate memory requirement in GB
    memory_requirement = 1 + (os.stat(data_filename).st_size // (2 ** 30))
    cmd = ("python -m mihifepe.master -data_filename '%s' -hierarchy_filename '%s' -model_generator_filename '%s' -output_dir '%s' "
           "-perturbation %s -num_shuffling_trials %d %s -features_per_worker %d -memory_requirement %d "
           "-eviction_timeout %d -idle_timeout %d"
           % (data_filename, hierarchy_filename, gen_model_filename, args.output_dir,
              args.perturbation, args.num_shuffling_trials, condor_val, args.features_per_worker, memory_requirement,
              args.eviction_timeout, args.idle_timeout))
    args.logger.info("Running cmd: %s" % cmd)
    subprocess.check_call(cmd, shell=True)
    args.logger.info("End running mihifepe")


def compare_results(args, hierarchy_root):
    """Compare results from mihifepe with ground truth results"""
    # Generate ground truth results
    # Write hierarchical FDR input file for ground truth values
    args.logger.info("Compare mihifepe results to ground truth")
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
                    node.pvalue = min(0.001, 1e-10 / (node.poly_coeff * node.bin_prob) ** 3)
                else:
                    node.pvalue = 0.999 * min([child.pvalue for child in node.children])
            writer.writerow([node.name, parent_name, node.pvalue, node.description])
    # Generate hierarchical FDR results for ground truth values
    ground_truth_dir = "%s/ground_truth_fdr" % args.output_dir
    cmd = ("python -m mihifepe.fdr.hierarchical_fdr_control -output_dir %s -procedure yekutieli "
           "-rectangle_leaves %s" % (ground_truth_dir, input_filename))
    args.logger.info("Running cmd: %s" % cmd)
    subprocess.check_call(cmd, shell=True)
    # Compare results
    ground_truth_outputs_filename = "%s/%s.png" % (ground_truth_dir, constants.TREE)
    args.logger.info("Ground truth results: %s" % ground_truth_outputs_filename)
    mihifepe_outputs_filename = "%s/%s/%s.png" % (args.output_dir, constants.HIERARCHICAL_FDR_DIR, constants.TREE)
    args.logger.info("mihifepe results: %s" % mihifepe_outputs_filename)


def evaluate(output_dir):
    """
    Evaluate mihifepe results - obtain power/FDR measures for all nodes/outer nodes/base features
    """
    # pylint: disable = too-many-locals
    def get_relevant_rejected(nodes, outer=False, leaves=False):
        """Get set of relevant and rejected nodes"""
        assert not (outer and leaves)
        if outer:
            nodes = [node for node in nodes if node.rejected and all([not child.rejected for child in node.children])]
        elif leaves:
            nodes = [node for node in nodes if node.is_leaf]
        relevant = [0 if node.description == constants.IRRELEVANT else 1 for node in nodes]
        rejected = [1 if node.rejected else 0 for node in nodes]
        return relevant, rejected

    tree_filename = "%s/%s/%s.json" % (output_dir, constants.HIERARCHICAL_FDR_DIR, constants.HIERARCHICAL_FDR_OUTPUTS)
    with open(tree_filename, "r") as tree_file:
        tree = JsonImporter().read(tree_file)
        nodes = list(anytree.PreOrderIter(tree))
        # All nodes FDR/power
        relevant, rejected = get_relevant_rejected(nodes)
        precision, recall, _, _ = precision_recall_fscore_support(relevant, rejected, average="binary")
        # Outer nodes FDR/power
        outer_relevant, outer_rejected = get_relevant_rejected(nodes, outer=True)
        outer_precision, outer_recall, _, _ = precision_recall_fscore_support(outer_relevant, outer_rejected, average="binary")
        # Base features FDR/power
        bf_relevant, bf_rejected = get_relevant_rejected(nodes, leaves=True)
        bf_precision, bf_recall, _, _ = precision_recall_fscore_support(bf_relevant, bf_rejected, average="binary")
        return Results(1 - precision, recall, 1 - outer_precision, outer_recall, 1 - bf_precision, bf_recall)


if __name__ == "__main__":
    main()
