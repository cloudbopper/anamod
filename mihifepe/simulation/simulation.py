"""Generates simulated data and model to test mihifepe algorithm"""

import argparse
import csv
import logging
import os
import pickle
import subprocess

import anytree
import h5py
import numpy as np
from scipy.cluster.hierarchy import linkage

from .. import constants

# TODO maybe: write arguments to separate readme.txt for documentating runs

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
    # Arguments passed to mihifepe
    parser.add_argument("-perturbation", default=constants.SHUFFLING, choices=[constants.ZEROING, constants.SHUFFLING])
    parser.add_argument("-num_shuffling_trials", type=int, default=100, help="Number of shuffling trials to average over, "
                        "when shuffling perturbations are selected")
    parser.add_argument("-condor", dest="condor", action="store_true",
                        help="Enable parallelization using condor (default disabled)")
    parser.add_argument("-no-condor", dest="condor", action="store_false", help="Disable parallelization using condor")
    parser.set_defaults(condor=False)
    parser.add_argument("-features_per_worker", type=int, default=1, help="worker load")

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
    args.logger.info("Begin mihifepe simulation")
    # Synthesize polynomial that generates ground truth
    relevant_features, poly_coeff = gen_polynomial(args)
    # Synthesize data
    probs, data = synthesize_data(args)
    # Generate hierarchy using clustering
    # TODO: Generate other hierarchies e.g. random
    clusters = cluster_data(args, data)
    hierarchy_root = gen_hierarchy(args, clusters)
    # Update hierarchy descriptions for future visualization
    update_hierarchy_relevance(hierarchy_root, relevant_features, poly_coeff, probs)
    # Generate targets (ground truth)
    targets = gen_targets(poly_coeff, data)
    # Write outputs - data, gen_model.py, hierarchy
    data_filename, hierarchy_filename, gen_model_filename = write_outputs(args, data, hierarchy_root, targets, poly_coeff)
    # Invoke feature importance algorithm
    run_mihifepe(args, data_filename, hierarchy_filename, gen_model_filename)
    # Compare mihifepe outputs with ground truth outputs
    compare_results(args, hierarchy_root)
    args.logger.info("End mihifepe simulation")


def synthesize_data(args):
    """Synthesize data"""
    # TODO: Correlations between features
    args.logger.info("Begin generating data")
    probs = np.random.uniform(size=args.num_features)
    data = np.random.binomial(1, probs, size=(args.num_instances, args.num_features))
    # TODO: train/test split?
    # train, test = data[:int(args.num_instances * .8), :], data[int(args.num_instances * .8):, :]
    args.logger.info("End generating data")
    return probs, data


def cluster_data(args, data):
    """Cluster data using hierarchical clustering with Hamming distance"""
    # Cluster data
    args.logger.info("Begin clustering data")
    clusters = linkage(data.transpose(), metric="hamming")
    args.logger.info("End clustering data")
    return clusters


def gen_hierarchy(args, clusters):
    """
    Organize clusters into hierarchy

    Args:
        clusters: linkage matrix (num_features-1 X 4)
                  rows indicate successive clustering iterations
                  columns, respectively: 1st cluster index, 2nd cluster index, distance, sample count
    Returns:
        hierarchy_root: root of resulting hierarchy over features
    """
    args.logger.info("Begin generating hierarchy")
    nodes = [anytree.Node(str(idx), static_indices=str(idx)) for idx in range(args.num_features)]
    for idx, cluster in enumerate(clusters):
        left_idx, right_idx, _, _ = cluster
        left_idx = int(left_idx)
        right_idx = int(right_idx)
        cluster_node = anytree.Node("%d+%d" % (left_idx, right_idx))
        nodes[left_idx].parent = cluster_node
        nodes[right_idx].parent = cluster_node
        nodes.append(cluster_node)
    hierarchy_root = nodes[-1]
    args.logger.info("End generating hierarchy")
    return hierarchy_root


def gen_polynomial(args):
    """Generate polynomial which decides the ground truth"""
    # Decide relevant features
    relevant_features = np.random.binomial(1, [args.fraction_relevant_features] * args.num_features)
    # Generate coefficients
    # TODO: higher powers, interaction terms, negative coefficients
    coefficients = np.multiply(relevant_features, np.random.uniform(size=args.num_features))
    return relevant_features, coefficients


def update_hierarchy_relevance(hierarchy_root, relevant_features, poly_coeff, probs):
    """
    Add feature relevance information to nodes of hierarchy:
    their probabilty of being enabled,
    their polynomial coefficient
    """
    for node in anytree.PostOrderIter(hierarchy_root):
        node.description = constants.IRRELEVANT
        if node.is_leaf:
            idx = int(node.name)
            if relevant_features[idx]:
                node.bin_prob = probs[idx]
                node.poly_coeff = poly_coeff[idx]
                node.description = ("%s feature:\nPolynomial coefficient: %f\nBinomial probability: %f"
                                    % (constants.RELEVANT, poly_coeff[idx], probs[idx]))
        else:
            for child in node.children:
                if child.description != constants.IRRELEVANT:
                    node.description = constants.RELEVANT


def gen_targets(poly_coeff, data):
    """Generate targets (ground truth) from polynomial"""
    return np.dot(data, poly_coeff)


def write_outputs(args, data, hierarchy_root, targets, model):
    """Write outputs to files"""
    def write_data():
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

    def write_hierarchy():
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
        with open(hierarchy_filename, "w") as hierarchy_file:
            writer = csv.writer(hierarchy_file, delimiter=",")
            writer.writerow([constants.NODE_NAME, constants.PARENT_NAME,
                             constants.DESCRIPTION, constants.STATIC_INDICES, constants.TEMPORAL_INDICES])
            for node in anytree.PreOrderIter(hierarchy_root):
                static_indices = node.static_indices if node.is_leaf else ""
                parent_name = node.parent.name if node.parent else ""
                writer.writerow([node.name, parent_name, node.description, static_indices, ""])
        return hierarchy_filename

    def write_model():
        """
        Write model to file in output directory.
        Write model_filename to config file in script directory.
        gen_model.py uses config file to load model.
        """
        # Write model to file
        model_filename = "%s/%s" % (args.output_dir, constants.MODEL_FILENAME)
        np.save(model_filename, model)
        # Write model_filename to config
        gen_model_config_filename = "%s/%s" % (os.path.dirname(os.path.abspath(__file__)), constants.GEN_MODEL_CONFIG_FILENAME)
        with open(gen_model_config_filename, "wb") as gen_model_config_file:
            pickle.dump(model_filename, gen_model_config_file)
            pickle.dump(args.noise_multiplier, gen_model_config_file)
        gen_model_filename = "%s/%s" % (os.path.dirname(os.path.abspath(__file__)), constants.GEN_MODEL_FILENAME)
        return gen_model_filename

    args.logger.info("Begin writing simulation files")
    data_filename = write_data()
    hierarchy_filename = write_hierarchy()
    gen_model_filename = write_model()
    args.logger.info("End writing simulation files")
    return data_filename, hierarchy_filename, gen_model_filename


def run_mihifepe(args, data_filename, hierarchy_filename, gen_model_filename):
    """Run mihifepe algorithm"""
    args.logger.info("Begin running mihifepe")
    condor_val = "-condor" if args.condor else "-no-condor"
    # Compute approximate memory requirement in GB
    memory_requirement = 1 + (os.stat(data_filename).st_size // (2 ** 30))
    cmd = ("python -m mihifepe.master -data_filename '%s' -hierarchy_filename '%s' -model_generator_filename '%s' -output_dir '%s' "
           "-perturbation %s -num_shuffling_trials %d %s -features_per_worker %d -memory_requirement %d"
           % (data_filename, hierarchy_filename, gen_model_filename, args.output_dir,
              args.perturbation, args.num_shuffling_trials, condor_val, args.features_per_worker, memory_requirement))
    args.logger.info("Running cmd: %s" % cmd)
    subprocess.check_call(cmd, shell=True)
    args.logger.info("End running mihifepe")


def compare_results(args, hierarchy_root):
    """Compare results from mihifepe with ground truth results"""
    # Generate ground truth results
    # Write hierarchical FDR input file for ground truth values
    args.logger.info("Compare mihifepe results to ground truth")
    input_filename = "%s/ground_truth_pvalues.csv" % args.output_dir
    with open(input_filename, "w") as input_file:
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


if __name__ == "__main__":
    main()
