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

from . import constants

# TODO maybe: write arguments to separate readme.txt for documentating runs

def main():
    """Main"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", type=int, default=constants.SEED)
    parser.add_argument("-num_instances", type=int, default=10000)
    parser.add_argument("-num_features", type=int, default=500)
    parser.add_argument("-output_dir", required=True)
    parser.add_argument("-percent_relevant_features", type=float, default=5)

    args = parser.parse_args()
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
    # Synthesize data
    data = synthesize_data(args)
    # Generate hierarchy using clustering
    clusters = cluster_data(args, data)
    hierarchy_root = gen_hierarchy(args, clusters)
    # Synthesize polynomial that generates ground truth
    relevant_features, poly_coeff = gen_polynomial(args)
    # Update hierarchy to contain relevance information
    update_hierarchy_relevance(hierarchy_root, relevant_features)
    # Generate targets (ground truth)
    targets = gen_targets(poly_coeff, data)
    # Synthesize model (perturbed version of polynomial)
    model = gen_model(args, poly_coeff)
    # Write outputs - data, gen_model.py, hierarchy
    data_filename, hierarchy_filename, gen_model_filename = write_outputs(args, data, hierarchy_root, targets, model)
    # Invoke feature importance algorithm
    run_mihifepe(args, data_filename, hierarchy_filename, gen_model_filename)
    args.logger.info("End mihifepe simulation")


def synthesize_data(args):
    """Synthesize data"""
    # TODO: Correlations between features
    args.logger.info("Begin generating data")
    probs = np.random.uniform(size=args.num_features)
    data = np.random.binomial(1, probs, size=(args.num_instances, args.num_features))
    # TODO: train/test split
    # train, test = data[:int(args.num_instances * .8), :], data[int(args.num_instances * .8):, :]
    args.logger.info("End generating data")
    return data


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
    relevant_features = np.random.binomial(1, [args.percent_relevant_features] * args.num_features)
    # Generate coefficients
    # TODO: higher powers, interaction terms
    coefficients = np.multiply(relevant_features, np.random.uniform(size=args.num_features))
    return relevant_features, coefficients


def update_hierarchy_relevance(hierarchy_root, relevant_features):
    """
    Tag nodes of hierarchy as relevant or not
    based on whether they contain any relevant feature
    """
    for node in anytree.PostOrderIter(hierarchy_root):
        if node.is_leaf:
            idx = int(node.name)
            if relevant_features[idx]:
                node.category = constants.RELEVANT
            else:
                node.category = constants.IRRELEVANT
        else:
            node.category = constants.IRRELEVANT
            for child in node.children:
                if child.category == constants.RELEVANT:
                    node.category = constants.RELEVANT


def gen_targets(poly_coeff, data):
    """Generate targets (ground truth) from polynomial"""
    return np.dot(data, poly_coeff)


def gen_model(args, poly_coeff):
    """Generate noisy version of polynomial, which acts as the model"""
    return poly_coeff # TODO: generate noisy version


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
        records = root.create_group(constants.RECORDS)
        record_ids = [records.create_group(idx) for idx in range(args.num_instances)]
        for idx, record_id in enumerate(record_ids):
            record_id.create_dataset(constants.TEMPORAL, data=[])
            record_id.create_dataset(constants.STATIC, data=data[idx])
            record_id.create_dataset(constants.TARGET, data=targets[idx])
        root.close()
        return data_filename

    def write_hierarchy():
        """
        Write hierarchy in CSV format.

        Columns:    *name*:             feature name
                    *category*:         feature category - acts a namespace. The pair (category, name) must be unique for each feature.
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
            writer.writerow([constants.NODE_NAME, constants.CATEGORY, constants.PARENT_NAME,
                             constants.DESCRIPTION, constants.STATIC_INDICES, constants.TEMPORAL_INDICES])
            for node in anytree.PreOrderIter(hierarchy_root):
                parent_name = node.parent.name if node.parent else ""
                writer.writerow([node.name, node.category, parent_name,
                                 "", node.static_indices, ""])
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
        return constants.GEN_MODEL_FILENAME

    args.logger.info("Begin writing simulation files")
    data_filename = write_data()
    hierarchy_filename = write_hierarchy()
    gen_model_filename = write_model()
    args.logger.info("End writing simulation files")
    return data_filename, hierarchy_filename, gen_model_filename


def run_mihifepe(args, data_filename, hierarchy_filename, gen_model_filename):
    """Run mihifepe algorithm"""
    args.logger.info("Begin running mihifepe")
    cmd = ("python -m mihifepe.master -data_filename '%s' -hierarchy_filename '%s' -model_filename '%s' -output_dir '%s'" %
           (data_filename, hierarchy_filename, gen_model_filename, args.output_dir))
    output = subprocess.check_output(cmd, shell=True)
    args.logger.info("mihifepe stdout: %s" % output)
    args.logger.info("End running mihifepe")


if __name__ == "__main__":
    main()
