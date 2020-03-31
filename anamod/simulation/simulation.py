"""Generates simulated data and model to test anamod algorithm"""

import argparse
import copy
import csv
from distutils.util import strtobool
import os
import shutil
import sys
from unittest.mock import patch

import anytree
import cloudpickle
import h5py
import numpy as np
from scipy.cluster.hierarchy import linkage
import synmod.master
from synmod.constants import CLASSIFIER, REGRESSOR, FEATURES_FILENAME, MODEL_FILENAME, INSTANCES_FILENAME

from anamod import constants, master, utils
from anamod.simulation.model_wrapper import ModelWrapper
from anamod.simulation import evaluation
from anamod.utils import CondorJobWrapper

# TODO maybe: write arguments to separate readme.txt for documenting runs


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
    common.add_argument("-include_interaction_only_features", help="include interaction-only features in model"
                        " in addition to linear + interaction features (default enabled)", type=strtobool, default=True)
    common.add_argument("-condor", help="Use condor for parallelization", type=strtobool, default=False)
    common.add_argument("-shared_filesystem", type=strtobool, default=False)
    common.add_argument("-cleanup", type=strtobool, default=True, help="Clean data and model files after completing simulation")
    # Hierarchical feature importance analysis arguments
    hierarchical = parser.add_argument_group("Hierarchical feature analysis arguments")
    hierarchical.add_argument("-noise_multiplier", type=float, default=.05,
                              help="Multiplicative factor for noise added to polynomial computation for irrelevant features")
    hierarchical.add_argument("-noise_type", choices=[constants.ADDITIVE_GAUSSIAN, constants.EPSILON_IRRELEVANT, constants.NO_NOISE],
                              default=constants.EPSILON_IRRELEVANT)
    hierarchical.add_argument("-hierarchy_type", help="Choice of hierarchy to generate", default=constants.CLUSTER_FROM_DATA,
                              choices=[constants.CLUSTER_FROM_DATA, constants.RANDOM])
    hierarchical.add_argument("-contiguous_node_names", type=strtobool, default=False, help="enable to change node names in hierarchy "
                              "to be contiguous for better visualization (but creating mismatch between node names and features indices)")
    hierarchical.add_argument("-analyze_interactions", help="enable analyzing interactions", type=strtobool, default=False)
    hierarchical.add_argument("-perturbation", default=constants.SHUFFLING, choices=[constants.ZEROING, constants.SHUFFLING])
    hierarchical.add_argument("-num_shuffling_trials", type=int, default=100, help="Number of shuffling trials to average over, "
                              "when shuffling perturbations are selected")
    # Temporal model analysis arguments
    temporal = parser.add_argument_group("Temporal model analysis arguments")
    temporal.add_argument("-sequence_length", help="sequence length for temporal models", type=int, default=20)
    temporal.add_argument("-model_type", default=REGRESSOR, choices=[CLASSIFIER, REGRESSOR])
    temporal.add_argument("-sequences_independent_of_windows", type=strtobool, dest="window_independent")
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
    targets = model.predict(data, labels=True) if args.model_type == CLASSIFIER else model.predict(data)
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
        evaluation.compare_with_ground_truth(args, hierarchy_root)
        # Evaluate anamod outputs - power/FDR for all nodes/outer nodes/base features
        results = evaluation.evaluate_hierarchical(args, model.relevant_feature_map, feature_id_map)
    else:
        # Temporal model analysis
        # FIXME: should have similar mode of parsing outputs for both analyses
        analyzed_features = run_anamod(args, pass_args, data_filename, model_filename)
        results = evaluation.evaluate_temporal(args, model, analyzed_features)
    args.logger.info("Results:\n%s" % str(results))
    results.write(args.output_dir)
    cleanup(args, data_filename, model_filename)
    args.logger.info("End anamod simulation")
    return results


def run_synmod(args):
    """Synthesize data and model"""
    args.logger.info("Begin running synmod")
    args = copy.copy(args)
    if args.analysis_type == constants.HIERARCHICAL:
        args.synthesis_type = constants.STATIC
    else:
        args.synthesis_type = constants.TEMPORAL
    if not args.condor:
        args.write_outputs = False
        return synmod.master.pipeline(args)
    # Spawn condor job to synthesize data
    # Compute size requirements
    data_size = args.num_instances * args.num_features * args.sequence_length // (8 * (2 ** 30))  # Data size in GB
    memory_requirement = "%dGB" % (1 + data_size)
    disk_requirement = "%dGB" % (4 + data_size)
    # Set up command-line arguments
    args.write_outputs = True
    args.sequences_independent_of_windows = args.window_independent
    cmd = "python3 -m synmod"
    job_dir = f"{args.output_dir}/synthesis"
    args.output_dir = os.path.abspath(job_dir) if args.shared_filesystem else os.path.basename(job_dir)
    for arg in ["output_dir", "num_features", "num_instances", "synthesis_type",
                "fraction_relevant_features", "num_interactions", "include_interaction_only_features", "seed", "write_outputs",
                "sequence_length", "sequences_independent_of_windows", "model_type"]:
        cmd += f" -{arg} {args.__getattribute__(arg)}"
    args.logger.info(f"Running cmd: {cmd}")
    # Launch and monitor job
    job = CondorJobWrapper(cmd, [], job_dir, shared_filesystem=args.shared_filesystem, memory=memory_requirement, disk=disk_requirement)
    job.run()
    CondorJobWrapper.monitor([job], cleanup=args.cleanup)
    # Extract data
    features, instances, model = [None] * 3
    with open(f"{job_dir}/{FEATURES_FILENAME}", "rb") as data_file:
        features = cloudpickle.load(data_file)
    instances = np.load(f"{job_dir}/{INSTANCES_FILENAME}")
    with open(f"{job_dir}/{MODEL_FILENAME}", "rb") as model_file:
        model = cloudpickle.load(model_file)
    if args.cleanup:
        shutil.rmtree(job_dir)
    args.logger.info("End running synmod")
    return features, instances, model


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
    hierarchical_analysis_options = ("-hierarchy_filename {0} -perturbation {1} -analyze_interactions {2}".format
                                     (hierarchy_filename, args.perturbation, args.analyze_interactions))
    temporal_analysis_options = ""
    analysis_options = hierarchical_analysis_options if args.analysis_type == constants.HIERARCHICAL else temporal_analysis_options
    args.logger.info("Passing the following arguments to anamod.master without parsing: %s" % pass_args)
    memory_requirement = 1 + (os.stat(data_filename).st_size // (2 ** 30))  # Compute approximate memory requirement in GB
    disk_requirement = 3 + memory_requirement
    cmd = ("python -m anamod.master -analysis_type {} -output_dir {} -condor {} -num_shuffling_trials {} -data_filename {} "
           "-model_filename {} -shared_filesystem {} -memory_requirement {} -disk_requirement {} -cleanup {} {} {}"
           .format(args.analysis_type,
                   args.output_dir,
                   args.condor,
                   args.num_shuffling_trials,
                   data_filename,
                   model_filename,
                   args.shared_filesystem,
                   memory_requirement,
                   disk_requirement,
                   args.cleanup,
                   analysis_options,
                   pass_args))
    args.logger.info("Running cmd: %s" % cmd)
    nargs = cmd.split()[2:]
    with patch.object(sys, 'argv', nargs):
        features = master.main()
    args.logger.info("End running anamod")
    return features


def cleanup(args, data_filename, model_filename):
    """Clean data and model files after completing simulation"""
    if not args.cleanup:
        return
    os.remove(data_filename)
    os.remove(model_filename)


if __name__ == "__main__":
    main()
