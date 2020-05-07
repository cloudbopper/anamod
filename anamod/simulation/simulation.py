"""Generates simulated data and model to test anamod algorithm"""

import argparse
import copy
from distutils.util import strtobool
import json
import os
import pickle
import pprint
import shutil

import anytree
import cloudpickle
import numpy as np
from scipy.cluster.hierarchy import linkage
import synmod.master
from synmod.constants import CLASSIFIER, REGRESSOR, FEATURES_FILENAME, MODEL_FILENAME, INSTANCES_FILENAME

from anamod import constants, utils, ModelAnalyzer
from anamod.simulation.model_wrapper import ModelWrapper
from anamod.simulation import evaluation
from anamod.utils import CondorJobWrapper


def main():
    """Main"""
    parser = argparse.ArgumentParser("python anamod.simulation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    common.add_argument("-condor_cleanup", type=strtobool, default=True, help="Clean condor cmd/out/err/log files after completing simulation")
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
    synthesized_features, data, model = run_synmod(args)
    targets = model.predict(data, labels=True) if args.model_type == CLASSIFIER else model.predict(data)
    # Create wrapper around ground-truth model
    model_wrapper = ModelWrapper(model, args.num_features, args.noise_type, args.noise_multiplier, args.seed)
    if args.analysis_type == constants.HIERARCHICAL:
        # Generate hierarchy using clustering (test data also used for clustering)
        hierarchy_root, feature_id_map = gen_hierarchy(args, data)
        # Update hierarchy descriptions for future visualization
        update_hierarchy_relevance(hierarchy_root, model.relevant_feature_map, synthesized_features)
        # Invoke feature importance algorithm
        analyzed_features = run_anamod(args, pass_args, model_wrapper, data, targets, hierarchy_root)
        # Compare anamod outputs with ground truth outputs
        evaluation.compare_with_ground_truth(args, hierarchy_root)
        # Evaluate anamod outputs - power/FDR for all nodes/outer nodes/base features
        results = evaluation.evaluate_hierarchical(args, model.relevant_feature_map, feature_id_map)
    else:
        # Temporal model analysis
        # FIXME: should have similar mode of parsing outputs for both analyses
        analyzed_features = run_anamod(args, pass_args, model_wrapper, data, targets)
        results = evaluation.evaluate_temporal(args, model, analyzed_features)
    summary = write_summary(args, model, results)
    write_io(args, model, synthesized_features, analyzed_features)
    args.logger.info("End anamod simulation")
    return summary


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
    CondorJobWrapper.monitor([job], cleanup=args.condor_cleanup, logger=args.logger)
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
    # TODO: Get rid of possibly redundant hierarchy attributes e.g. vidx
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
                feature_id_map[idx] = node.idx[0]
            else:
                node.min_child_vidx = min([child.min_child_vidx for child in node.children])
                node.max_child_vidx = max([child.vidx for child in node.children])
                node.num_base_features = sum([child.num_base_features for child in node.children])
                node.name = "[%d-%d] (size: %d)" % (node.min_child_vidx, node.max_child_vidx, node.num_base_features)
    return hierarchy_root, feature_id_map


def gen_random_hierarchy(args):
    """Generates balanced random hierarchy"""
    args.logger.info("Begin generating hierarchy")
    nodes = [anytree.Node(str(idx), idx=[idx]) for idx in range(args.num_features)]
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
    nodes = [anytree.Node(str(idx), idx=[idx]) for idx in range(args.num_features)]
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
            idx = node.idx[0]
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


def run_anamod(args, pass_args, model, data, targets, hierarchy=None):  # pylint: disable = too-many-arguments
    """Run analysis algorithms"""
    args.logger.info("Begin running anamod")
    # Add options
    options = {}
    options["feature_hierarchy"] = hierarchy
    options["output_dir"] = args.output_dir
    options["seed"] = args.seed
    options["memory_requirement"] = 1 + (data.nbytes // (2 ** 30))
    options["disk_requirement"] = 3 + options["memory_requirement"]
    options["analysis_type"] = args.analysis_type
    options["condor"] = args.condor
    options["shared_filesystem"] = args.shared_filesystem
    options["num_shuffling_trials"] = args.num_shuffling_trials
    options["cleanup"] = args.cleanup
    if args.analysis_type == constants.HIERARCHICAL:
        options["perturbation"] = args.perturbation
        options["analyze_interactions"] = args.analyze_interactions
    args.logger.info("Passing the following arguments to anamod.master without parsing: %s" % pass_args)
    pass_args = process_pass_args(pass_args)
    options = {**pass_args, **options}  # Merge dictionaries
    # Create analyzer
    analyzer = ModelAnalyzer(model, data, targets, **options)
    # Run analyzer
    args.logger.info(f"Analyzing model with options: {pprint.pformat(options)}")
    features = analyzer.analyze()
    cleanup(args, analyzer.data_filename, analyzer.model_filename)
    args.logger.info("End running anamod")
    return features


def process_pass_args(pass_args):
    """Process list of unrecognized arguments, to pass to anamod.master"""
    assert len(pass_args) % 2 == 0, f"Odd argument count in pass_args: {pass_args} ; is a value missing?"
    pass_args = {pass_args[idx].strip("-"): pass_args[idx + 1] for idx in range(0, len(pass_args), 2)}  # Make dict
    return pass_args


def cleanup(args, data_filename, model_filename):
    """Clean data and model files after completing simulation"""
    # TODO: clean up hierarchy file
    if not args.cleanup:
        return
    for filename in [data_filename, model_filename]:
        if filename is not None and os.path.exists(filename):
            os.remove(filename)


def write_summary(args, model, results):
    """Write simulation summary"""
    config = dict(analysis_type=args.analysis_type,
                  num_instances=args.num_instances,
                  num_features=args.num_features,
                  sequence_length=args.sequence_length,
                  model_type=model.__class__.__name__,
                  num_shuffling_trials=args.num_shuffling_trials,
                  sequences_independent_of_windows=args.window_independent)
    # pylint: disable = protected-access
    model_summary = dict(operation=model._aggregator.__class__.__name__,
                         polynomial=model.sym_polynomial_fn.__repr__())
    summary = {constants.CONFIG: config, constants.MODEL: model_summary, constants.RESULTS: results}
    summary_filename = f"{args.output_dir}/{constants.SIMULATION_SUMMARY_FILENAME}"
    args.logger.info(f"Writing summary to {summary_filename}")
    with open(summary_filename, "w") as summary_file:
        json.dump(summary, summary_file, indent=2)
    return summary


def write_io(args, model, synthesized_features, analyzed_features, ):
    """Write simulation inputs and outputs (model and features)"""
    with open(f"{args.output_dir}/{constants.MODEL_FILENAME}", "wb") as model_file:
        cloudpickle.dump(model, model_file, protocol=pickle.DEFAULT_PROTOCOL)
    with open(f"{args.output_dir}/{constants.SYNTHESIZED_FEATURES_FILENAME}", "wb") as synthesized_features_file:
        cloudpickle.dump(synthesized_features, synthesized_features_file, protocol=pickle.DEFAULT_PROTOCOL)
    with open(f"{args.output_dir}/{constants.ANALYZED_FEATURES_FILENAME}", "wb") as analyzed_features_file:
        cloudpickle.dump(analyzed_features, analyzed_features_file, protocol=pickle.DEFAULT_PROTOCOL)


if __name__ == "__main__":
    main()
