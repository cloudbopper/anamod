"""Run a bunch of simulations (requires condor)"""

import argparse
from collections import namedtuple
import csv
import logging
import os
import subprocess

import anytree
from anytree.importer import JsonImporter
from sklearn.metrics import precision_recall_fscore_support

from mihifepe import constants
# TODO: change all import paths to absolute

INSTANCE_COUNTS = "instance_counts"
NOISE_LEVELS = "noise_levels"
FEATURE_COUNTS = "feature_counts"
SHUFFLING_COUNTS = "shuffling_counts"
OUTPUTS = "%s/%s_%s"
PRECISION = "Precision"
RECALL = "Recall"
FSCORE = "F-score"

Simulation = namedtuple("Simulation", ["cmd", "output_dir", "param"])

def main():
    """Main"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-generate_results", action="store_true")
    parser.add_argument("-analyze_results", action="store_true")
    parser.add_argument("-type", choices=[INSTANCE_COUNTS, FEATURE_COUNTS, NOISE_LEVELS, SHUFFLING_COUNTS],
                        default=INSTANCE_COUNTS)
    parser.add_argument("-perturbation", choices=[constants.ZEROING, constants.SHUFFLING], default=constants.SHUFFLING)
    parser.add_argument("-output_dir", required=True)

    args = parser.parse_args()
    assert args.generate_results or args.analyze_results, "At least one of -generate_results, -analyze_results must be enabled"
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logging.basicConfig(level=logging.INFO, filename="%s/run_simulations.log" % args.output_dir,
                        format="%(asctime)s: %(message)s")
    logger = logging.getLogger(__name__)
    args.logger = logger

    args.logger.info("Begin running/analyzing simulations")
    simulations = parametrize_simulations(args)
    if args.generate_results:
        run_simulations(args, simulations)
    if args.analyze_results:
        analyze_simulations(args, simulations)
    args.logger.info("End running simulations")


def parametrize_simulations(args):
    """Parametrize simulations"""
    if args.type == INSTANCE_COUNTS:
        return instance_count_sims(args)
    if args.type == FEATURE_COUNTS:
        return feature_count_sims(args)
    if args.type == NOISE_LEVELS:
        return noise_level_sims(args)
    if args.type == SHUFFLING_COUNTS:
        return shuffling_count_sims(args)
    raise NotImplementedError("Unknown simulation type")


def instance_count_sims(args):
    """Configure simulations for different values of instance counts"""
    sims = []
    seed = 9184
    instance_counts = [16 * 2 ** x for x in range(11)]
    for instance_count in instance_counts:
        output_dir = OUTPUTS % (args.output_dir, INSTANCE_COUNTS, str(instance_count))
        cmd = ("python -m mihifepe.simulation.simulation -num_features 500 -fraction_relevant_features 0.1 -noise_multiplier 0.01 "
               "-clustering_instance_count %d -perturbation %s -num_shuffling_trials 500 -condor "
               "-num_instances %d -seed %d -output_dir %s" % (instance_counts[-1], args.perturbation, instance_count, seed, output_dir))
        sims.append(Simulation(cmd, output_dir, instance_count))
    return sims


def feature_count_sims(args):
    """Run simulations for different values of feature counts"""
    sims = []
    seed = 7185
    feature_counts = [8 * 2 ** x for x in range(8)]
    for feature_count in feature_counts:
        output_dir = OUTPUTS % (args.output_dir, FEATURE_COUNTS, str(feature_count))
        cmd = ("python -m mihifepe.simulation.simulation -num_instances 10000 -fraction_relevant_features 0.1 "
               "-noise_multiplier 0.01 -perturbation %s -num_shuffling_trials 500 -condor "
               "-num_features %d -seed %d -output_dir %s" % (args.perturbation, feature_count, seed, output_dir))
        sims.append(Simulation(cmd, output_dir, feature_count))
    return sims


def noise_level_sims(args):
    """Run simulations for different values of noise"""
    sims = []
    seed = 85100
    noise_levels = [0.0] + [0.0001 * 2 ** x for x in range(13)]
    for noise_level in noise_levels:
        output_dir = OUTPUTS % (args.output_dir, NOISE_LEVELS, str(noise_level))
        cmd = ("python -m mihifepe.simulation.simulation -num_instances 10000 -fraction_relevant_features 0.1 "
               "-noise_multiplier %f -perturbation %s -num_shuffling_trials 500 -condor "
               "-num_features 500 -seed %d -output_dir %s" % (noise_level, args.perturbation, seed, output_dir))
        sims.append(Simulation(cmd, output_dir, noise_level))
    return sims


def shuffling_count_sims(args):
    """Run simulations for different number of shuffling trials"""
    sims = []
    seed = 185
    shuffling_counts = [2000, 1000, 750, 500, 250, 100]
    for shuffling_count in shuffling_counts:
        output_dir = OUTPUTS % (args.output_dir, SHUFFLING_COUNTS, str(shuffling_count))
        cmd = ("python -m mihifepe.simulation.simulation -num_instances 10000 -fraction_relevant_features 0.1 "
               "-noise_multiplier 0.01 -perturbation shuffling -num_shuffling_trials %d -condor "
               "-num_features 500 -seed %d -output_dir %s" % (shuffling_count, seed, output_dir))
        sims.append(Simulation(cmd, output_dir, shuffling_count))
    return sims


def run_simulations(args, simulations):
    """Run simulations"""
    for sim in simulations:
        args.logger.info("Running simulation: '%s'" % sim.cmd)
        subprocess.check_call(sim.cmd, shell=True)


def analyze_simulations(args, simulations):
    """Analyze results of completed simulations"""
    # pylint: disable = too-many-locals
    results_filename = "%s/all_simulation_results_%s.csv" % (args.output_dir, args.type)
    with open(results_filename, "w") as results_file:
        writer = csv.writer(results_file, delimiter=",")
        writer.writerow([args.type, PRECISION, RECALL, FSCORE])
        # Load tree of rejected hypotheses for each sim
        for sim in simulations:
            tree_filename = "%s/%s/%s.json" % (sim.output_dir, constants.HIERARCHICAL_FDR_DIR, constants.HIERARCHICAL_FDR_OUTPUTS)
            with open(tree_filename, "r") as tree_file:
                tree = JsonImporter().read(tree_file)
                nodes = list(anytree.PreOrderIter(tree))
                relevant = [0 if node.description == constants.IRRELEVANT else 1 for node in nodes]
                rejected = [1 if node.rejected else 0 for node in nodes]
                precision, recall, fscore, _ = precision_recall_fscore_support(relevant, rejected, average="binary")
                writer.writerow([str(x) for x in [sim.param, precision, recall, fscore]])


if __name__ == "__main__":
    main()
