"""Run a bunch of simulations (requires condor)"""

import argparse
from collections import namedtuple
import csv
import logging
import os
import time
import subprocess

from mihifepe import constants
from mihifepe.simulation.simulation import evaluate
# TODO: change all import paths to absolute

INSTANCE_COUNTS = "instance_counts"
NOISE_LEVELS = "noise_levels"
FEATURE_COUNTS = "feature_counts"
SHUFFLING_COUNTS = "shuffling_counts"
OUTPUTS = "%s/%s_%s"
Simulation = namedtuple("Simulation", ["cmd", "output_dir", "param"])

def main():
    """Main"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-analyze_results_only", action="store_true", help="only analyze results, instead of generating them as well"
                        " (useful when results already generated")
    parser.add_argument("-type", choices=[INSTANCE_COUNTS, FEATURE_COUNTS, NOISE_LEVELS, SHUFFLING_COUNTS],
                        default=INSTANCE_COUNTS)
    parser.add_argument("-perturbation", choices=[constants.ZEROING, constants.SHUFFLING], default=constants.ZEROING)
    parser.add_argument("-output_dir", required=True)
    parser.add_argument("-seed", type=int, default=None)
    parser.add_argument("-hierarchy_type", choices=[constants.CLUSTER_FROM_DATA, constants.RANDOM], default=constants.RANDOM)
    parser.add_argument("-noise_type", choices=[constants.ADDITIVE_GAUSSIAN, constants.EPSILON_IRRELEVANT], default=constants.ADDITIVE_GAUSSIAN)

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logging.basicConfig(level=logging.INFO, filename="%s/run_simulations.log" % args.output_dir,
                        format="%(asctime)s: %(message)s")
    logger = logging.getLogger(__name__)
    args.logger = logger

    args.logger.info("Begin running/analyzing simulations")
    simulations = parametrize_simulations(args)
    run_simulations(args, simulations)
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
    seed = 9184 if args.seed is None else args.seed
    instance_counts = [16 * 2 ** x for x in range(11)]
    for instance_count in instance_counts:
        output_dir = OUTPUTS % (args.output_dir, INSTANCE_COUNTS, str(instance_count))
        cmd = ("python -m mihifepe.simulation.simulation -num_features 500 -fraction_relevant_features 0.1 -noise_multiplier 0.1 "
               "-clustering_instance_count %d -perturbation %s -num_shuffling_trials 500 -condor "
               "-num_instances %d -seed %d -output_dir %s -hierarchy_type %s -noise_type %s" %
               (instance_counts[-1], args.perturbation, instance_count, seed, output_dir, args.hierarchy_type, args.noise_type))
        sims.append(Simulation(cmd, output_dir, instance_count))
    return sims


def feature_count_sims(args):
    """Run simulations for different values of feature counts"""
    sims = []
    seed = 7185 if args.seed is None else args.seed
    feature_counts = [8 * 2 ** x for x in range(8)]
    for feature_count in feature_counts:
        output_dir = OUTPUTS % (args.output_dir, FEATURE_COUNTS, str(feature_count))
        cmd = ("python -m mihifepe.simulation.simulation -num_instances 10000 -fraction_relevant_features 0.1 "
               "-noise_multiplier 0.01 -perturbation %s -num_shuffling_trials 500 -condor "
               "-num_features %d -seed %d -output_dir %s -hierarchy_type %s -noise_type %s" %
               (args.perturbation, feature_count, seed, output_dir, args.hierarchy_type, args.noise_type))
        sims.append(Simulation(cmd, output_dir, feature_count))
    return sims


def noise_level_sims(args):
    """Run simulations for different values of noise"""
    sims = []
    seed = 85100 if args.seed is None else args.seed
    noise_levels = [0.0] + [0.0001 * 2 ** x for x in range(13)]
    for noise_level in noise_levels:
        output_dir = OUTPUTS % (args.output_dir, NOISE_LEVELS, str(noise_level))
        cmd = ("python -m mihifepe.simulation.simulation -num_instances 10000 -fraction_relevant_features 0.1 "
               "-noise_multiplier %f -perturbation %s -num_shuffling_trials 500 -condor "
               "-num_features 500 -seed %d -output_dir %s -hierarchy_type %s -noise_type %s" %
               (noise_level, args.perturbation, seed, output_dir, args.hierarchy_type, args.noise_type))
        sims.append(Simulation(cmd, output_dir, noise_level))
    return sims


def shuffling_count_sims(args):
    """Run simulations for different number of shuffling trials"""
    sims = []
    seed = 185 if args.seed is None else args.seed
    shuffling_counts = [1000, 750, 500, 250, 100]
    for shuffling_count in shuffling_counts:
        output_dir = OUTPUTS % (args.output_dir, SHUFFLING_COUNTS, str(shuffling_count))
        cmd = ("python -m mihifepe.simulation.simulation -num_instances 1000 -fraction_relevant_features 0.1 "
               "-noise_multiplier 0.01 -perturbation shuffling -num_shuffling_trials %d -condor "
               "-num_features 500 -seed %d -output_dir %s -hierarchy_type %s -noise_type %s" %
               (shuffling_count, seed, output_dir, args.hierarchy_type, args.noise_type))
        sims.append(Simulation(cmd, output_dir, shuffling_count))
    return sims


def run_simulations(args, simulations):
    """Runs simulations in parallel"""
    if not args.analyze_results_only:
        for sim in simulations:
            args.logger.info("Running simulation: '%s'" % sim.cmd)
            sim.popen = subprocess.Popen(sim.cmd, shell=True)


def analyze_simulations(args, simulations):
    """Runs and/or analyzes simulations"""
    if not args.analyze_results_only:
        # Wait for runs to complete
        while not all([sim.popen.poll() is not None for sim in simulations]):
            time.sleep(30)

    results_filename = "%s/all_simulation_results_%s.csv" % (args.output_dir, args.type)
    with open(results_filename, "w", newline="") as results_file:
        writer = csv.writer(results_file, delimiter=",")
        writer.writerow([args.type, constants.FDR, constants.POWER, constants.OUTER_NODES_FDR, constants.OUTER_NODES_POWER,
                         constants.BASE_FEATURES_FDR, constants.BASE_FEATURES_POWER])
        for sim in simulations:
            results = evaluate(sim.output_dir)
            writer.writerow([str(x) for x in [sim.param] + list(results)])
    # Format nicely
    subprocess.check_call("column -t -s ',' {0} > {0}".format(results_filename))


if __name__ == "__main__":
    main()
