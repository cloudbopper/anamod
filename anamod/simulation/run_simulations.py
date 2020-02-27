"""Run a bunch of simulations"""

import argparse
from collections import OrderedDict
import configparser
import json
import os
import time
import subprocess

from anamod import constants, utils
from anamod.constants import DEFAULT, INSTANCE_COUNTS, NOISE_LEVELS, FEATURE_COUNTS, SHUFFLING_COUNTS, ALL_SIMULATION_RESULTS

OUTPUTS = "%s/%s_%s"


class Simulation():
    """Simulation helper class"""
    # pylint: disable = too-few-public-methods
    def __init__(self, cmd, output_dir, param, popen=None):
        self.cmd = cmd
        self.output_dir = output_dir
        self.param = param
        self.popen = popen


def main():
    """Main"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-analyze_results_only", action="store_true", help="only analyze results, instead of generating them as well"
                        " (useful when results already generated")
    parser.add_argument("-type", choices=[DEFAULT, INSTANCE_COUNTS, FEATURE_COUNTS, NOISE_LEVELS, SHUFFLING_COUNTS],
                        default=DEFAULT)
    parser.add_argument("-analysis_type", default=constants.TEMPORAL, choices=[constants.TEMPORAL, constants.HIERARCHICAL])
    parser.add_argument("-output_dir", required=True)
    parser.add_argument("-seed", type=int, required=True)
    parser.add_argument("-wait_period", type=int, default=30, help="Time in seconds to wait before checking subprocess(es) progress")

    args, pass_arglist = parser.parse_known_args()  # Any unknown options will be passed to simulation.py
    args.pass_arglist = " ".join(pass_arglist)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.logger = utils.get_logger(__name__, "%s/run_simulations.log" % args.output_dir)
    args.config_filename = constants.CONFIG_HIERARCHICAL if args.analysis_type == constants.HIERARCHICAL else constants.CONFIG_TEMPORAL

    args.logger.info("Begin running/analyzing simulations")
    args.config = load_config(args)
    simulations = parametrize_simulations(args)
    run_simulations(args, simulations)
    analyze_simulations(args, simulations)
    args.logger.info("End running simulations")


def load_config(args):
    """Load simulation configuration"""
    config_filename = "%s/%s" % (os.path.dirname(__file__), args.config_filename)
    assert os.path.exists(config_filename)
    config = configparser.ConfigParser()
    config.read(config_filename)
    dconfig = config[args.type]
    sconfig = ""
    for option, value in dconfig.items():
        if value == "-1":
            continue
        sconfig += "-%s " % option
        sconfig += "%s " % value
    return sconfig


def parametrize_simulations(args):
    """Parametrize simulations"""
    if args.type == DEFAULT:
        return default_sims(args)
    if args.type == INSTANCE_COUNTS:
        return instance_count_sims(args)
    if args.type == FEATURE_COUNTS:
        return feature_count_sims(args)
    if args.type == NOISE_LEVELS:
        return noise_level_sims(args)
    if args.type == SHUFFLING_COUNTS:
        return shuffling_count_sims(args)
    raise NotImplementedError("Unknown simulation type")


def default_sims(args):
    """Configure simulation with default config values"""
    sims = []
    output_dir = OUTPUTS % (args.output_dir, DEFAULT, DEFAULT)
    cmd = ("python -m anamod.simulation.simulation %s -seed %d -output_dir %s %s" %
           (args.config, args.seed, output_dir, args.pass_arglist))
    sims.append(Simulation(cmd, output_dir, DEFAULT))
    return sims


def instance_count_sims(args):
    """Configure simulations for different values of instance counts"""
    sims = []
    instance_counts = [16 * 2 ** x for x in range(11)]
    max_instance_count = instance_counts[-1]
    for instance_count in instance_counts:
        output_dir = OUTPUTS % (args.output_dir, INSTANCE_COUNTS, str(instance_count))
        cmd = ("python -m anamod.simulation.simulation %s -num_instances %d -clustering_instance_count %d "
               "-seed %d -output_dir %s %s" %
               (args.config, instance_count, max_instance_count, args.seed, output_dir, args.pass_arglist))
        sims.append(Simulation(cmd, output_dir, instance_count))
    return sims


def feature_count_sims(args):
    """Run simulations for different values of feature counts"""
    sims = []
    feature_counts = [8 * 2 ** x for x in range(8)]
    for feature_count in feature_counts:
        output_dir = OUTPUTS % (args.output_dir, FEATURE_COUNTS, str(feature_count))
        cmd = ("python -m anamod.simulation.simulation %s -num_features %d -seed %d -output_dir %s %s" %
               (args.config, feature_count, args.seed, output_dir, args.pass_arglist))
        sims.append(Simulation(cmd, output_dir, feature_count))
    return sims


def noise_level_sims(args):
    """Run simulations for different values of noise"""
    sims = []
    noise_levels = [0.0] + [0.01 * 2 ** x for x in range(8)]
    for noise_level in noise_levels:
        output_dir = OUTPUTS % (args.output_dir, NOISE_LEVELS, str(noise_level))
        cmd = ("python -m anamod.simulation.simulation %s -noise_multiplier %f -seed %d -output_dir %s %s" %
               (args.config, noise_level, args.seed, output_dir, args.pass_arglist))
        sims.append(Simulation(cmd, output_dir, noise_level))
    return sims


def shuffling_count_sims(args):
    """Run simulations for different number of shuffling trials"""
    sims = []
    shuffling_counts = [1000, 750, 500, 250, 100]
    for shuffling_count in shuffling_counts:
        output_dir = OUTPUTS % (args.output_dir, SHUFFLING_COUNTS, str(shuffling_count))
        cmd = ("python -m anamod.simulation.simulation %s -num_shuffling_trials %d -seed %d -output_dir %s %s" %
               (args.config, shuffling_count, args.seed, output_dir, args.pass_arglist))
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
            time.sleep(args.wait_period)

    # Collate simulation results
    results_filename = "%s/%s_%s.json" % (args.output_dir, ALL_SIMULATION_RESULTS, args.type)
    with open(results_filename, "w") as results_file:
        root = OrderedDict()
        for sim in simulations:
            sim_results_filename = "%s/%s" % (sim.output_dir, constants.SIMULATION_RESULTS_FILENAME)
            with open(sim_results_filename, "r") as sim_results_file:
                data = json.load(sim_results_file)
                root[sim.param] = data
        json.dump(root, results_file)


if __name__ == "__main__":
    main()
