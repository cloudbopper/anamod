"""Run a bunch of simulations"""

import argparse
from collections import namedtuple, OrderedDict
from copy import copy
from distutils.util import strtobool
import configparser
import json
import os
import time
import subprocess

from anamod import constants, utils
from anamod.constants import DEFAULT, INSTANCE_COUNTS, NOISE_LEVELS, FEATURE_COUNTS, SHUFFLING_COUNTS, ALL_SIMULATION_RESULTS
from anamod.constants import SEQUENCE_LENGTHS, WINDOW_SEQUENCE_DEPENDENCE, MODEL_TYPES

OUTPUTS = "%s/%s_%s"
TestParam = namedtuple("TestParameter", ["key", "values"])


class Simulation():
    """Simulation helper class"""
    # pylint: disable = too-few-public-methods
    def __init__(self, cmd, output_dir, param, popen=None):
        self.cmd = cmd
        self.output_dir = output_dir
        self.param = param
        self.popen = popen

    def __hash__(self):
        return hash(self.cmd)


def main():
    """Main"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-analyze_results_only", type=strtobool, default=False, help="only analyze results, instead of generating them as well"
                        " (useful when results already generated")
    parser.add_argument("-type", choices=[DEFAULT, INSTANCE_COUNTS, FEATURE_COUNTS, NOISE_LEVELS, SHUFFLING_COUNTS,
                                          SEQUENCE_LENGTHS, WINDOW_SEQUENCE_DEPENDENCE, MODEL_TYPES],
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
    args.config, test_param = load_config(args)
    simulations = parametrize_simulations(args, test_param)
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
    test_param = None
    for option, value in dconfig.items():
        if "," in value:
            test_param = TestParam(option, [item.strip() for item in value.split(",")])
            continue
        sconfig += "-%s " % option
        sconfig += "%s " % value
    return sconfig, test_param


def parametrize_simulations(args, test_param):
    """Parametrize simulations"""
    sims = []
    if test_param is None:
        test_param = TestParam("", [DEFAULT])
    key, values = test_param
    for value in values:
        output_dir = OUTPUTS % (args.output_dir, args.type, value)
        test_param_str = "-%s %s" % (key, value) if key else ""
        cmd = ("python -m anamod.simulation.simulation %s %s -seed %d -output_dir %s %s" %
               (test_param_str, args.config, args.seed, output_dir, args.pass_arglist))
        sims.append(Simulation(cmd, output_dir, value))
    return sims


def run_simulations(args, simulations):
    """Runs simulations in parallel"""
    if not args.analyze_results_only:
        running_sims = set()
        for sim in simulations:
            args.logger.info("Running simulation: '%s'" % sim.cmd)
            sim.popen = subprocess.Popen(sim.cmd, shell=True)
            running_sims.add(sim)
        while running_sims:
            time.sleep(args.wait_period)
            for sim in copy(running_sims):
                returncode = sim.popen.poll()
                if returncode is not None:
                    running_sims.remove(sim)
                    if returncode != 0:
                        args.logger.error(f"Simulation {sim.cmd} failed; see logs in {sim.output_dir}")


def analyze_simulations(args, simulations):
    """Collate and analyze simulation results"""
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
