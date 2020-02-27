"""Run multiple trials of multiple simulations"""

import argparse
from distutils.util import strtobool
import json
from collections import namedtuple, OrderedDict
import os
import subprocess

import numpy as np
from anamod import constants, utils
from anamod.constants import DEFAULT, INSTANCE_COUNTS, NOISE_LEVELS, FEATURE_COUNTS, SHUFFLING_COUNTS
from anamod.constants import SEQUENCE_LENGTHS, WINDOW_SEQUENCE_DEPENDENCE, MODEL_TYPES

TRIAL = "trial"
SUMMARY_FILENAME = "all_trials_summary"

Trial = namedtuple("Trial", ["cmd", "output_dir"])


def main(strargs=""):
    """Main"""
    args = parse_arguments(strargs)
    return pipeline(args)


def parse_arguments(strargs):
    """Parse arguments from input string or command-line"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-num_trials", type=int, default=3)
    parser.add_argument("-start_seed", type=int, default=100000)
    parser.add_argument("-type", choices=[DEFAULT, INSTANCE_COUNTS, FEATURE_COUNTS, NOISE_LEVELS, SHUFFLING_COUNTS,
                                          SEQUENCE_LENGTHS, WINDOW_SEQUENCE_DEPENDENCE, MODEL_TYPES],
                        default=DEFAULT)
    parser.add_argument("-analysis_type", default=constants.TEMPORAL, choices=[constants.TEMPORAL, constants.HIERARCHICAL])
    parser.add_argument("-summarize_only", help="enable to assume that the results are already generated,"
                        " and just summarize them", type=strtobool, default=False)
    parser.add_argument("-output_dir", required=True)
    args, pass_arglist = parser.parse_known_args(strargs.split(" ")) if strargs else parser.parse_known_args()
    args.pass_arglist = " ".join(pass_arglist)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.logger = utils.get_logger(__name__, "%s/run_trials.log" % args.output_dir)
    return args


def pipeline(args):
    """Pipeline"""
    trials = gen_trials(args)
    run_trials(args, trials)
    return summarize_trials(args, trials)


def gen_trials(args):
    """Generate multiple trials, each with multiple simulations"""
    trials = []
    for seed in range(args.start_seed, args.start_seed + args.num_trials):
        output_dir = "%s/trial_%s_%d" % (args.output_dir, args.type, seed)
        cmd = ("python -m anamod.simulation.run_simulations -seed %d -type %s -analysis_type %s -output_dir %s %s" %
               (seed, args.type, args.analysis_type, output_dir, args.pass_arglist))
        trials.append(Trial(cmd, output_dir))
    return trials


def run_trials(args, trials):
    """Run multiple trials, each with multiple simulations"""
    if not args.summarize_only:
        for trial in trials:
            args.logger.info("Running trial: %s" % trial.cmd)
            subprocess.check_call(trial.cmd, shell=True)


def summarize_trials(args, trials):
    """Summarize outputs from trials"""
    data = {}
    for tidx, trial in enumerate(trials):
        trial_results_filename = "%s/%s_%s.json" % (trial.output_dir, constants.ALL_SIMULATION_RESULTS, args.type)
        with open(trial_results_filename, "r") as trial_results_file:
            sim_data = json.load(trial_results_file, object_pairs_hook=OrderedDict)
            for param, results in sim_data.items():
                for key, value in results.items():
                    if key not in data:
                        data[key] = OrderedDict()
                    if param not in data[key]:
                        data[key][param] = np.zeros(args.num_trials)
                    if key == constants.WINDOW_OVERLAP:
                        value = np.mean(list(value.values()))  # FIXME: Want to plot window accuracy by overlap (instead of average window overlap)
                    data[key][param][tidx] = value
    return data


if __name__ == "__main__":
    main()
