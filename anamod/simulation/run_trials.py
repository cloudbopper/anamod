"""Run multiple trials of multiple simulations"""

import argparse
from copy import copy
from distutils.util import strtobool
import json
from collections import OrderedDict
import os
import subprocess
import time

import numpy as np
from anamod import constants, utils
from anamod.constants import DEFAULT, INSTANCE_COUNTS, NOISE_LEVELS, FEATURE_COUNTS, SHUFFLING_COUNTS
from anamod.constants import SEQUENCE_LENGTHS, WINDOW_SEQUENCE_DEPENDENCE, MODEL_TYPES

TRIAL = "trial"
SUMMARY_FILENAME = "all_trials_summary"


class Trial():
    """Trial helper class"""
    # pylint: disable = too-few-public-methods
    def __init__(self, cmd, output_dir, popen=None):
        self.cmd = cmd
        self.output_dir = output_dir
        self.popen = popen
        self.returncode = None

    def __hash__(self):
        return hash(self.cmd)


def main(strargs=""):
    """Main"""
    args = parse_arguments(strargs)
    return pipeline(args)


def parse_arguments(strargs):
    """Parse arguments from input string or command-line"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-num_trials", type=int, default=3, help="number of trials to perform and average results over")
    parser.add_argument("-max_concurrent_trials", type=int, default=1, help="number of trials to run concurrently"
                        " (only increase if running on htcondor)")
    parser.add_argument("-trial_wait_period", type=int, default=60, help="time in seconds to wait before checking trial status")
    parser.add_argument("-start_seed", type=int, default=100000, help="randomization seed for first trial, incremented for"
                        " every subsequent trial.")
    parser.add_argument("-type", choices=[DEFAULT, INSTANCE_COUNTS, FEATURE_COUNTS, NOISE_LEVELS, SHUFFLING_COUNTS,
                                          SEQUENCE_LENGTHS, WINDOW_SEQUENCE_DEPENDENCE, MODEL_TYPES],
                        default=DEFAULT, help="type of parameter to vary across simulations")
    parser.add_argument("-analysis_type", default=constants.TEMPORAL, choices=[constants.TEMPORAL, constants.HIERARCHICAL],
                        help="type of analysis to perform")
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
    args.logger.info(f"Begin running trials with config: {args}")
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
    # TODO: Maybe time trials as well, both actual and CPU time (which should exclude condor delays)
    if args.summarize_only:
        return

    def prune_running_trials(running_trials):
        """Prune trials that have completed running"""
        time.sleep(args.trial_wait_period)
        for trial in copy(running_trials):
            trial.returncode = trial.popen.poll()
            if trial.returncode is not None:
                running_trials.remove(trial)

    running_trials = set()
    # Run trials
    for trial in trials:
        while len(running_trials) >= args.max_concurrent_trials:  # Wait for one or more trials to complete before launching more
            prune_running_trials(running_trials)
        args.logger.info("Running trial: %s" % trial.cmd)
        trial.popen = subprocess.Popen(trial.cmd, shell=True)
        running_trials.add(trial)
    # Wait for all trials to complete
    while running_trials:
        prune_running_trials(running_trials)
    # Report errors for failed trials
    for trial in trials:
        if trial.returncode != 0:
            args.logger.error(f"Trial {trial.cmd} failed; see logs in {trial.output_dir}")


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
