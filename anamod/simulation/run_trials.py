"""Run multiple trials of multiple simulations"""

import argparse
from collections import namedtuple, OrderedDict
from copy import copy
import configparser
from distutils.util import strtobool
import json
import os
import subprocess
import time

import numpy as np
from anamod import constants, utils
from anamod.constants import DEFAULT, INSTANCE_COUNTS, NOISE_LEVELS, FEATURE_COUNTS, SHUFFLING_COUNTS
from anamod.constants import SEQUENCE_LENGTHS, WINDOW_SEQUENCE_DEPENDENCE, MODEL_TYPES, TEST

TRIAL = "trial"
SUMMARY_FILENAME = "all_trials_summary"
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


class Trial():  # pylint: disable = too-many-instance-attributes
    """Class that parametrizes, runs, monitors, and analyzes a group of simulations"""
    def __init__(self, seed, sim_type, analysis_type, output_dir, pass_args, **kwargs):
        # pylint: disable = too-many-arguments
        self.seed = seed
        self.type = sim_type
        self.analysis_type = analysis_type
        self.output_dir = output_dir
        self.pass_args = pass_args
        self.summarize_only = kwargs.get("summarize_only", False)
        self.setup_simulations()

    def __hash__(self):
        return hash(self.seed)

    def setup_simulations(self):
        """Set up simulations"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.logger = utils.get_logger(__name__, "%s/run_simulations.log" % self.output_dir)
        self.config_filename = constants.CONFIG_HIERARCHICAL if self.analysis_type == constants.HIERARCHICAL else constants.CONFIG_TEMPORAL
        self.logger.info(f"Trial for seed {self.seed}: Begin running/analyzing simulations")
        self.config, test_param = self.load_config()
        self.simulations = self.parametrize_simulations(test_param)
        self.error = False  # Flag to indicate if any simulation failed
        self.running_sims = set()  # Set of simulations running concurrently

    def load_config(self):
        """Load simulation configuration"""
        config_filename = "%s/%s" % (os.path.dirname(__file__), self.config_filename)
        assert os.path.exists(config_filename)
        config = configparser.ConfigParser()
        config.read(config_filename)
        dconfig = config[self.type]
        sconfig = ""
        test_param = None
        for option, value in dconfig.items():
            if "," in value:
                test_param = TestParam(option, [item.strip() for item in value.split(",")])
                continue
            sconfig += "-%s " % option
            sconfig += "%s " % value
        return sconfig, test_param

    def parametrize_simulations(self, test_param):
        """Parametrize simulations"""
        sims = []
        if test_param is None:
            test_param = TestParam("", [constants.DEFAULT])
        key, values = test_param
        for value in values:
            output_dir = OUTPUTS % (self.output_dir, self.type, value)
            test_param_str = "-%s %s" % (key, value) if key else ""
            cmd = ("python -m anamod.simulation.simulation %s %s -seed %d -output_dir %s %s" %
                   (test_param_str, self.config, self.seed, output_dir, self.pass_args))
            sims.append(Simulation(cmd, output_dir, value))
        return sims

    def run_simulations(self):
        """Runs simulations in parallel"""
        if self.summarize_only:
            return
        for sim in self.simulations:
            self.logger.info(f"Running simulation: '{sim.cmd}'")
            sim.popen = subprocess.Popen(sim.cmd, shell=True)
            self.running_sims.add(sim)

    def monitor_simulations(self):
        """Monitor simulation progress and returns completion status"""
        for sim in copy(self.running_sims):
            returncode = sim.popen.poll()
            if returncode is not None:
                self.running_sims.remove(sim)
                if returncode != 0:
                    self.logger.error(f"Simulation {sim.cmd} failed; see logs in {sim.output_dir}")
                    self.error = True  # pylint: disable = attribute-defined-outside-init
        if not self.running_sims:
            assert not self.error, f"run_simulations.py failed, see log at {self.output_dir}"
            self.analyze_simulations()
            return True
        return False

    def analyze_simulations(self):
        """Collate and analyze simulation results"""
        results_filename = "%s/%s_%s.json" % (self.output_dir, constants.ALL_SIMULATION_RESULTS, self.type)
        with open(results_filename, "w") as results_file:
            root = OrderedDict()
            for sim in self.simulations:
                sim_results_filename = "%s/%s" % (sim.output_dir, constants.SIMULATION_RESULTS_FILENAME)
                with open(sim_results_filename, "r") as sim_results_file:
                    data = json.load(sim_results_file)
                    root[sim.param] = data
            json.dump(root, results_file)
        self.logger.info("Trial for seed {self.seed}: End running/analyzing simulations")


def main(strargs=""):
    """Main"""
    args = parse_arguments(strargs)
    return pipeline(args)


def parse_arguments(strargs):
    """Parse arguments from input string or command-line"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-num_trials", type=int, default=3, help="number of trials to perform and average results over")
    parser.add_argument("-max_concurrent_simulations", type=int, default=4, help="number of simulations to run concurrently"
                        " (only increase if running on htcondor)")
    parser.add_argument("-trial_wait_period", type=int, default=60, help="time in seconds to wait before checking trial status")
    parser.add_argument("-start_seed", type=int, default=100000, help="randomization seed for first trial, incremented for"
                        " every subsequent trial.")
    parser.add_argument("-type", choices=[DEFAULT, INSTANCE_COUNTS, FEATURE_COUNTS, NOISE_LEVELS, SHUFFLING_COUNTS,
                                          SEQUENCE_LENGTHS, WINDOW_SEQUENCE_DEPENDENCE, MODEL_TYPES, TEST],
                        default=DEFAULT, help="type of parameter to vary across simulations")
    parser.add_argument("-analysis_type", default=constants.TEMPORAL, choices=[constants.TEMPORAL, constants.HIERARCHICAL],
                        help="type of analysis to perform")
    parser.add_argument("-summarize_only", type=strtobool, default=False,
                        help="attempt to summarize results assuming they're already generated")
    parser.add_argument("-output_dir", required=True)
    args, pass_arglist = parser.parse_known_args(strargs.split(" ")) if strargs else parser.parse_known_args()
    args.pass_args = " ".join(pass_arglist)

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
    trials = set()
    for seed in range(args.start_seed, args.start_seed + args.num_trials):
        output_dir = "%s/trial_%s_%d" % (args.output_dir, args.type, seed)
        trials.add(Trial(seed, args.type, args.analysis_type, output_dir, args.pass_args))
    return trials


def run_trials(args, trials):
    """Run multiple trials, each with multiple simulations"""
    # TODO: Maybe time trials as well, both actual and CPU time (which should exclude condor delays)
    if args.summarize_only:
        return
    running_trials = set()  # Trials currently running
    waiting_trials = copy(trials)  # Trials that haven't yet started
    unfinished_trials = copy(trials)  # Trials that haven't finished for whatever reason
    while unfinished_trials:
        concurrent_simulation_count = 0
        # Monitor running trials and determine available slots
        for trial in copy(running_trials):
            concurrent_simulation_count += len(trial.running_sims)
            completed = trial.monitor_simulations()
            if completed:
                running_trials.remove(trial)
                unfinished_trials.remove(trial)
        # Run trials if slots available
        for trial in copy(waiting_trials):
            if concurrent_simulation_count < args.max_concurrent_simulations:
                trial.run_simulations()
                waiting_trials.remove(trial)
                running_trials.add(trial)
                concurrent_simulation_count += len(trial.running_sims)
        if not unfinished_trials:
            break  # All trials completed, don't need to wait
        time.sleep(args.trial_wait_period)  # Wait before resuming monitoring/launching trials
    # Report errors for failed trials
    error = False
    for trial in trials:
        if trial.error:
            args.logger.error(f"Trial {trial.cmd} failed; see logs in {trial.output_dir}")
            error = True
    assert not error, f"run_trials.py failed, see log at {args.output_dir}"


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
                    # FIXME: Want to plot window accuracy by overlap (instead of average window overlap)
                    if key == constants.WINDOW_OVERLAP:
                        value = np.mean(list(value.values())) if value else 0.
                    data[key][param][tidx] = value
    return data


if __name__ == "__main__":
    main()
