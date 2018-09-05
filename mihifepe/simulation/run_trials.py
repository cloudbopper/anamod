"""Run multiple trials of multiple simulations"""

import argparse
import csv
from collections import namedtuple
import logging
import os
import subprocess

from mihifepe import constants

TRIAL = "trial"
SUMMARY_FILENAME = "all_trials_summary"

Trial = namedtuple("Trial", ["cmd", "output_dir"])

def main():
    """Main"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-num_trials", type=int, default=3)
    parser.add_argument("-start_seed", type=int, default=100000)
    parser.add_argument("-type", choices=[constants.INSTANCE_COUNTS, constants.FEATURE_COUNTS, constants.NOISE_LEVELS, constants.SHUFFLING_COUNTS],
                        default=constants.INSTANCE_COUNTS)
    parser.add_argument("-summarize_only", action="store_true")
    parser.add_argument("-output_dir", required=True)

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logging.basicConfig(level=logging.INFO, filename="%s/run_trials.log" % args.output_dir,
                        format="%(asctime)s: %(message)s")
    logger = logging.getLogger(__name__)
    args.logger = logger

    trials = gen_trials(args)
    run_trials(args, trials)
    summarize_trials(args, trials)


def gen_trials(args):
    """Generate multiple trials, each with multiple simulations"""
    trials = []
    for seed in range(args.start_seed, args.start_seed + args.num_trials):
        output_dir = "%s/trial_%s_%d" % (args.output_dir, args.type, seed)
        cmd = "python -m mihifepe.simulation.run_simulations -seed %d -type %s -output_dir %s" % (seed, args.type, output_dir)
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
    summary_filename = "%s/%s.csv" % (args.output_dir, SUMMARY_FILENAME)
    for trial in trials:
        None # TODO


if __name__ == "__main__":
    main()
