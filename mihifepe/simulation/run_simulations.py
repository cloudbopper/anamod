"""Run a bunch of simulations (requires condor)"""

import argparse
import logging
import os
import subprocess

INSTANCE_COUNTS = "instance_counts"
NOISE_LEVELS = "noise_levels"
FEATURE_COUNTS = "feature_counts"
OUTPUTS = "%s/%s_%s"

def main():
    """Main"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-generate_results", action="store_true")
    parser.add_argument("-visualize_results", action="store_true")
    parser.add_argument("-type", choices=[INSTANCE_COUNTS, FEATURE_COUNTS, NOISE_LEVELS],
                        default=INSTANCE_COUNTS)
    parser.add_argument("-output_dir", required=True)

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logging.basicConfig(level=logging.INFO, filename="%s/run_simulations.log" % args.output_dir,
                        format="%(asctime)s: %(message)s")
    logger = logging.getLogger(__name__)
    args.logger = logger

    args.logger.info("Begin running/analyzing simulations")
    if args.generate_results:
        if args.type == INSTANCE_COUNTS:
            run_instance_count_sims(args)
        elif args.type == FEATURE_COUNTS:
            run_feature_count_sims(args)
        elif args.type == NOISE_LEVELS:
            raise NotImplementedError() # TODO
    if args.visualize_results:
        raise NotImplementedError() # TODO
    args.logger.info("End running simulations")


def run_instance_count_sims(args):
    """Run simulations for different values of instance counts"""
    seed = 9184
    instance_counts = [50, 100, 500, 1000, 5000, 10000]
    for instance_count in instance_counts:
        args.logger.info("Simulating with instance count %d" % instance_count)
        output_dir = OUTPUTS % (args.output_dir, INSTANCE_COUNTS, str(instance_count))
        cmd = ("python -m mihifepe.simulation.simulation -num_features 500 -fraction_relevant_features 0.1 -noise_multiplier 0.01 "
               "-cluster_instance_count 10000 -perturbation shuffling -num_shuffling_trials 500 -condor "
               "-num_instances %d -seed %d -output_dir %s" % (instance_count, seed, output_dir))
        args.logger.info("Running command: %s" % cmd)
        subprocess.check_call(cmd, shell=True)
        args.logger.info("Completed running command: %s" % cmd)


def run_feature_count_sims(args):
    """Run simulations for different values of feature counts"""
    seed = 7185
    feature_counts = [10, 50, 100, 500, 1000]
    for feature_count in feature_counts:
        args.logger.info("Simulating with feature count %d" % feature_count)
        output_dir = OUTPUTS % (args.output_dir, FEATURE_COUNTS, str(feature_count))
        cmd = ("python -m mihifepe.simulation.simulation -num_instances 1000 -fraction_relevant_features 0.1 "
               "-noise_multiplier 0.01 -perturbation zeroing -condor "
               "-num_features %d -seed %d -output_dir %s" % (feature_count, seed, output_dir))
        args.logger.info("Running command: %s" % cmd)
        subprocess.check_call(cmd, shell=True)
        args.logger.info("Completed running command: %s" % cmd)


if __name__ == "__main__":
    main()
