"""Generate condor tests from non-condor tests"""

import argparse
from distutils.util import strtobool
import os
import re
import shutil
from anamod.core.constants import HIERARCHICAL, TEMPORAL

TESTS = {HIERARCHICAL: "test_hierarchical.py", TEMPORAL: "test_temporal.py"}
GOLDS = {HIERARCHICAL: "test_hierarchical", TEMPORAL: "test_temporal"}
CONDOR_TEST_DIRECTORY = "condor_tests"
TEST_SIMULATION = "test_simulation"
TEST_CONDOR_SIMULATION = "test_condor_simulation"
TEST_TRIAL = "test_trial"
TEST_CONDOR_TRIAL = "test_condor_trial"
CONDOR_ARGS = "-condor 1 -memory_requirement 1 -disk_requirement 1"
SUBSTITUTIONS = {TEST_SIMULATION: TEST_CONDOR_SIMULATION,
                 "python -m anamod.simulation": f"python -m anamod.simulation {CONDOR_ARGS}",
                 TEST_TRIAL: TEST_CONDOR_TRIAL,
                 "python -m anamod.run_trials": f"python -m anamod.run_trials {CONDOR_ARGS}"}
# TODO: Copy resources directory and change names as needed


def main():
    """Generate condor tests by copying/substituting existing non-condor tests"""
    # pylint: disable = too-many-locals
    # Parse args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-overwrite_golds", help="overwrite existing gold files", type=strtobool, default=True)
    parser.add_argument("-type", default=HIERARCHICAL, choices=[HIERARCHICAL, TEMPORAL])
    args = parser.parse_args()
    test_dir = os.path.dirname(os.path.realpath(__file__))
    condor_test_dir = os.path.join(test_dir, CONDOR_TEST_DIRECTORY)
    if not os.path.exists(condor_test_dir):
        os.makedirs(condor_test_dir)
    test_filename = TESTS[args.type]
    gold_dir = GOLDS[args.type]
    # Write test file
    condor_test_filename = os.path.join(test_dir, CONDOR_TEST_DIRECTORY, test_filename)
    with open(condor_test_filename, "w", encoding="utf-8") as condor_test_file:
        test_filename_abs = os.path.join(test_dir, test_filename)
        with open(test_filename_abs, "r", encoding="utf-8") as test_file:
            for line in test_file:
                for pattern, replacement in SUBSTITUTIONS.items():
                    line = re.sub(pattern, replacement, line)
                condor_test_file.write(line)
    # Write gold files
    condor_gold_dir = os.path.join(condor_test_dir, gold_dir)
    if not os.path.exists(condor_gold_dir):
        os.makedirs(condor_gold_dir)
    gold_dir_abs = os.path.join(test_dir, gold_dir)
    for gold_filename in os.listdir(gold_dir_abs):
        if TEST_TRIAL in gold_filename:
            condor_gold_filename = os.path.join(condor_gold_dir, re.sub(TEST_TRIAL, TEST_CONDOR_TRIAL, gold_filename))
        else:
            condor_gold_filename = os.path.join(condor_gold_dir, re.sub(TEST_SIMULATION, TEST_CONDOR_SIMULATION, gold_filename))
        if not os.path.exists(condor_gold_filename) or args.overwrite_golds:
            shutil.copyfile(os.path.join(gold_dir_abs, gold_filename), condor_gold_filename)


if __name__ == "__main__":
    main()
