"""Generate condor tests from non-condor tests"""

import argparse
import os
import re
import shutil

TEST_FILENAME = "test_mihifepe.py"
GOLD_DIRECTORY = "test_mihifepe"
CONDOR_TEST_DIRECTORY = "condor_tests"
TEST_SIMULATION = "test_simulation"
TEST_CONDOR_SIMULATION = "test_condor_simulation"
SUBSTITUTIONS = {TEST_SIMULATION: TEST_CONDOR_SIMULATION, "python -m mihifepe.simulation": "python -m mihifepe.simulation -condor"}


def main():
    """Generate condor tests by copying/substituting existing non-condor tests"""
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("-overwrite_golds", help="overwrite existing gold files", action="store_true")
    args = parser.parse_args()
    test_dir = os.path.dirname(os.path.realpath(__file__))
    condor_test_dir = os.path.join(test_dir, CONDOR_TEST_DIRECTORY)
    if not os.path.exists(condor_test_dir):
        os.makedirs(condor_test_dir)
    # Write test file
    test_filename = os.path.join(test_dir, TEST_FILENAME)
    condor_test_filename = os.path.join(test_dir, CONDOR_TEST_DIRECTORY, TEST_FILENAME)
    with open(condor_test_filename, "w") as condor_test_file:
        with open(test_filename, "r") as test_file:
            for line in test_file:
                for pattern, replacement in SUBSTITUTIONS.items():
                    line = re.sub(pattern, replacement, line)
                condor_test_file.write(line)
    # Write gold files
    gold_dir = os.path.join(test_dir, GOLD_DIRECTORY)
    condor_gold_dir = os.path.join(condor_test_dir, GOLD_DIRECTORY)
    if not os.path.exists(condor_gold_dir):
        os.makedirs(condor_gold_dir)
    for gold_filename in os.listdir(gold_dir):
        condor_gold_filename = os.path.join(condor_gold_dir, re.sub(TEST_SIMULATION, TEST_CONDOR_SIMULATION, gold_filename))
        if not os.path.exists(condor_gold_filename) or args.overwrite_golds:
            shutil.copyfile(os.path.join(gold_dir, gold_filename), condor_gold_filename)


if __name__ == "__main__":
    main()
