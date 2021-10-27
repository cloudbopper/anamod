"""Utility functions for tests"""

import logging

from anamod.core import constants

RESOURCES_DIR = "res"


def setup_logfile(caplog):
    """Set up logfile for test"""
    caplog.set_level(logging.INFO)
    formatting = "%(asctime)s: %(levelname)s: %(name)s: %(message)s"
    handler = caplog.handler
    handler.setFormatter(logging.Formatter(formatting))


def write_logfile(caplog, output_dir):
    """Write log file"""
    log_filename = f"{output_dir}/test.log"
    with open(log_filename, "w", encoding="utf-8") as log_file:
        log_file.write(caplog.text)
    caplog.clear()


def pre_test(func_name, tmpdir, caplog):
    """Pre-test setup"""
    output_dir = f"{tmpdir}/output_dir_{func_name}"
    setup_logfile(caplog)
    return output_dir


def post_test(file_regression, caplog, output_dir):
    """Post-test verification"""
    write_logfile(caplog, output_dir)
    summary_filename = f"{output_dir}/{constants.SIMULATION_SUMMARY_FILENAME}"
    with open(summary_filename, "r", encoding="utf-8") as summary_file:
        summary = "".join(summary_file.readlines())
    file_regression.check(summary, extension="_summary.json")
    important_features_filename = f"{output_dir}/{constants.FEATURE_IMPORTANCE}.csv"
    with open(important_features_filename, "r", encoding="utf-8") as important_features_file:
        important_features = "".join(sorted(important_features_file.readlines()))
    file_regression.check(important_features, extension="_feature_importance.csv")
