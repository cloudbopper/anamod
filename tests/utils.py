"""Utility functions for tests"""

import logging

from anamod.core import constants


def setup_logfile(caplog):
    """Set up logfile for test"""
    caplog.set_level(logging.INFO)
    formatting = "%(asctime)s: %(levelname)s: %(name)s: %(message)s"
    handler = caplog.handler
    handler.setFormatter(logging.Formatter(formatting))


def write_logfile(caplog, output_dir):
    """Write log file"""
    log_filename = "{0}/test.log".format(output_dir)
    with open(log_filename, "w") as log_file:
        log_file.write(caplog.text)
    caplog.clear()


def pre_test(func_name, tmpdir, caplog):
    """Pre-test setup"""
    output_dir = "%s/output_dir_%s" % (tmpdir, func_name)
    setup_logfile(caplog)
    return output_dir


def post_test(file_regression, caplog, output_dir, interactions=False):
    """Post-test verification"""
    write_logfile(caplog, output_dir)
    summary_filename = f"{output_dir}/{constants.SIMULATION_SUMMARY_FILENAME}"
    with open(summary_filename, "r") as summary_file:
        summary = "".join(summary_file.readlines())
    file_regression.check(summary, extension="_summary.json")
    fdr_filename = "%s/%s/%s.csv" % (output_dir, constants.HIERARCHICAL_FDR_DIR, constants.HIERARCHICAL_FDR_OUTPUTS)
    if interactions:
        fdr_filename = "%s/%s/%s.csv" % (output_dir, constants.INTERACTIONS_FDR_DIR, constants.HIERARCHICAL_FDR_OUTPUTS)
    with open(fdr_filename, "r") as fdr_file:
        fdr = "".join(fdr_file.readlines())
    file_regression.check(fdr, extension="_fdr.csv")
