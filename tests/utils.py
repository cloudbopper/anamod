"""Utility functions for tests"""

import logging

from anamod import constants


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
    pvalues_filename = "%s/%s" % (output_dir, constants.PVALUES_FILENAME)
    if interactions:
        pvalues_filename = "%s/%s" % (output_dir, constants.INTERACTIONS_PVALUES_FILENAME)
    with open(pvalues_filename, "r") as pvalues_file:
        pvalues = sorted(pvalues_file.readlines())
    file_regression.check("\n".join(pvalues), extension="_pvalues.csv")
    fdr_filename = "%s/%s/%s.csv" % (output_dir, constants.HIERARCHICAL_FDR_DIR, constants.HIERARCHICAL_FDR_OUTPUTS)
    if interactions:
        fdr_filename = "%s/%s/%s.csv" % (output_dir, constants.INTERACTIONS_FDR_DIR, constants.HIERARCHICAL_FDR_OUTPUTS)
    with open(fdr_filename, "r") as fdr_file:
        fdr = sorted(fdr_file.readlines())
    file_regression.check("\n".join(fdr), extension="_fdr.txt")
