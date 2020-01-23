"""Utility functions for tests"""

import logging


def setup_logfile(caplog):
    """Set up logfile for test"""
    caplog.set_level(logging.INFO)
    formatting = "%(asctime)s: %(levelname)s: %(name)s: %(message)s"
    handler = caplog.handler
    handler.setFormatter(logging.Formatter(formatting))


def pre_test(func_name, tmpdir, caplog):
    """Pre-test setup"""
    output_dir = "%s/output_dir_%s" % (tmpdir, func_name)
    setup_logfile(caplog)
    return output_dir


def write_logfile(caplog, output_dir):
    """Write log file"""
    log_filename = "{0}/test.log".format(output_dir)
    with open(log_filename, "w") as log_file:
        log_file.write(caplog.text)
    caplog.clear()
