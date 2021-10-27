"""Utility functions for tests"""
import csv
import json
import logging
import re

import numpy as np

from anamod.core import constants

RESOURCES_DIR = "res"
FP_KEYS = {"importance_score"}


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
    with open(summary_filename, "rb") as summary_file:
        summary = json.load(summary_file)
    file_regression.check(json.dumps(round_fp(summary), indent=2), extension="_summary.json")
    # CSV doesn't indicate data types, so first identify FP values, then replace them with rounded values to avoid FP variations across platforms
    pattern = re.compile(r"importance_score|pvalue")
    important_features_filename = f"{output_dir}/{constants.FEATURE_IMPORTANCE}.csv"
    with open(important_features_filename, "r", encoding="utf-8") as important_features_file:
        reader = csv.DictReader(important_features_file)
        strarr = [",".join(reader.fieldnames)]
        for row in reader:
            rowarr = [str(round_fp(float(value))) if pattern.search(key) else value for key, value in row.items()]
            strarr.append(",".join(rowarr))
        file_regression.check("\n".join(sorted(strarr)), extension="_feature_importance.csv")


def round_fp(data):
    """Round FP values in data to avoid regression test errors due to FP precision variations across platforms"""
    dtype = type(data)
    if dtype == float:
        return round(data, 10)
    elif dtype == np.ndarray:
        return np.round(data, 10)
    elif dtype == dict:
        for key, value in data.items():
            data[key] = round_fp(value)
    elif dtype == list:
        return [round_fp(item) for item in data]
    return data
