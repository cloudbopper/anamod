"""Tests for `mihifepe` package."""

import subprocess
import sys
import tempfile

import pytest

from mihifepe import constants

# pylint: disable = invalid-name, redefined-outer-name, protected-access

@pytest.fixture
def tempdir():
    """Create temporary directory for tests to run in"""
    return tempfile.mkdtemp()


def test_simulation_random_hierarchy(file_regression, tempdir):
    """Test simulation with zeroing perturbation and random hierarchy"""
    func_name = sys._getframe().f_code.co_name
    output_dir = "%s/output_dir_%s" % (tempdir, func_name)
    pvalues_filename = "%s/%s" % (output_dir, constants.PVALUES_FILENAME)
    cmd = ("python -m mihifepe.simulation -seed 1 -num_instances 100 -num_features 10 -fraction_relevant_features 0.5"
           " -hierarchy_type random -perturbation zeroing -output_dir %s" % output_dir)
    subprocess.check_call(cmd, shell=True)
    with open(pvalues_filename, "r") as pvalues_file:
        pvalues = pvalues_file.read()
    file_regression.check(pvalues, extension="_pvalues.csv")
    fdr_filename = "%s/%s/%s.csv" % (output_dir, constants.HIERARCHICAL_FDR_DIR, constants.HIERARCHICAL_FDR_OUTPUTS)
    with open(fdr_filename, "r") as fdr_file:
        fdr = fdr_file.read()
    file_regression.check(fdr, extension="_fdr.csv")

