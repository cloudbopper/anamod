"""Tests for `mihifepe` package."""

import subprocess
import sys

from mihifepe import constants

# pylint: disable = invalid-name, redefined-outer-name, protected-access

# @pytest.fixture
# def tempdir():
#     """Create temporary directory for tests to run in"""
#     return tempfile.mkdtemp()


def test_condor_simulation_random_hierarchy(file_regression, tmpdir):
    """Test simulation with random hierarchy"""
    func_name = sys._getframe().f_code.co_name
    output_dir = "%s/output_dir_%s" % (tmpdir, func_name)
    pvalues_filename = "%s/%s" % (output_dir, constants.PVALUES_FILENAME)
    cmd = ("python -m mihifepe.simulation -seed 1 -num_instances 100 -num_features 10 -fraction_relevant_features 0.5"
           " -contiguous_node_names -hierarchy_type random -perturbation zeroing -output_dir %s -condor" % output_dir)
    subprocess.check_call(cmd, shell=True)
    with open(pvalues_filename, "r") as pvalues_file:
        pvalues = sorted(pvalues_file.readlines())
    file_regression.check("\n".join(pvalues), extension="_pvalues.csv")
    fdr_filename = "%s/%s/%s.csv" % (output_dir, constants.HIERARCHICAL_FDR_DIR, constants.HIERARCHICAL_FDR_OUTPUTS)
    with open(fdr_filename, "r") as fdr_file:
        fdr = sorted(fdr_file.readlines())
    file_regression.check("\n".join(fdr), extension="_fdr.json")


def test_condor_simulation_clustering_hierarchy(file_regression, tmpdir):
    """Test simulation with clustering hierarchy"""
    func_name = sys._getframe().f_code.co_name
    output_dir = "%s/output_dir_%s" % (tmpdir, func_name)
    pvalues_filename = "%s/%s" % (output_dir, constants.PVALUES_FILENAME)
    cmd = ("python -m mihifepe.simulation -seed 2 -num_instances 100 -num_features 10 -fraction_relevant_features 0.5"
           " -contiguous_node_names -hierarchy_type cluster_from_data -perturbation zeroing -output_dir %s -condor" % output_dir)
    subprocess.check_call(cmd, shell=True)
    with open(pvalues_filename, "r") as pvalues_file:
        pvalues = sorted(pvalues_file.readlines())
    file_regression.check("\n".join(pvalues), extension="_pvalues.csv")
    fdr_filename = "%s/%s/%s.csv" % (output_dir, constants.HIERARCHICAL_FDR_DIR, constants.HIERARCHICAL_FDR_OUTPUTS)
    with open(fdr_filename, "r") as fdr_file:
        fdr = sorted(fdr_file.readlines())
    file_regression.check("\n".join(fdr), extension="_fdr.json")


def test_condor_simulation_shuffling_perturbation(file_regression, tmpdir):
    """Test simulation with clustering hierarchy"""
    func_name = sys._getframe().f_code.co_name
    output_dir = "%s/output_dir_%s" % (tmpdir, func_name)
    pvalues_filename = "%s/%s" % (output_dir, constants.PVALUES_FILENAME)
    cmd = ("python -m mihifepe.simulation -seed 3 -num_instances 100 -num_features 10 -fraction_relevant_features 0.5"
           " -contiguous_node_names -hierarchy_type random -perturbation shuffling -num_shuffling_trials 10 -output_dir %s -condor" % output_dir)
    subprocess.check_call(cmd, shell=True)
    with open(pvalues_filename, "r") as pvalues_file:
        pvalues = sorted(pvalues_file.readlines())
    file_regression.check("\n".join(pvalues), extension="_pvalues.csv")
    fdr_filename = "%s/%s/%s.csv" % (output_dir, constants.HIERARCHICAL_FDR_DIR, constants.HIERARCHICAL_FDR_OUTPUTS)
    with open(fdr_filename, "r") as fdr_file:
        fdr = sorted(fdr_file.readlines())
    file_regression.check("\n".join(fdr), extension="_fdr.json")


def test_condor_simulation_gaussian_noise(file_regression, tmpdir):
    """Test simulation with clustering hierarchy"""
    func_name = sys._getframe().f_code.co_name
    output_dir = "%s/output_dir_%s" % (tmpdir, func_name)
    pvalues_filename = "%s/%s" % (output_dir, constants.PVALUES_FILENAME)
    cmd = ("python -m mihifepe.simulation -seed 4 -num_instances 100 -num_features 10 -fraction_relevant_features 0.5"
           " -contiguous_node_names -hierarchy_type random -perturbation zeroing -noise_multiplier 0.1 -noise_type additive_gaussian"
           " -output_dir %s -condor" % output_dir)
    subprocess.check_call(cmd, shell=True)
    with open(pvalues_filename, "r") as pvalues_file:
        pvalues = sorted(pvalues_file.readlines())
    file_regression.check("\n".join(pvalues), extension="_pvalues.csv")
    fdr_filename = "%s/%s/%s.csv" % (output_dir, constants.HIERARCHICAL_FDR_DIR, constants.HIERARCHICAL_FDR_OUTPUTS)
    with open(fdr_filename, "r") as fdr_file:
        fdr = sorted(fdr_file.readlines())
    file_regression.check("\n".join(fdr), extension="_fdr.json")


def test_condor_simulation_interactions(file_regression, tmpdir):
    """Test simulation with interactions"""
    func_name = sys._getframe().f_code.co_name
    output_dir = "%s/output_dir_%s" % (tmpdir, func_name)
    pvalues_filename = "%s/%s" % (output_dir, constants.INTERACTIONS_PVALUES_FILENAME)
    cmd = ("python -m mihifepe.simulation -seed 5 -num_instances 100 -num_features 10 -fraction_relevant_features 0.5"
           " -analyze_interactions -hierarchy_type random -perturbation zeroing -noise_multiplier 0.0 -noise_type additive_gaussian"
           " -num_interactions 3 -output_dir %s -condor" % output_dir)
    subprocess.check_call(cmd, shell=True)
    with open(pvalues_filename, "r") as pvalues_file:
        pvalues = sorted(pvalues_file.readlines())
    file_regression.check("\n".join(pvalues), extension="_pvalues.csv")
    fdr_filename = "%s/%s/%s.csv" % (output_dir, constants.INTERACTIONS_FDR_DIR, constants.HIERARCHICAL_FDR_OUTPUTS)
    with open(fdr_filename, "r") as fdr_file:
        fdr = sorted(fdr_file.readlines())
    file_regression.check("\n".join(fdr), extension="_fdr.json")


def test_condor_simulation_all_pairwise_interactions(file_regression, tmpdir):
    """Test simulation with all pairwise (leaf)interactions"""
    func_name = sys._getframe().f_code.co_name
    output_dir = "%s/output_dir_%s" % (tmpdir, func_name)
    pvalues_filename = "%s/%s" % (output_dir, constants.INTERACTIONS_PVALUES_FILENAME)
    cmd = ("python -m mihifepe.simulation -seed 5 -num_instances 100 -num_features 10 -fraction_relevant_features 0.5"
           " -analyze_interactions -hierarchy_type random -perturbation zeroing -noise_multiplier 0.0 -noise_type additive_gaussian"
           " -num_interactions 3 -output_dir %s -analyze_all_pairwise_interactions -condor" % output_dir)
    subprocess.check_call(cmd, shell=True)
    with open(pvalues_filename, "r") as pvalues_file:
        pvalues = sorted(pvalues_file.readlines())
    file_regression.check("\n".join(pvalues), extension="_pvalues.csv")
    fdr_filename = "%s/%s/%s.csv" % (output_dir, constants.INTERACTIONS_FDR_DIR, constants.HIERARCHICAL_FDR_OUTPUTS)
    with open(fdr_filename, "r") as fdr_file:
        fdr = sorted(fdr_file.readlines())
    file_regression.check("\n".join(fdr), extension="_fdr.json")
