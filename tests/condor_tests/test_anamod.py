"""Tests for `anamod` package."""

import logging
import sys
from unittest.mock import patch

from anamod import constants
from anamod.simulation import simulation

# pylint: disable = invalid-name, redefined-outer-name, protected-access


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


def test_condor_simulation_random_hierarchy(file_regression, tmpdir, caplog):
    """Test simulation with random hierarchy"""
    func_name = sys._getframe().f_code.co_name
    setup_logfile(caplog)
    output_dir = "%s/output_dir_%s" % (tmpdir, func_name)
    pvalues_filename = "%s/%s" % (output_dir, constants.PVALUES_FILENAME)
    cmd = ("python -m anamod.simulation -condor -seed 1 -num_instances 100 -num_features 10 -fraction_relevant_features 0.5"
           " -contiguous_node_names -hierarchy_type random -perturbation zeroing -no-condor-cleanup -output_dir %s" % output_dir)
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        simulation.main()
    write_logfile(caplog, output_dir)
    with open(pvalues_filename, "r") as pvalues_file:
        pvalues = sorted(pvalues_file.readlines())
    file_regression.check("\n".join(pvalues), extension="_pvalues.csv")
    fdr_filename = "%s/%s/%s.csv" % (output_dir, constants.HIERARCHICAL_FDR_DIR, constants.HIERARCHICAL_FDR_OUTPUTS)
    with open(fdr_filename, "r") as fdr_file:
        fdr = sorted(fdr_file.readlines())
    file_regression.check("\n".join(fdr), extension="_fdr.json")


def test_condor_simulation_clustering_hierarchy(file_regression, tmpdir, caplog):
    """Test simulation with clustering hierarchy"""
    func_name = sys._getframe().f_code.co_name
    setup_logfile(caplog)
    output_dir = "%s/output_dir_%s" % (tmpdir, func_name)
    pvalues_filename = "%s/%s" % (output_dir, constants.PVALUES_FILENAME)
    cmd = ("python -m anamod.simulation -condor -seed 2 -num_instances 100 -num_features 10 -fraction_relevant_features 0.5"
           " -contiguous_node_names -hierarchy_type cluster_from_data -perturbation zeroing -no-condor-cleanup -output_dir %s" % output_dir)
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        simulation.main()
    write_logfile(caplog, output_dir)
    with open(pvalues_filename, "r") as pvalues_file:
        pvalues = sorted(pvalues_file.readlines())
    file_regression.check("\n".join(pvalues), extension="_pvalues.csv")
    fdr_filename = "%s/%s/%s.csv" % (output_dir, constants.HIERARCHICAL_FDR_DIR, constants.HIERARCHICAL_FDR_OUTPUTS)
    with open(fdr_filename, "r") as fdr_file:
        fdr = sorted(fdr_file.readlines())
    file_regression.check("\n".join(fdr), extension="_fdr.json")


def test_condor_simulation_shuffling_perturbation(file_regression, tmpdir, caplog):
    """Test simulation with clustering hierarchy"""
    func_name = sys._getframe().f_code.co_name
    setup_logfile(caplog)
    output_dir = "%s/output_dir_%s" % (tmpdir, func_name)
    pvalues_filename = "%s/%s" % (output_dir, constants.PVALUES_FILENAME)
    cmd = ("python -m anamod.simulation -condor -seed 3 -num_instances 100 -num_features 10 -fraction_relevant_features 0.5"
           " -contiguous_node_names -hierarchy_type random -perturbation shuffling -num_shuffling_trials 10 -output_dir %s" % output_dir)
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        simulation.main()
    write_logfile(caplog, output_dir)
    with open(pvalues_filename, "r") as pvalues_file:
        pvalues = sorted(pvalues_file.readlines())
    file_regression.check("\n".join(pvalues), extension="_pvalues.csv")
    fdr_filename = "%s/%s/%s.csv" % (output_dir, constants.HIERARCHICAL_FDR_DIR, constants.HIERARCHICAL_FDR_OUTPUTS)
    with open(fdr_filename, "r") as fdr_file:
        fdr = sorted(fdr_file.readlines())
    file_regression.check("\n".join(fdr), extension="_fdr.json")


def test_condor_simulation_gaussian_noise(file_regression, tmpdir, caplog):
    """Test simulation with clustering hierarchy"""
    func_name = sys._getframe().f_code.co_name
    setup_logfile(caplog)
    output_dir = "%s/output_dir_%s" % (tmpdir, func_name)
    pvalues_filename = "%s/%s" % (output_dir, constants.PVALUES_FILENAME)
    cmd = ("python -m anamod.simulation -condor -seed 4 -num_instances 100 -num_features 10 -fraction_relevant_features 0.5"
           " -contiguous_node_names -hierarchy_type random -perturbation zeroing -noise_multiplier 0.1 -noise_type additive_gaussian"
           " -output_dir %s" % output_dir)
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        simulation.main()
    write_logfile(caplog, output_dir)
    with open(pvalues_filename, "r") as pvalues_file:
        pvalues = sorted(pvalues_file.readlines())
    file_regression.check("\n".join(pvalues), extension="_pvalues.csv")
    fdr_filename = "%s/%s/%s.csv" % (output_dir, constants.HIERARCHICAL_FDR_DIR, constants.HIERARCHICAL_FDR_OUTPUTS)
    with open(fdr_filename, "r") as fdr_file:
        fdr = sorted(fdr_file.readlines())
    file_regression.check("\n".join(fdr), extension="_fdr.json")


def test_condor_simulation_interactions(file_regression, tmpdir, caplog):
    """Test simulation with interactions"""
    func_name = sys._getframe().f_code.co_name
    setup_logfile(caplog)
    output_dir = "%s/output_dir_%s" % (tmpdir, func_name)
    pvalues_filename = "%s/%s" % (output_dir, constants.INTERACTIONS_PVALUES_FILENAME)
    cmd = ("python -m anamod.simulation -condor -seed 9 -num_instances 100 -num_features 10 -fraction_relevant_features 0.5"
           " -analyze_interactions -hierarchy_type random -perturbation zeroing -noise_type none"
           " -num_interactions 3 -output_dir %s" % output_dir)
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        simulation.main()
    write_logfile(caplog, output_dir)
    with open(pvalues_filename, "r") as pvalues_file:
        pvalues = sorted(pvalues_file.readlines())
    file_regression.check("\n".join(pvalues), extension="_pvalues.csv")
    fdr_filename = "%s/%s/%s.csv" % (output_dir, constants.INTERACTIONS_FDR_DIR, constants.HIERARCHICAL_FDR_OUTPUTS)
    with open(fdr_filename, "r") as fdr_file:
        fdr = sorted(fdr_file.readlines())
    file_regression.check("\n".join(fdr), extension="_fdr.json")


def test_condor_simulation_all_pairwise_interactions(file_regression, tmpdir, caplog):
    """Test simulation with all pairwise (leaf)interactions"""
    func_name = sys._getframe().f_code.co_name
    setup_logfile(caplog)
    output_dir = "%s/output_dir_%s" % (tmpdir, func_name)
    pvalues_filename = "%s/%s" % (output_dir, constants.INTERACTIONS_PVALUES_FILENAME)
    cmd = ("python -m anamod.simulation -condor -seed 9 -num_instances 100 -num_features 10 -fraction_relevant_features 0.5"
           " -analyze_interactions -hierarchy_type random -perturbation zeroing -noise_multiplier 0.0 -noise_type additive_gaussian"
           " -num_interactions 3 -output_dir %s -analyze_all_pairwise_interactions" % output_dir)
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        simulation.main()
    write_logfile(caplog, output_dir)
    with open(pvalues_filename, "r") as pvalues_file:
        pvalues = sorted(pvalues_file.readlines())
    file_regression.check("\n".join(pvalues), extension="_pvalues.csv")
    fdr_filename = "%s/%s/%s.csv" % (output_dir, constants.INTERACTIONS_FDR_DIR, constants.HIERARCHICAL_FDR_OUTPUTS)
    with open(fdr_filename, "r") as fdr_file:
        fdr = sorted(fdr_file.readlines())
    file_regression.check("\n".join(fdr), extension="_fdr.json")


def test_condor_simulation_noisy_interactions(file_regression, tmpdir, caplog):
    """Test simulation with interactions and noisy model"""
    func_name = sys._getframe().f_code.co_name
    setup_logfile(caplog)
    output_dir = "%s/output_dir_%s" % (tmpdir, func_name)
    pvalues_filename = "%s/%s" % (output_dir, constants.INTERACTIONS_PVALUES_FILENAME)
    cmd = ("python -m anamod.simulation -condor -seed 9 -num_instances 100 -num_features 10 -fraction_relevant_features 0.5"
           " -analyze_interactions -hierarchy_type random -perturbation zeroing -noise_multiplier 0.1 -noise_type additive_gaussian"
           " -num_interactions 3 -output_dir %s" % output_dir)
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        simulation.main()
    write_logfile(caplog, output_dir)
    with open(pvalues_filename, "r") as pvalues_file:
        pvalues = sorted(pvalues_file.readlines())
    file_regression.check("\n".join(pvalues), extension="_pvalues.csv")
    fdr_filename = "%s/%s/%s.csv" % (output_dir, constants.INTERACTIONS_FDR_DIR, constants.HIERARCHICAL_FDR_OUTPUTS)
    with open(fdr_filename, "r") as fdr_file:
        fdr = sorted(fdr_file.readlines())
    file_regression.check("\n".join(fdr), extension="_fdr.json")


def test_condor_simulation_shuffling_interactions(file_regression, tmpdir, caplog):
    """Test simulation with interactions and shuffling perturbations"""
    func_name = sys._getframe().f_code.co_name
    setup_logfile(caplog)
    output_dir = "%s/output_dir_%s" % (tmpdir, func_name)
    pvalues_filename = "%s/%s" % (output_dir, constants.INTERACTIONS_PVALUES_FILENAME)
    cmd = ("python -m anamod.simulation -condor -seed 9 -num_instances 1000 -num_features 10 -fraction_relevant_features 0.5"
           " -analyze_interactions -hierarchy_type random -perturbation shuffling -num_shuffling_trials 200"
           " -noise_multiplier 0.0 -noise_type additive_gaussian -num_interactions 3 -output_dir %s" % output_dir)
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        simulation.main()
    write_logfile(caplog, output_dir)
    with open(pvalues_filename, "r") as pvalues_file:
        pvalues = sorted(pvalues_file.readlines())
    file_regression.check("\n".join(pvalues), extension="_pvalues.csv")
    fdr_filename = "%s/%s/%s.csv" % (output_dir, constants.INTERACTIONS_FDR_DIR, constants.HIERARCHICAL_FDR_OUTPUTS)
    with open(fdr_filename, "r") as fdr_file:
        fdr = sorted(fdr_file.readlines())
    file_regression.check("\n".join(fdr), extension="_fdr.json")
