"""Tests for `anamod` package."""

import sys
from unittest.mock import patch

from anamod.simulation import simulation
from tests.utils import pre_test, post_test


# pylint: disable = invalid-name, protected-access
def test_condor_simulation_random_hierarchy(file_regression, tmpdir, caplog):
    """Test simulation with random hierarchy"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    cmd = ("python -m anamod.simulation -condor 1 -memory_requirement 1 -disk_requirement 1"
           " -seed 1 -num_instances 100 -num_features 10 -fraction_relevant_features 0.5"
           " -contiguous_node_names 1 -hierarchy_type random -perturbation zeroing -condor_cleanup 0 -output_dir %s" % output_dir)
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        simulation.main()
    post_test(file_regression, caplog, output_dir)


def test_condor_simulation_clustering_hierarchy(file_regression, tmpdir, caplog):
    """Test simulation with clustering hierarchy"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    cmd = ("python -m anamod.simulation -condor 1 -memory_requirement 1 -disk_requirement 1"
           " -seed 2 -num_instances 100 -num_features 10 -fraction_relevant_features 0.5"
           " -contiguous_node_names 1 -hierarchy_type cluster_from_data -perturbation zeroing -condor_cleanup 0 -output_dir %s" % output_dir)
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        simulation.main()
    post_test(file_regression, caplog, output_dir)


def test_condor_simulation_shuffling_perturbation(file_regression, tmpdir, caplog):
    """Test simulation with clustering hierarchy"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    cmd = ("python -m anamod.simulation -condor 1 -memory_requirement 1 -disk_requirement 1"
           " -seed 3 -num_instances 100 -num_features 10 -fraction_relevant_features 0.5"
           " -contiguous_node_names 1 -hierarchy_type random -perturbation shuffling -num_shuffling_trials 10"
           " -condor_cleanup 0 -output_dir %s" % output_dir)
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        simulation.main()
    post_test(file_regression, caplog, output_dir)


def test_condor_simulation_gaussian_noise(file_regression, tmpdir, caplog):
    """Test simulation with clustering hierarchy"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    cmd = ("python -m anamod.simulation -condor 1 -memory_requirement 1 -disk_requirement 1"
           " -seed 4 -num_instances 100 -num_features 10 -fraction_relevant_features 0.5"
           " -contiguous_node_names 1 -hierarchy_type random -perturbation zeroing -noise_multiplier 0.1 -noise_type additive_gaussian"
           " -condor_cleanup 0 -output_dir %s" % output_dir)
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        simulation.main()
    post_test(file_regression, caplog, output_dir)


def test_condor_simulation_interactions(file_regression, tmpdir, caplog):
    """Test simulation with interactions"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    cmd = ("python -m anamod.simulation -condor 1 -memory_requirement 1 -disk_requirement 1"
           " -seed 9 -num_instances 100 -num_features 10 -fraction_relevant_features 0.5"
           " -analyze_interactions 1 -hierarchy_type random -perturbation zeroing -noise_type none"
           " -num_interactions 3 -condor_cleanup 0 -output_dir %s" % output_dir)
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        simulation.main()
    post_test(file_regression, caplog, output_dir, interactions=True)


def test_condor_simulation_all_pairwise_interactions(file_regression, tmpdir, caplog):
    """Test simulation with all pairwise (leaf)interactions"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    cmd = ("python -m anamod.simulation -condor 1 -memory_requirement 1 -disk_requirement 1"
           " -seed 9 -num_instances 100 -num_features 10 -fraction_relevant_features 0.5"
           " -analyze_interactions 1 -hierarchy_type random -perturbation zeroing -noise_multiplier 0.0 -noise_type additive_gaussian"
           " -num_interactions 3 -condor_cleanup 0 -output_dir %s -analyze_all_pairwise_interactions 1" % output_dir)
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        simulation.main()
    post_test(file_regression, caplog, output_dir, interactions=True)


def test_condor_simulation_noisy_interactions(file_regression, tmpdir, caplog):
    """Test simulation with interactions and noisy model"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    cmd = ("python -m anamod.simulation -condor 1 -memory_requirement 1 -disk_requirement 1"
           " -seed 9 -num_instances 100 -num_features 10 -fraction_relevant_features 0.5"
           " -analyze_interactions 1 -hierarchy_type random -perturbation zeroing -noise_multiplier 0.1 -noise_type additive_gaussian"
           " -num_interactions 3 -condor_cleanup 0 -output_dir %s" % output_dir)
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        simulation.main()
    post_test(file_regression, caplog, output_dir, interactions=True)
