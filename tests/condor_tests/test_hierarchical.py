"""Tests for `anamod` package."""

import os
import sys
from unittest.mock import patch

import pytest

from anamod import model_loader
from anamod.simulation import simulation
from tests.utils import pre_test, post_test


# pylint: disable = invalid-name, protected-access
def test_condor_simulation_random_hierarchy(file_regression, tmpdir, caplog, shared_fs):
    """Test simulation with random hierarchy"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    cmd = ("python -m anamod.simulation -condor 1 -memory_requirement 1 -disk_requirement 1"
           " -seed 1 -num_instances 100 -num_features 30 -fraction_relevant_features 0.5 -noise_multiplier 0.1"
           " -analysis_type hierarchical -contiguous_node_names 1 -hierarchy_type random -perturbation shuffling -cleanup 0"
           f" -model_loader_filename {os.path.abspath(model_loader.__file__)}"
           f" -shared_filesystem {shared_fs} -output_dir {output_dir}")
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        simulation.main()
    post_test(file_regression, caplog, output_dir)


def test_condor_simulation_clustering_hierarchy(file_regression, tmpdir, caplog, shared_fs):
    """Test simulation with clustering hierarchy"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    cmd = ("python -m anamod.simulation -condor 1 -memory_requirement 1 -disk_requirement 1"
           " -seed 2 -num_instances 100 -num_features 30 -fraction_relevant_features 0.5 -noise_multiplier 0.1"
           " -analysis_type hierarchical -contiguous_node_names 1 -hierarchy_type cluster_from_data -perturbation shuffling -cleanup 0"
           f" -shared_filesystem {shared_fs} -output_dir {output_dir}")
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        simulation.main()
    post_test(file_regression, caplog, output_dir)


def test_condor_simulation_shuffling_perturbation(file_regression, tmpdir, caplog, shared_fs):
    """Test simulation with clustering hierarchy"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    cmd = ("python -m anamod.simulation -condor 1 -memory_requirement 1 -disk_requirement 1"
           " -seed 3 -num_instances 100 -num_features 30 -fraction_relevant_features 0.5 -noise_multiplier 0.1"
           " -contiguous_node_names 1 -hierarchy_type random -perturbation shuffling -num_shuffling_trials 10 -cleanup 0"
           f" -analysis_type hierarchical -shared_filesystem {shared_fs} -output_dir {output_dir}")
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        simulation.main()
    post_test(file_regression, caplog, output_dir)


@pytest.mark.skip(reason="Very low interactions power - need to verify theory works with permutations")
def test_condor_simulation_interactions(file_regression, tmpdir, caplog, shared_fs):
    """Test simulation with interactions"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    cmd = ("python -m anamod.simulation -condor 1 -memory_requirement 1 -disk_requirement 1"
           " -seed 9 -num_instances 100 -num_features 30 -fraction_relevant_features 0.5 -analysis_type hierarchical"
           " -analyze_interactions 1 -hierarchy_type random -perturbation shuffling -noise_multiplier 0.0"
           " -num_interactions 3 -cleanup 0"
           f" -shared_filesystem {shared_fs} -output_dir {output_dir}")
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        simulation.main()
    post_test(file_regression, caplog, output_dir, interactions=True)


@pytest.mark.skip(reason="Very low interactions power - need to verify theory works with permutations")
def test_condor_simulation_all_pairwise_interactions(file_regression, tmpdir, caplog, shared_fs):
    """Test simulation with all pairwise (leaf)interactions"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    cmd = ("python -m anamod.simulation -condor 1 -memory_requirement 1 -disk_requirement 1"
           " -seed 9 -num_instances 100 -num_features 30 -fraction_relevant_features 0.5 -analysis_type hierarchical"
           " -analyze_interactions 1 -hierarchy_type random -perturbation shuffling -noise_multiplier 0.0"
           " -num_interactions 3 -cleanup 0 -analyze_all_pairwise_interactions 1"
           f" -shared_filesystem {shared_fs} -output_dir {output_dir}")
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        simulation.main()
    post_test(file_regression, caplog, output_dir, interactions=True)


@pytest.mark.skip(reason="Very low interactions power - need to verify theory works with permutations")
def test_condor_simulation_noisy_interactions(file_regression, tmpdir, caplog, shared_fs):
    """Test simulation with interactions and noisy model"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    cmd = ("python -m anamod.simulation -condor 1 -memory_requirement 1 -disk_requirement 1"
           " -seed 9 -num_instances 100 -num_features 30 -fraction_relevant_features 0.5 -analysis_type hierarchical"
           " -analyze_interactions 1 -hierarchy_type random -perturbation shuffling -noise_multiplier 0.1"
           f" -model_loader_filename {os.path.abspath(model_loader.__file__)}"
           " -num_interactions 3 -cleanup 0"
           f" -shared_filesystem {shared_fs} -output_dir {output_dir}")
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        simulation.main()
    post_test(file_regression, caplog, output_dir, interactions=True)
