"""Tests for temporal model analysis"""

import os
import sys
from unittest.mock import patch

from anamod import model_loader
from anamod.simulation import simulation
from tests.utils import pre_test, write_logfile


# pylint: disable = protected-access, invalid-name
def test_condor_simulation_regressor1(data_regression, tmpdir, caplog, shared_fs):
    """Test simulation with regression model over temporal data"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    cmd = ("python -m anamod.simulation -condor 1 -memory_requirement 1 -disk_requirement 1"
           " -seed 100 -analysis_type temporal -num_instances 100 -num_features 10"
           " -model_type regressor -num_shuffling_trials 10"
           " -fraction_relevant_features 0.5 -cleanup 0"
           f" -shared_filesystem {shared_fs} -output_dir {output_dir}")
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        results = simulation.main()
    write_logfile(caplog, output_dir)
    data_regression.check(str(results))


def test_condor_simulation_classifier1(data_regression, tmpdir, caplog, shared_fs):
    """Test simulation with classification model over temporal data"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    cmd = ("python -m anamod.simulation -condor 1 -memory_requirement 1 -disk_requirement 1"
           " -seed 100 -analysis_type temporal -num_instances 100 -num_features 10"
           " -model_type classifier -num_shuffling_trials 10"
           f" -model_loader_filename {os.path.abspath(model_loader.__file__)}"
           " -fraction_relevant_features 0.5 -cleanup 0"
           f" -shared_filesystem {shared_fs} -output_dir {output_dir}")
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        results = simulation.main()
    write_logfile(caplog, output_dir)
    data_regression.check(str(results))