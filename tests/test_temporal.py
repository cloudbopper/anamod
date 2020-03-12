"""Tests for temporal model analysis"""

import sys
from unittest.mock import patch

from anamod.simulation import simulation
from tests.utils import pre_test, write_logfile


# pylint: disable = protected-access, invalid-name
def test_simulation_regressor1(data_regression, tmpdir, caplog):
    """Test simulation with random hierarchy"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    cmd = ("python -m anamod.simulation -seed 100 -analysis_type temporal -num_instances 100 -num_features 10 "
           "-model_type regressor -num_shuffling_trials 10 "
           "-fraction_relevant_features 0.5 -condor_cleanup 0 -output_dir %s" % output_dir)
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        results = simulation.main()
    write_logfile(caplog, output_dir)
    data_regression.check(str(results))


def test_simulation_classifier1(data_regression, tmpdir, caplog):
    """Test simulation with random hierarchy"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    cmd = ("python -m anamod.simulation -seed 100 -analysis_type temporal -num_instances 100 -num_features 10 "
           "-model_type classifier -num_shuffling_trials 10 "
           "-fraction_relevant_features 0.5 -condor_cleanup 0 -output_dir %s" % output_dir)
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        results = simulation.main()
    write_logfile(caplog, output_dir)
    data_regression.check(str(results))
