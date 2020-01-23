"""Tests for temporal model analysis"""

import sys
from unittest.mock import patch

from anamod.simulation import simulation
from tests.utils import write_logfile, pre_test


# pylint: disable = protected-access
def test_simulation_basic1(tmpdir, caplog):
    """Test simulation with random hierarchy"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    cmd = ("python -m anamod.simulation -seed 100 -analysis_type temporal -num_instances 100 -num_features 10 "
           "-fraction_relevant_features 0.5 -no-condor-cleanup -output_dir %s" % output_dir)
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        simulation.main()
    write_logfile(caplog, output_dir)
