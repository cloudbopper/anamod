"""Tests for baseline explainers"""
import csv
import logging
import sys
from unittest.mock import patch

from anamod.baselines import explain_model, run_baselines
from tests.utils import pre_test, write_logfile


def check_metrics(data_regression, metrics_filename):
    """Verify metrics match gold files"""
    with open(metrics_filename, "r") as metrics_file:
        reader = csv.DictReader(metrics_file)
        row = next(reader)
        data = {field: row[field] for field in reader.fieldnames if "Runtime" not in field}
        if "sage" not in metrics_filename:
            # SAGE results non-deterministic
            data_regression.check(data)


# pylint: disable = protected-access, invalid-name
def test_simulation_baseline_anamod(data_regression, tmpdir, caplog, shared_fs):
    """Test anamod baseline explainer"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    cmd = (f"python -m anamod.baselines.run_baselines -condor 0 -shared_filesystem {shared_fs}"
           f" -start_seed 200000 -config demo -explainer anamod -output_dir {output_dir}")
    logging.getLogger().info(f"Cmd: {cmd}")
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        run_baselines.main()
    write_logfile(caplog, output_dir)
    metrics_filename = f"{output_dir}/{explain_model.EXPLAINER_EVALUATION_FILENAME}"
    check_metrics(data_regression, metrics_filename)


def test_simulation_baseline_anamod_all(data_regression, tmpdir, caplog, shared_fs):
    """Test anamod baseline explainer using -evaluate_all_nonzeros"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    cmd = (f"python -m anamod.baselines.run_baselines -condor 0 -shared_filesystem {shared_fs}"
           f" -start_seed 200000 -evaluate_all_nonzeros 1 -config demo -explainer anamod -output_dir {output_dir}")
    logging.getLogger().info(f"Cmd: {cmd}")
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        run_baselines.main()
    write_logfile(caplog, output_dir)
    metrics_filename = f"{output_dir}/{explain_model.EXPLAINER_EVALUATION_FILENAME}"
    check_metrics(data_regression, metrics_filename)


def test_simulation_baseline_lime(data_regression, tmpdir, caplog, shared_fs):
    """Test lime baseline explainer"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    cmd = (f"python -m anamod.baselines.run_baselines -condor 0 -shared_filesystem {shared_fs}"
           f" -start_seed 200000 -config demo -explainer lime -output_dir {output_dir}")
    logging.getLogger().info(f"Cmd: {cmd}")
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        run_baselines.main()
    write_logfile(caplog, output_dir)
    metrics_filename = f"{output_dir}/{explain_model.EXPLAINER_EVALUATION_FILENAME}"
    check_metrics(data_regression, metrics_filename)


def test_simulation_baseline_sage(data_regression, tmpdir, caplog, shared_fs):
    """Test sage baseline explainer"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    cmd = (f"python -m anamod.baselines.run_baselines -condor 0 -shared_filesystem {shared_fs}"
           f" -start_seed 200000 -config demo -explainer sage -output_dir {output_dir}")
    logging.getLogger().info(f"Cmd: {cmd}")
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        run_baselines.main()
    write_logfile(caplog, output_dir)
    metrics_filename = f"{output_dir}/{explain_model.EXPLAINER_EVALUATION_FILENAME}"
    check_metrics(data_regression, metrics_filename)


def test_simulation_baseline_sage_mean(data_regression, tmpdir, caplog, shared_fs):
    """Test sage-mean baseline explainer"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    cmd = (f"python -m anamod.baselines.run_baselines -condor 0 -shared_filesystem {shared_fs}"
           f" -start_seed 200000 -config demo -explainer sage-mean -output_dir {output_dir}")
    logging.getLogger().info(f"Cmd: {cmd}")
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        run_baselines.main()
    write_logfile(caplog, output_dir)
    metrics_filename = f"{output_dir}/{explain_model.EXPLAINER_EVALUATION_FILENAME}"
    check_metrics(data_regression, metrics_filename)


def test_simulation_baseline_sage_zero(data_regression, tmpdir, caplog, shared_fs):
    """Test sage-zero baseline explainer"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    cmd = (f"python -m anamod.baselines.run_baselines -condor 0 -shared_filesystem {shared_fs}"
           f" -start_seed 200000 -config demo -explainer sage-zero -output_dir {output_dir}")
    logging.getLogger().info(f"Cmd: {cmd}")
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        run_baselines.main()
    write_logfile(caplog, output_dir)
    metrics_filename = f"{output_dir}/{explain_model.EXPLAINER_EVALUATION_FILENAME}"
    check_metrics(data_regression, metrics_filename)


def test_simulation_baseline_perm(data_regression, tmpdir, caplog, shared_fs):
    """Test perm baseline explainer"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    cmd = (f"python -m anamod.baselines.run_baselines -condor 0 -shared_filesystem {shared_fs}"
           f" -start_seed 200000 -config demo -explainer perm -output_dir {output_dir}")
    logging.getLogger().info(f"Cmd: {cmd}")
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        run_baselines.main()
    write_logfile(caplog, output_dir)
    metrics_filename = f"{output_dir}/{explain_model.EXPLAINER_EVALUATION_FILENAME}"
    check_metrics(data_regression, metrics_filename)


def test_simulation_baseline_perm_fdr(data_regression, tmpdir, caplog, shared_fs):
    """Test perm-fdr baseline explainer"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    # First run PERM, then PERM-FDR (uses PERM results)
    cmd = (f"python -m anamod.baselines.run_baselines -condor 0 -shared_filesystem {shared_fs}"
           f" -start_seed 200000 -config demo -explainer perm -output_dir {output_dir}")
    logging.getLogger().info(f"Cmd: {cmd}")
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        run_baselines.main()
    # Run PERM-FDR in the same output directory
    cmd.replace("perm", "perm-fdr", 1)
    logging.getLogger().info(f"Cmd: {cmd}")
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        run_baselines.main()
    write_logfile(caplog, output_dir)
    metrics_filename = f"{output_dir}/{explain_model.EXPLAINER_EVALUATION_FILENAME}"
    check_metrics(data_regression, metrics_filename)


def test_simulation_baseline_occlusion_zero(data_regression, tmpdir, caplog, shared_fs):
    """Test occlusion-zero baseline explainer"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    cmd = (f"python -m anamod.baselines.run_baselines -condor 0 -shared_filesystem {shared_fs}"
           f" -start_seed 200000 -config demo -explainer occlusion-zero -output_dir {output_dir}")
    logging.getLogger().info(f"Cmd: {cmd}")
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        run_baselines.main()
    write_logfile(caplog, output_dir)
    metrics_filename = f"{output_dir}/{explain_model.EXPLAINER_EVALUATION_FILENAME}"
    check_metrics(data_regression, metrics_filename)


def test_simulation_baseline_occlusion_uniform(data_regression, tmpdir, caplog, shared_fs):
    """Test occlusion-uniform baseline explainer"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    cmd = (f"python -m anamod.baselines.run_baselines -condor 0 -shared_filesystem {shared_fs}"
           f" -start_seed 200002 -config demo -explainer occlusion-uniform -output_dir {output_dir}")
    logging.getLogger().info(f"Cmd: {cmd}")
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        run_baselines.main()
    write_logfile(caplog, output_dir)
    metrics_filename = f"{output_dir}/{explain_model.EXPLAINER_EVALUATION_FILENAME}"
    check_metrics(data_regression, metrics_filename)


def test_simulation_baseline_cxplain(data_regression, tmpdir, caplog, shared_fs):
    """Test cxplain baseline explainer"""
    func_name = sys._getframe().f_code.co_name
    output_dir = pre_test(func_name, tmpdir, caplog)
    cmd = (f"python -m anamod.baselines.run_baselines -condor 0 -shared_filesystem {shared_fs}"
           f" -start_seed 200000 -config demo -explainer cxplain -output_dir {output_dir}")
    logging.getLogger().info(f"Cmd: {cmd}")
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        run_baselines.main()
    write_logfile(caplog, output_dir)
    metrics_filename = f"{output_dir}/{explain_model.EXPLAINER_EVALUATION_FILENAME}"
    check_metrics(data_regression, metrics_filename)
