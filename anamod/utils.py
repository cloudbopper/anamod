"""Common utility functions"""
import logging
from collections import namedtuple
import contextlib
import os
import sys
import time

import numpy as np
try:
    import htcondor
    from htcondor import JobEventType, JobAction
except ImportError:
    pass  # Caller performs its own check to validate condor availability

from anamod.constants import VIRTUAL_ENV


def get_logger(name, filename, level=logging.INFO):
    """Return logger configure to write to filename"""
    formatting = "%(asctime)s: %(levelname)s: %(name)s: %(message)s"
    logging.basicConfig(level=level, filename=filename, format=formatting)  # if not already configured
    logger = logging.getLogger(name)
    return logger


def round_value(value, decimals=4):
    """Round input to avoid floating-point errors"""
    return np.around(value, decimals=decimals)


Filenames = namedtuple("Filenames", ["exec_filename", "log_filename", "out_filename", "err_filename"])


class CondorJobWrapper():
    """Schedule jobs using condor"""
    # pylint: disable = too-many-instance-attributes
    idx = 0

    def __init__(self, cmd, input_files, job_dir, **kwargs):
        """
        TODO: Improve documentation
        Common considerations:
        - job_dir should be an empty directory for condor files and job outputs, not the current working directory
        If shared_filesystem is disabled:
        - input files will be transferred to compute server
        - input file paths need to be accessible from submit server
        - cmd has access to input files/directory within the working directory on the compute server
        - job_dir will be transferred back from compute server
        - software will be downloaded and installed from the github package
        If shared_filesystem is enabled:
        - no file transfers or differing file path assumptions for submit/compute servers
        - cmd has access to absolute file/directory paths
        - assumes code running inside virtualenv
        """
        # TODO: unified list of kwargs
        self.name = f"job_{CondorJobWrapper.idx}"
        CondorJobWrapper.idx += 1
        self.cmd = cmd
        self.input_files = input_files + [f"http://proxy.chtc.wisc.edu/SQUID/chtc/python3{sys.version_info.minor}.tar.gz"]  # List of input files
        self.job_dir = job_dir  # Directory for job logs/outputs in submit host
        if not os.path.exists(self.job_dir):
            os.makedirs(self.job_dir)
        self.job_dir_remote = os.path.basename(self.job_dir.rstrip("/"))
        self.shared_filesystem = kwargs.get("shared_filesystem", False)
        self.filenames = Filenames(exec_filename=f"{self.job_dir}/{self.name}.sh",
                                   log_filename=f"{self.job_dir}/{self.name}.log",
                                   out_filename=f"{self.job_dir}/{self.name}.out",
                                   err_filename=f"{self.job_dir}/{self.name}.err")
        self.job = self.create_job(**kwargs)
        self.cluster_id = -1  # set by running job

    def create_job(self, **kwargs):
        """Create job"""
        self.create_executable()
        memory = kwargs.get("memory", "1GB")
        disk = kwargs.get("disk", "4GB")
        job = htcondor.Submit({"initialdir": f"{self.job_dir}",
                               "executable": f"{self.filenames.exec_filename}",
                               "output": f"{self.filenames.out_filename}",
                               "error": f"{self.filenames.err_filename}",
                               "log": f"{self.filenames.log_filename}",
                               "request_memory": f"{memory}",
                               "request_disk": f"{disk}",
                               "universe": "vanilla",
                               "should_transfer_files": "NO" if self.shared_filesystem else "YES",
                               "transfer_input_files": ",".join(self.input_files),
                               "transfer_output_files": f"{self.job_dir_remote}/"
                               })
        return job

    def create_executable(self):
        """Create executable shell script"""
        with open(self.filenames.exec_filename, "w") as exec_file:
            # Setup environment and inputs
            exec_file.write("#!/bin/sh\n")
            if not self.shared_filesystem:
                # TODO: make more general by reading package from argument
                exec_file.write(f"mkdir {self.job_dir_remote}\n"
                                "tar -xzf python38.tar.gz\n"
                                "export PATH=${PWD}/python/bin/:${PATH}\n"
                                "export PYTHONPATH=${PWD}/packages\n"
                                "export LC_ALL=en_US.UTF-8\n"
                                "python3 -m pip install --upgrade pip\n"
                                "python3 -m pip install git+https://github.com/cloudbopper/anamod@condor_nonshared --target ${PWD}/packages\n")
            else:
                virtualenv = os.environ.get(VIRTUAL_ENV, "")
                exec_file.write(f"source ${virtualenv}/bin/activate")
            # Execute command
            exec_file.write(f"{self.cmd}\n")

    def run(self):
        """Run job"""
        # Remove log file since it's used for tracking job progress
        with contextlib.suppress(OSError):
            os.remove(self.filenames.log_filename)
        # Run job
        schedd = htcondor.Schedd()
        with schedd.transaction() as txn:
            self.cluster_id = self.job.queue(txn)

    def cleanup(self, **kwargs):
        """Clean up intermediate files generated by job"""
        if not kwargs.get("cleanup", False):
            return
        with contextlib.suppress(OSError):
            for filename in self.filenames:
                os.remove(filename)

    @staticmethod
    def monitor(jobs, **kwargs):
        """Monitor jobs"""
        # pylint: disable = too-many-nested-blocks
        events = [htcondor.JobEventLog(job.filenames.log_filename).events(0) for job in jobs]
        num_unfinished_jobs = len(jobs)
        job_finished = [False] * num_unfinished_jobs
        while num_unfinished_jobs > 0:
            for idx in filter(lambda idx: not job_finished[idx], range(len(jobs))):
                job = jobs[idx]
                for event in events[idx]:
                    # FIXME: handle jobs held due to condor errors - e.g. low memory
                    event_type = event.type
                    if event_type == JobEventType.JOB_TERMINATED:
                        if event["TerminatedNormally"]:
                            if event["ReturnValue"] != 0:
                                raise RuntimeError(f"Cmd: '{job.cmd}' terminated normally with non-zero return code - "
                                                   f"see error file: {job.filenames.err_filename}.")
                            job_finished[idx] = True
                            num_unfinished_jobs -= 1
                            job.cleanup()
                            break
                        # Terminated abnormally
                        CondorJobWrapper.remove_jobs(jobs, reason=f"Job {job.name} terminated abnormally")
                        raise RuntimeError(f"Cmd: '{job.cmd}' terminated abnormally - see log: {job.filenames.log_filename}.")
                    if (event_type == JobEventType.JOB_ABORTED
                            or event_type == JobEventType.SHADOW_EXCEPTION  # noqa: W503
                            or (event_type == JobEventType.JOB_EVICTED and not event["TerminatedNormally"])):  # noqa: W503
                        CondorJobWrapper.remove_jobs(jobs, reason=f"Job {job.name} failed to terminate")
                        raise RuntimeError(f"Cmd: '{job.cmd}' failed to terminate - see log: {job.filenames.log_filename}.")
            time.sleep(30)

    @staticmethod
    def remove_jobs(jobs, reason=None):
        """Remove jobs from condor queue"""
        schedd = htcondor.Schedd()
        for job in jobs:
            schedd.act(JobAction.Remove, f"ClusterId=={job.cluster_id}", reason=reason)
