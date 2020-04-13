"""Common utility functions"""
import logging
from collections import namedtuple, OrderedDict
import contextlib
import os
import sys
import time

import numpy as np
try:
    # TODO: Add note about installing htcondor to documentation
    import htcondor
    from htcondor import JobAction, JobEventType
except ImportError:
    pass  # Caller performs its own check to validate condor availability

from anamod.constants import VIRTUAL_ENV, EVENT_LOG_TRACKING

CONDOR_MAX_RUNNING_TIME = 4 * 3600
CONDOR_MAX_WAIT_TIME = 300  # Time to wait for job to start running before retrying
CONDOR_MAX_RETRIES = 5
# Reference: https://htcondor.readthedocs.io/en/latest/classad-attributes/job-classad-attributes.html
CONDOR_HOLD_RETRY_CODES = set([6, 7, 8, 9, 10, 11, 12, 13, 14])


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


class CondorJobFailure(RuntimeError):
    """Exception for condor job failure"""


class CondorJobWrapper():
    """Schedule jobs using condor"""
    # pylint: disable = too-many-instance-attributes
    idx = 0

    def __init__(self, cmd, input_files, job_dir, **kwargs):
        """
        Creates htcondor job wrapper. Use 'run' to submit job and 'monitor' to monitor the status of submitted jobs.
        Args:
        * cmd: command to run on condor execute node
            * If non-shared FS, input file/directory paths supplied to cmd must be stripped of submit node directory structure
            * If shared FS, input file/directory paths supplied to cmd must be absolute paths, since cmd will be run from inside job_dir
       * input_files: list of input files for cmd (must be accessible from submit node)
            * If non-shared FS, these will be transferred to root working directory in execute node
        * job_dir: empty directory for condor logs (submit node) and job outputs (execute node) (will be created if it doesn't exist)
            * If non-shared FS, outputs in this directory will be transferred back to submit node from execute node
        Keyword args:
        * shared_filesystem: Flag to specify shared/non-shared FS
        * memory: amount of memory to request on condor execute node, default 1GB
        * disk: amount of disk storage to request on condor execute node, default 4GB
        * package: software package to install via pip on execute node, default cloudbopper/anamod (relevant for non-shared FS only)
        Other considerations:
        * If non-shared FS, software downloaded and installed in execute node from github package cloudbopper/anamod.git
        * If shared FS, assumes that the submit node code is running inside virtualenv and tries to activate this on execute node
        """
        # Distinguish jobs for monitoring
        self.name = f"job_{CondorJobWrapper.idx}"
        CondorJobWrapper.idx += 1
        # Set up job input files and working directory
        self.cmd = cmd
        self.input_files = ([os.path.abspath(input_file) for input_file in input_files])
        for input_file in self.input_files:
            assert os.path.exists(input_file)
        self.input_files += [f"http://proxy.chtc.wisc.edu/SQUID/chtc/python3{sys.version_info.minor}.tar.gz"]
        self.job_dir = os.path.abspath(job_dir)  # Directory for job logs/outputs in submit host
        if not os.path.exists(self.job_dir):
            os.makedirs(self.job_dir)
        self.job_dir_remote = os.path.basename(self.job_dir.rstrip("/"))
        self.filenames = Filenames(exec_filename=f"{self.job_dir}/{self.name}.sh",
                                   log_filename=f"{self.job_dir}/{self.name}.log",
                                   out_filename=f"{self.job_dir}/{self.name}.out",
                                   err_filename=f"{self.job_dir}/{self.name}.err")
        # Process keyword args
        self.shared_filesystem = kwargs.get("shared_filesystem", False)
        memory = kwargs.get("memory", "1GB")
        disk = kwargs.get("disk", "4GB")
        package = kwargs.get("package", "git+https://github.com/cloudbopper/anamod")
        # Create job
        self.job = self.create_job(memory, disk, package)
        # Set by running job:
        self.cluster_id = -1
        self.tries = 0
        self.running = False
        self.submit_time = -1
        self.execute_time = -1

    def create_job(self, memory, disk, package):
        """Create job"""
        self.create_executable(package)
        job = htcondor.Submit({"initialdir": f"{self.job_dir}",
                               "executable": f"{self.filenames.exec_filename}",
                               "output": f"{self.filenames.out_filename}",
                               "error": f"{self.filenames.err_filename}",
                               "log": f"{self.filenames.log_filename}",
                               "request_memory": f"{memory}",
                               "request_disk": f"{disk}",
                               "universe": "vanilla",
                               "should_transfer_files": "NO" if self.shared_filesystem else "YES",
                               "transfer_input_files": "" if self.shared_filesystem else ",".join(self.input_files),
                               "transfer_output_files": "" if self.shared_filesystem else f"{self.job_dir_remote}/",
                               # Send the job to Held state on external failures
                               "on_exit_hold": "ExitBySignal == true",
                               # Periodically retry the jobs every 10 minutes, up to a maximum of 5 retries.
                               "periodic_release": "(NumJobStarts < 5) && ((CurrentTime - EnteredCurrentStatus) > 600)"
                               })
        return job

    def create_executable(self, package):
        """Create executable shell script"""
        with open(self.filenames.exec_filename, "w") as exec_file:
            # Setup environment and inputs
            exec_file.write("#!/bin/sh\n")
            if not self.shared_filesystem:
                exec_file.write(f"mkdir {self.job_dir_remote}\n"
                                f"tar -xzf python3{sys.version_info.minor}.tar.gz\n"
                                "export PATH=${PWD}/python/bin/:${PATH}\n"
                                "export PYTHONPATH=${PWD}/packages\n"
                                "export LC_ALL=en_US.UTF-8\n"
                                "python3 -m pip install --upgrade pip\n"
                                f"python3 -m pip install {package} --target ${{PWD}}/packages\n")
            else:
                virtualenv = os.environ.get(VIRTUAL_ENV, "")
                exec_file.write(f"source {virtualenv}/bin/activate\n")
            # Execute command
            exec_file.write(f"{self.cmd}\n")
        os.chmod(self.filenames.exec_filename, 0o777)

    def run(self):
        """Run job"""
        # Remove log file since it's used for tracking job progress
        if os.path.exists(self.filenames.log_filename):
            os.replace(self.filenames.log_filename, f"{self.filenames.log_filename}.{self.tries}")
        # Run job
        schedd = htcondor.Schedd()
        with schedd.transaction() as txn:
            self.cluster_id = self.job.queue(txn)
        self.tries += 1
        self.running = False
        self.submit_time = time.time()

    def cleanup(self, cleanup):
        """Clean up intermediate files generated by job"""
        if not cleanup:
            return
        with contextlib.suppress(OSError):
            for filename in self.filenames:
                os.remove(filename)

    @staticmethod
    def monitor(jobs, cleanup=False, tracking=EVENT_LOG_TRACKING):
        """
        Monitor running jobs until completion
        Reference: https://htcondor.readthedocs.io/en/latest/apis/python-bindings/advanced/Scalable-Job-Tracking.html
        * Event log tracking uses log files to track job status
            * May error out if opening too many event logs simultaneously
              (greater than sysctl fs.inotify.max_user_instances)
        * Poll-based tracking uses condor_schedd to track job status
            * Completed jobs need to query condor history, from which job may leak out of if not queried soon enough
            * HTCondor staff discourages using this to reduce server load
        """
        for job in jobs:
            assert job.cluster_id != -1  # Job needs to be running
        if tracking == EVENT_LOG_TRACKING:
            CondorJobWrapper.monitor_event_logs(jobs, cleanup)
        else:
            # Poll-based tracking
            CondorJobWrapper.monitor_polling(jobs, cleanup)
        if jobs and jobs[0].shared_filesystem:
            time.sleep(30)  # Time to allow file changes to reflect in shared filesystem

    @staticmethod
    def monitor_event_logs(jobs, cleanup):
        """Monitor jobs using event logs"""
        running_jobs_set = OrderedDict.fromkeys(jobs)
        while running_jobs_set:  # pylint: disable = too-many-nested-blocks
            running_jobs = list(running_jobs_set.keys())
            for job in running_jobs:
                events = htcondor.JobEventLog(job.filenames.log_filename).events(0)
                for event in events:
                    event_type = event.type
                    # Reference: https://htcondor.readthedocs.io/en/latest/apis/python-bindings/api/htcondor.html#reading-job-events
                    if event_type == JobEventType.EXECUTE:
                        if not job.running:
                            job.execute_time = time.time()
                            job.running = True
                    if event_type == JobEventType.JOB_TERMINATED:
                        if event["TerminatedNormally"]:
                            if event["ReturnValue"] != 0:
                                CondorJobWrapper.process_failure(job, "terminated normally with non-zero return code", jobs, retry=True)
                            else:
                                CondorJobWrapper.process_success(job, running_jobs_set, cleanup)
                        else:
                            CondorJobWrapper.process_failure(job, "terminated abnormally", jobs, retry=True)
                    elif event_type == JobEventType.JOB_HELD:
                        hold_reason_code = event["HoldReasonCode"]
                        if hold_reason_code != 1:
                            CondorJobWrapper.process_failure(job, event["HoldReason"], jobs, retry=hold_reason_code in CONDOR_HOLD_RETRY_CODES)
                CondorJobWrapper.process_timeout(job, jobs)
            time.sleep(60)

    @staticmethod
    def monitor_polling(jobs, cleanup):
        """Monitor jobs by polling condor_schedd"""
        schedd = htcondor.Schedd()
        running_jobs_set = OrderedDict.fromkeys(jobs)
        while running_jobs_set:
            running_jobs = list(running_jobs_set.keys())
            try:
                query_responses = list(schedd.xquery("true", ["ClusterId", "JobStatus", "ExitCode", "HoldReason", "HoldReasonCode"]))
                # This will fail if job leaks out of returned truncated history:
                history_responses = list(schedd.history("true", ["ClusterId", "JobStatus", "ExitCode", "HoldReason", "HoldReasonCode"]))
            except RuntimeError:
                # Timeout when waiting for remote host
                time.sleep(60)
                continue
            query_classads = {classad["ClusterId"]: classad for classad in query_responses}
            history_classads = {classad["ClusterId"]: classad for classad in history_responses}
            for job in running_jobs:
                classad = query_classads.get(job.cluster_id)
                if not classad:
                    classad = history_classads.get(job.cluster_id)
                assert classad
                job_status = classad["JobStatus"]
                # Reference: https://htcondor.readthedocs.io/en/latest/classad-attributes/job-classad-attributes.html
                if job_status == 2:  # Job running
                    if not job.running:
                        job.execute_time = time.time()
                        job.running = True
                if job_status == 4:  # Job completed
                    if classad["ExitCode"] != 0:
                        CondorJobWrapper.process_failure(job, "terminated normally with non-zero return code", jobs, retry=True)
                    else:
                        CondorJobWrapper.process_success(job, running_jobs_set, cleanup)
                elif job_status == 5:  # Job held
                    hold_reason_code = classad["HoldReasonCode"]
                    if hold_reason_code != 1:
                        CondorJobWrapper.process_failure(job, classad["HoldReason"], jobs, retry=hold_reason_code in CONDOR_HOLD_RETRY_CODES)
                CondorJobWrapper.process_timeout(job, jobs)
            time.sleep(60)

    @staticmethod
    def process_success(job, running_jobs_set, cleanup):
        """Remove successful job from queue"""
        running_jobs_set.pop(job)
        job.cleanup(cleanup)

    @staticmethod
    def process_failure(job, reason, jobs, retry=False):
        """Restart or crash failed job"""
        remove_reason = (f"Job {job.name} failed: {reason}: see error file: {job.filenames.err_filename}.")
        if retry and job.tries <= CONDOR_MAX_RETRIES:
            remove_reason += (f" Retrying - attempt {job.tries + 1}")
            CondorJobWrapper.remove_jobs([job], reason=remove_reason)
            job.run()
            return
        CondorJobWrapper.remove_jobs(jobs, reason=remove_reason)
        raise CondorJobFailure(remove_reason)

    @staticmethod
    def process_timeout(job, jobs):
        """Restart or crash timed-out job"""
        current_time = time.time()
        waiting_time = current_time - job.submit_time
        running_time = current_time - job.execute_time
        remove_reason = ""
        if job.running and running_time > CONDOR_MAX_RUNNING_TIME:
            remove_reason = f"failed to run in {CONDOR_MAX_RUNNING_TIME} seconds;"
        elif not job.running and waiting_time > CONDOR_MAX_WAIT_TIME:
            remove_reason = f"failed to start running in {CONDOR_MAX_RUNNING_TIME} seconds;"
        if remove_reason:
            if job.tries > CONDOR_MAX_RETRIES:
                remove_reason += f" exceeded max retries {CONDOR_MAX_RETRIES}"
                CondorJobWrapper.process_failure(job, remove_reason, jobs)
            remove_reason += (f" retrying - attempt {job.tries + 1}")
            CondorJobWrapper.remove_jobs(jobs, reason=remove_reason)
            job.run()

    @staticmethod
    def remove_jobs(jobs, reason=None):
        """Remove jobs from condor queue"""
        schedd = htcondor.Schedd()
        for job in jobs:
            schedd.act(JobAction.Remove, f"ClusterId=={job.cluster_id}", reason=reason)
