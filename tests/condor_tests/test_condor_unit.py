"""Test condor functionality"""

import os
from anamod.core.utils import CondorJobWrapper


# pylint: disable = protected-access, too-many-locals
def test_condor_cat(tmpdir, shared_fs, tracking):
    """Test condor functionality"""
    num_jobs = 100
    dirs = [None] * num_jobs
    jobs = [None] * num_jobs
    for idx in range(num_jobs):
        filename = f"{tmpdir}/file{idx}.txt"
        with open(filename, "w", encoding="utf-8") as filep:
            filep.write(f"{idx}")
        dirs[idx] = f"{tmpdir}/dir{idx}"
        if shared_fs:
            cmd = f"cat {os.path.abspath(filename)} > {os.path.abspath(dirs[idx])}/newfile.txt"
        else:
            cmd = f"cat {os.path.basename(filename)} > {os.path.basename(dirs[idx])}/newfile.txt"
        job = CondorJobWrapper(cmd, [filename], dirs[idx], shared_filesystem=shared_fs, tracking=tracking,
                               avoid_bad_hosts=True, retry_arbitrary_failures=True)
        job.run()
        jobs[idx] = job
    CondorJobWrapper.monitor(jobs)
    output = ""
    for directory in dirs:
        with open(f"{directory}/newfile.txt", "r", encoding="utf-8") as newfile:
            output += newfile.read()
    assert output == "".join([f"{idx}" for idx in range(num_jobs)])
