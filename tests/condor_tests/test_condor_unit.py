"""Test condor on non-shared filesystem"""

import os
from anamod.utils import CondorJobWrapper

# TODO: Test cleanup


# pylint: disable = protected-access, too-many-locals
def test_condor_cat(tmpdir, shared_fs):
    """Test condor functionality"""
    files = [f"{tmpdir}/file0.txt", f"{tmpdir}/file1.txt"]
    with open(files[0], "w") as file0:
        file0.write("Hello")
    with open(files[1], "w") as file1:
        file1.write("World")
    dirs = [f"{tmpdir}/dir0", f"{tmpdir}/dir1"]
    if shared_fs:
        cmds = [f"cat {os.path.abspath(files[idx])} > {os.path.abspath(dirs[idx])}/newfile.txt" for idx in range(2)]
    else:
        cmds = [f"cat {os.path.basename(files[idx])} > {os.path.basename(dirs[idx])}/newfile.txt" for idx in range(2)]
    jobs = [None] * 2
    for idx, cmd in enumerate(cmds):
        directory = dirs[idx]
        job = CondorJobWrapper(cmd, [files[idx]], directory, shared_filesystem=shared_fs)
        job.run()
        jobs[idx] = job
    CondorJobWrapper.monitor(jobs)
    output = ""
    for directory in dirs:
        with open(f"{directory}/newfile.txt", "r") as newfile:
            output += newfile.read()
    assert output == "HelloWorld"
