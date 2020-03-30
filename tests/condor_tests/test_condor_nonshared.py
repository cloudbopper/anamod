"""Test condor on non-shared filesystem"""

from anamod.utils import CondorJobWrapper

# TODO: Test relative/absolute file paths, cleanup


# pylint: disable = protected-access, too-many-locals
def test_condor_nonshared(tmpdir):
    """Test condor functionality"""
    files = [f"{tmpdir}/file0.txt", f"{tmpdir}/file1.txt"]
    with open(files[0], "w") as file0:
        file0.write("Hello")
    with open(files[1], "w") as file1:
        file1.write("World")
    cmds = ["cat file0.txt > dir0/newfile.txt", "cat file1.txt > dir1/newfile.txt"]
    dirs = [f"{tmpdir}/dir0", f"{tmpdir}/dir1"]
    jobs = [None] * 2
    for idx, cmd in enumerate(cmds):
        directory = dirs[idx]
        job = CondorJobWrapper(cmd, [files[idx]], directory)
        job.run()
        jobs[idx] = job
    CondorJobWrapper.monitor(jobs)
    output = ""
    for directory in dirs:
        with open(f"{directory}/newfile.txt", "r") as newfile:
            output += newfile.read()
    assert output == "HelloWorld"
