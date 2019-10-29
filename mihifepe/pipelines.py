"""Serial and distributed (condor) perturbation pipelines"""

import copy
import csv
from datetime import datetime
import glob
import math
import os
import pickle
import re
import subprocess
import time

import h5py
import numpy as np

from mihifepe import constants, worker
from mihifepe.feature import Feature


class SerialPipeline():
    """Serial (non-condor) implementation"""
    # pylint: disable=too-few-public-methods
    def __init__(self, args, logger, feature_nodes):
        self.args = copy.deepcopy(args)
        self.logger = logger
        self.feature_nodes = feature_nodes

    def run(self):
        """Run serial pipeline"""
        self.logger.info("Begin running serial pipeline")
        condor_helper = CondorPipeline(self.args, self.logger, [])
        condor_helper.task_count = 1
        if not self.args.compile_results_only:
            # Write all features to file
            self.args.task_idx = 0
            condor_helper.write_features(self.args, self.feature_nodes)
            # Run worker pipeline
            worker.pipeline(self.args, self.logger)
        # Aggregate results
        results = condor_helper.compile_results()
        if self.args.cleanup:
            condor_helper.cleanup()
        return results


class CondorPipeline():
    """Class managing condor pipeline for distributing load across workers"""

    def __init__(self, args, logger, feature_nodes):
        self.master_args = copy.deepcopy(args)
        self.logger = logger
        self.feature_nodes = feature_nodes
        self.virtual_env = ""
        if constants.VIRTUAL_ENV in os.environ:
            self.virtual_env = os.path.split(os.environ[constants.VIRTUAL_ENV])[1]
        assert self.master_args.memory_requirement >= 1, "Required memory must be 1 or more GB"
        self.memory_requirement = str(self.master_args.memory_requirement)
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.task_count = math.ceil(len(self.feature_nodes) / self.master_args.features_per_worker)

    @staticmethod
    def get_output_filepath(targs, prefix, suffix="txt"):
        """Helper function to generate output filepath"""
        return "%s/%s_worker_%d.%s" % (targs.output_dir, prefix, targs.task_idx, suffix)

    def write_features(self, targs, task_features):
        """Write features to file"""
        targs.features_filename = self.get_output_filepath(targs, "features")
        with open(targs.features_filename, "w", newline="") as features_file:
            writer = csv.writer(features_file)
            writer.writerow([constants.NODE_NAME,
                             constants.STATIC_INDICES, constants.TEMPORAL_INDICES])
            for feature_node in task_features:
                writer.writerow([feature_node.name,
                                 Feature.pack_indices(feature_node.static_indices),
                                 Feature.pack_indices(feature_node.temporal_indices)])

    def write_arguments(self, targs):
        """Write task-specific arguments"""
        args_filename = self.get_output_filepath(targs, "args", suffix="pkl")
        targs.args_filename = args_filename
        with open(args_filename, "wb") as args_file:
            pickle.dump(targs, args_file)

    def write_submit_file(self, targs):
        """Write task-specific submit file"""
        template_filename = "%s/condor_template.sub" % self.script_dir
        submit_filename = self.get_output_filepath(targs, "condor_task", suffix="sub")
        task = {}
        task[constants.ARGS_FILENAME] = targs.args_filename
        task[constants.LOG_FILENAME] = self.get_output_filepath(targs, "log")
        task[constants.OUTPUT_FILENAME] = self.get_output_filepath(targs, "out")
        task[constants.ERROR_FILENAME] = self.get_output_filepath(targs, "err")
        task[constants.VIRTUAL_ENV] = self.virtual_env
        task[constants.MEMORY_REQUIREMENT] = self.memory_requirement
        task[constants.SCRIPT_DIR] = self.script_dir
        submit_file = open(submit_filename, "w")
        with open(template_filename, "r") as template_file:
            for line in template_file:
                for key in task:
                    line = line.replace(key, task[key])
                submit_file.write(line)
        submit_file.close()
        task[constants.SUBMIT_FILENAME] = submit_filename
        task[constants.CMD] = "condor_submit %s" % submit_filename
        task[constants.ATTEMPT] = 0
        task[constants.NORMAL_FAILURE_COUNT] = 0
        return task

    def launch_tasks(self, tasks):
        """Launch condor tasks"""
        for task in tasks:
            self.launch_task(task)

    def launch_task(self, task):
        """Launch condor task and return launch success"""
        if task[constants.ATTEMPT] < constants.MAX_ATTEMPTS:
            task[constants.ATTEMPT] += 1
            self.logger.info("\nAttempt %d: running cmd: '%s'" % (task[constants.ATTEMPT], task[constants.CMD]))
            try:
                output = subprocess.check_output(task[constants.CMD], shell=True).decode("utf-8")
                self.logger.info(output)
                cluster = re.search("cluster ([0-9]+)", output)
                assert cluster
                cluster = cluster.groups()[0]
                task[constants.CLUSTER] = int(cluster)
                task[constants.JOB_START_TIME] = datetime.now()
                return True
            except subprocess.CalledProcessError as err:
                self.logger.warn("condor_submit command failed with error: %s;\nRe-attempting..." % err)
                return self.launch_task(task)
        else:
            self.logger.error("\nFailed to run cmd: '%s' successfully in the alloted number of attempts %d"
                              % (task[constants.CMD], constants.MAX_ATTEMPTS))
            return False

    def monitor_tasks(self, tasks):
        """Monitor condor tasks, restarting as necessary"""
        # TODO: refactor
        # pylint: disable = too-many-branches, too-many-nested-blocks, too-many-statements, too-many-locals
        unfinished_tasks = 1
        kill_filename = "%s/kill_file.txt" % self.master_args.output_dir
        self.logger.info("\nMonitoring/restarting tasks until all tasks verified completed. "
                         "Delete file %s to end task monitoring/restarting\n" % kill_filename)
        with open(kill_filename, "w") as kill_file:
            kill_file.write("Delete this file to kill process\n")
        while unfinished_tasks and os.path.isfile(kill_filename):
            failed_tasks = []
            rerun_tasks = []
            release_tasks = []
            unfinished_tasks = 0
            time.sleep(30)
            for task in tasks:
                if constants.JOB_COMPLETE in task:
                    continue
                unfinished_tasks += 1
                task_updated = False

                # Parse log file to determine task status
                # TODO: Potentially use condor_status instead of parsing log file
                with open(task[constants.LOG_FILENAME]) as log_file:
                    lines = [line.strip() for line in log_file]
                for line in reversed(lines):
                    if line.find(constants.ABNORMAL_TERMINATION) >= 0:
                        self.logger.warn("Cmd '%s' failed due to abnormal termination. Re-attempting..." % task[constants.CMD])
                        rerun_tasks.append(task)
                        task_updated = True
                        break
                    if line.find(constants.NORMAL_TERMINATION_SUCCESS) >= 0:
                        self.logger.info("Cmd '%s' completed successfully" % task[constants.CMD])
                        task[constants.JOB_COMPLETE] = 1
                        unfinished_tasks -= 1
                        task_updated = True
                        break
                    if line.find(constants.NORMAL_TERMINATION_FAILURE) >= 0:
                        task[constants.NORMAL_FAILURE_COUNT] += 1
                        if task[constants.NORMAL_FAILURE_COUNT] > constants.MAX_NORMAL_FAILURE_COUNT:
                            self.logger.error("Cmd '%s' terminated with invalid return code. Reached maximum number of "
                                              "normal failures %d on condor workers, attempting in master node." %
                                              (task[constants.CMD], constants.MAX_NORMAL_FAILURE_COUNT))
                            task[constants.JOB_COMPLETE] = 1
                            unfinished_tasks -= 1
                            failed_tasks.append(task)
                            task_updated = True
                            break
                        self.logger.warn("Cmd '%s' terminated normally with invalid return code. Re-running assuming condor failure "
                                         "(attempt %d of %d)..."
                                         % (task[constants.CMD], task[constants.NORMAL_FAILURE_COUNT] + 1, constants.MAX_NORMAL_FAILURE_COUNT + 1))
                        rerun_tasks.append(task)
                        task_updated = True
                        break
                    if line.find(constants.JOB_HELD) >= 0:
                        release_tasks.append(task)
                        task_updated = True
                        break

                if not task_updated:
                    schedule_rerun = False
                    # Evict job still if timeout exceeded - it will be put back in queue and restarted/resumed elsewhere
                    elapsed_time = (datetime.now() - task[constants.JOB_START_TIME]).seconds
                    if elapsed_time > self.master_args.eviction_timeout:
                        self.logger.info("Job '%s' time limit exceeded, scheduling rerun" % task[constants.CMD])
                        schedule_rerun = True

                    # Some jobs get stuck in idle indefinitely (likely due to condor bug) - attempt to run these in master node
                    job_status = subprocess.check_output("condor_q -format '%d' JobStatus {0}".format(task[constants.CLUSTER]), shell=True)
                    if job_status and int(job_status) == 1 and elapsed_time > self.master_args.idle_timeout:
                        self.logger.warn("Job '%s' has been idle for too long, scheduling rerun" % task[constants.CMD])
                        schedule_rerun = True

                    if schedule_rerun:
                        subprocess.call("condor_rm %d" % task[constants.CLUSTER], shell=True)
                        rerun_tasks.append(task)

            # Release jobs if held
            if release_tasks:
                for task in release_tasks:
                    try:
                        self.logger.info("\nTask was held, releasing. Cmd: '%s'" % task[constants.CMD])
                        subprocess.check_output("condor_release %d" % task[constants.CLUSTER], shell=True)
                    except subprocess.CalledProcessError as err:
                        self.logger.warn(err)
                        self.logger.warn("'%s' may have failed due to job being automatically released in the interim - "
                                         "proceeding under assumption the job was automatically released." % err.cmd)

            # Rerun jobs that didn't complete, likely due to condor issues
            if rerun_tasks:
                for task in rerun_tasks:
                    for filetype in [constants.LOG_FILENAME, constants.OUTPUT_FILENAME, constants.ERROR_FILENAME]:
                        if os.path.isfile(task[filetype]):
                            dirname, basename = os.path.split(task[filetype])
                            root, ext = os.path.splitext(basename)
                            new_filename = "%s/%s_attempt_%d%s" % (dirname, root, task[constants.ATTEMPT], ext)
                            if os.path.isfile(new_filename):
                                self.logger.warn("File %s already exists, overwriting." % new_filename)
                            os.rename(task[filetype], new_filename)
                time.sleep(30)  # To prevent infinite-idle condor issue
                for task in rerun_tasks:
                    launch_success = self.launch_task(task)
                    if not launch_success:
                        task[constants.JOB_COMPLETE] = 1
                        unfinished_tasks -= 1
                        failed_tasks.append(task)

            # Attempt to run tasks that failed on condor in master node
            if failed_tasks:
                for task in failed_tasks:
                    self.logger.error("Task '%s' failed to run on condor, attempting on master node" % task[constants.CMD])
                    cmd = "python -m mihifepe.worker %s" % task[constants.ARGS_FILENAME]
                    self.logger.info("Running cmd '%s'" % cmd)
                    subprocess.check_call(cmd, shell=True)

        if os.path.isfile(kill_filename):
            os.remove(kill_filename)
        self.logger.info("All workers completed running successfully, cleaning up condor files")

    def create_tasks(self):
        """Create condor task setup"""
        tasks = []
        task_idx = 0
        node_idx = 0
        while node_idx < len(self.feature_nodes):
            targs = copy.deepcopy(self.master_args)
            targs.task_idx = task_idx
            task_features = self.feature_nodes[node_idx:min(len(self.feature_nodes), node_idx + targs.features_per_worker)]
            self.write_features(targs, task_features)
            self.write_arguments(targs)
            tasks.append(self.write_submit_file(targs))
            task_idx += 1
            node_idx += targs.features_per_worker
        assert task_idx == self.task_count
        return tasks

    def compile_results(self):
        """Compile condor task results"""
        self.logger.info("Compiling condor task results")
        all_losses = {}
        all_predictions = {}
        targets = None
        for task_idx in range(self.task_count):
            results_filename = "results_worker_%d.hdf5" % task_idx
            self.logger.info("Processing %s" % results_filename)
            root = h5py.File("%s/%s" % (self.master_args.output_dir, results_filename), "r")

            def load_data(group):
                """Helper function to load data"""
                results = {}
                for feature_id, feature_data in group.items():
                    results[feature_id] = feature_data.value
                return results

            all_losses.update(load_data(root[constants.LOSSES]))
            all_predictions.update(load_data(root[constants.PREDICTIONS]))
            if task_idx == 0:
                # Only first worker outputs labels since they're common
                # TODO: .value is deprecated (http://docs.h5py.org/en/latest/whatsnew/2.1.html?highlight=value), remove
                targets = root[constants.TARGETS].value  # pylint: disable = no-member
        assert targets is not None
        return targets, all_losses, all_predictions

    def cleanup(self):
        """Clean files after completion"""
        self.logger.info("Begin file cleanup")
        filetypes = ["err*", "out*", "log*", "args*", "condor_task*", "results*", "features*", "worker*"]
        for filetype in filetypes:
            for filename in glob.glob("%s/%s" % (self.master_args.output_dir, filetype)):
                os.remove(filename)
        self.logger.info("End file cleanup")

    def run(self):
        """Run condor pipeline"""
        self.logger.info("Begin condor pipeline")
        if not self.master_args.compile_results_only:
            tasks = self.create_tasks()
            self.launch_tasks(tasks)
            self.monitor_tasks(tasks)
        targets, losses, predictions = self.compile_results()
        if self.master_args.cleanup:
            self.cleanup()
        self.logger.info("End condor pipeline")
        return targets, losses, predictions


def round_vectordict(vectordict):
    """Round dictionary of vectors to 4 decimals to avoid floating-point errors"""
    return {key: round_vector(value) for (key, value) in vectordict.items()}


def round_vector(vector):
    """Round vector to 4 decimals to avoid floating-point errors"""
    return np.around(vector, decimals=4)
