"""
mihifepe master pipeline
Given test samples, a trained model and a feature hierarchy,
computes the effect on the model's output loss after perturbing the features/feature groups
in the hierarchy.
"""

import argparse
import copy
import csv
from datetime import datetime
import logging
import os
import pickle
import re
import subprocess
import time

import anytree
import h5py
import numpy as np
from sklearn.metrics import roc_auc_score

from .compute_p_values import compute_p_value
from . import constants
from .feature import Feature
from . import worker

def main():
    """Parse arguments"""
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument("-model_generator_filename", help="python script that generates model "
                        "object for subsequent callbacks to model.predict", required=True)
    parser.add_argument("-hierarchy_filename", help="Feature hierarchy in CSV format", required=True)
    parser.add_argument("-data_filename", help="Test data in HDF5 format", required=True)
    parser.add_argument("-output_dir", help="Output directory", required=True)
    # Optional arguments
    parser.add_argument("-perturbation", default=constants.ZEROING, choices=[constants.ZEROING, constants.SHUFFLING],
                        help="type of perturbation to perform:\n"
                        "%s (default): works on both static and temporal data\n"
                        "%s: works only on static data" % (constants.ZEROING, constants.SHUFFLING))
    parser.add_argument("-num_shuffling_trials", type=int, default=500, help="Number of shuffling trials to average over, "
                        "when shuffling perturbations are selected")
    parser.add_argument("-condor", dest="condor", action="store_true",
                        help="Enable parallelization using condor (default disabled)")
    parser.add_argument("-no-condor", dest="condor", action="store_false", help="Disable parallelization using condor")
    parser.set_defaults(condor=False)
    parser.add_argument("-features_per_worker", type=int, default=10, help="worker load")
    parser.add_argument("-eviction_timeout", type=int, default=14400, help="time in seconds to allow condor jobs"
                        " to run before evicting and restarting them on another condor node")
    parser.add_argument("-idle_timeout", type=int, default=3600, help="time in seconds to allow condor jobs"
                        " to stay idle before removing them from condor and attempting them on the master node.")
    parser.add_argument("-memory_requirement", type=int, default=16, help="memory requirement in GB, minimum 1, default 16")
    parser.add_argument("-compile_results_only", help="only compile results (assuming they already exist), "
                        "skipping actually launching jobs", action="store_true")
    parser.add_argument("-model_type", default=constants.REGRESSION,
                        help="Model type - output includes perturbed AUROCs for binary classifiers",
                        choices=[constants.BINARY_CLASSIFIER, constants.CLASSIFIER, constants.REGRESSION],)

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    np.random.seed(constants.SEED)
    logging.basicConfig(level=logging.INFO, filename="%s/master.log" % args.output_dir,
                        format="%(asctime)s: %(message)s")
    logger = logging.getLogger(__name__)
    pipeline(args, logger)


def pipeline(args, logger):
    """Master pipeline"""
    logger.info("Begin mihifepe master pipeline with args: %s" % args)
    # Load hierarchy from file
    hierarchy_root = load_hierarchy(args.hierarchy_filename)
    # Flatten hierarchy to allow partitioning across workers
    feature_nodes = flatten_hierarchy(hierarchy_root)
    # Perturb features
    targets, losses, predictions = perturb_features(args, logger, feature_nodes)
    # Compute p-values
    losses, predictions = round_vectors(losses, predictions)
    compute_p_values(args, hierarchy_root, targets, losses, predictions)
    # Run hierarchical FDR
    hierarchical_fdr(args, logger)
    logger.info("End mihifepe master pipeline")


def load_hierarchy(hierarchy_filename):
    """
    Load hierarchy from CSV.

    Args:
        hierarchy_filename: CSV specifying hierarchy in required format (see mihifepe/spec.md)

    Returns:
        anytree node representing root of hierarchy

    """
    root = None
    nodes = {}
    # Construct nodes
    with open(hierarchy_filename) as hierarchy_file:
        reader = csv.DictReader(hierarchy_file)
        for row in reader:
            node = Feature(row[constants.NODE_NAME],
                           parent_name=row[constants.PARENT_NAME], description=row[constants.DESCRIPTION],
                           static_indices=Feature.unpack_indices(row[constants.STATIC_INDICES]),
                           temporal_indices=Feature.unpack_indices(row[constants.TEMPORAL_INDICES]))
            assert node.name not in nodes, "Node name must be unique across all features: %s" % node.name
            nodes[node.name] = node
    # Construct tree
    for node in nodes.values():
        if not node.parent_name:
            assert not root, "Invalid tree structure: %s and %s both have no parent" % (root.node_name, node.node_name)
            root = node
        else:
            assert node.parent_name in nodes, "Invalid tree structure: no parent named %s" % node.parent_name
            node.parent = nodes[node.parent_name]
    assert root, "Invalid tree structure: root node missing (every node has a parent)"
    # Checks
    all_static_indices = set()
    all_temporal_indices = set()
    for node in anytree.PostOrderIter(root):
        if node.is_leaf:
            assert node.static_indices or node.temporal_indices, "Leaf node %s must have at least one index of either type" % node.name
            assert not all_static_indices.intersection(node.static_indices), "Leaf node %s has static index overlap with other leaf nodes" % node.name
            assert not all_temporal_indices.intersection(node.temporal_indices), "Leaf node %s has temporal index overlap with other leaf nodes" % node.name
        else:
            # Ensure non-leaf nodes have empty initial indices
            assert not node.static_indices, "Non-leaf node %s has non-empty initial indices" % node.name
            assert not node.temporal_indices, "Non-leaf node %s has non-empty initial indices" % node.name
    # Populate data structures
    for node in anytree.PostOrderIter(root):
        for child in node.children:
            node.static_indices += child.static_indices
            node.temporal_indices += child.temporal_indices
    return root


def flatten_hierarchy(hierarchy_root):
    """
    Flatten hierarchy to allow partitioning across workers

    Args:
        hierarchy_root: root of feature hierarchy

    Returns:
        Flattened hierarchy comprising list of features/feature groups
    """
    nodes = list(anytree.PreOrderIter(hierarchy_root))
    nodes.append(Feature(constants.BASELINE, description="No perturbation")) # Baseline corresponds to no perturbation
    np.random.shuffle(nodes) # To balance load across workers
    return nodes


def perturb_features(args, logger, feature_nodes):
    """
    Perturb features, observe effect on model loss and aggregate results

    Args:
        args: Command-line arguments
        feature_nodes: flattened feature hierarchy comprising nodes for base features/feature groups

    Returns:
        Aggregated results from workers
    """
    # Partition features, Launch workers, Aggregate results
    worker_pipeline = SerialPipeline(args, logger, feature_nodes)
    if args.condor:
        worker_pipeline = CondorPipeline(args, logger, feature_nodes)
    return worker_pipeline.run()


def round_vectors(losses, predictions):
    """Round to 4 decimals to avoid floating-point errors"""

    def round_vectordict(vectordict):
        """Round dictionary of vectors"""
        return {key: np.around(value, decimals=4) for (key, value) in vectordict.items()}

    losses = round_vectordict(losses)
    predictions = round_vectordict(predictions)
    return losses, predictions


def compute_p_values(args, hierarchy_root, targets, losses, predictions):
    """Evaluates and compares different feature erasures"""
    # pylint: disable = too-many-locals
    outfile = open("%s/%s" % (args.output_dir, constants.PVALUES_FILENAME), "w")
    writer = csv.writer(outfile, delimiter=",")
    writer.writerow([constants.NODE_NAME, constants.PARENT_NAME, constants.DESCRIPTION, constants.EFFECT_SIZE,
                     constants.MEAN_LOSS, constants.PVALUE_LOSSES])
    baseline_loss = losses[constants.BASELINE]
    for node in anytree.PreOrderIter(hierarchy_root):
        name = node.name
        parent_name = node.parent.name if node.parent else ""
        loss = losses[node.name]
        mean_loss = np.mean(loss)
        pvalue_loss = compute_p_value(baseline_loss, loss)
        # Compute AUROC depending on whether task is binary classification or not:
        auroc = ""
        if args.model_type == constants.BINARY_CLASSIFIER:
            prediction = predictions[node.name]
            auroc = roc_auc_score(targets, prediction)
        writer.writerow([name, parent_name, node.description, auroc, mean_loss, pvalue_loss])
    outfile.close()


def hierarchical_fdr(args, logger):
    """Performs hierarchical FDR control on results"""
    input_filename = "%s/%s" % (args.output_dir, constants.PVALUES_FILENAME)
    output_dir = "%s/%s" % (args.output_dir, constants.HIERARCHICAL_FDR_DIR)
    cmd = ("python -m mihifepe.fdr.hierarchical_fdr_control -output_dir %s -procedure yekutieli "
           "-rectangle_leaves %s" % (output_dir, input_filename))
    logger.info("Running cmd: %s" % cmd)
    subprocess.check_call(cmd, shell=True)


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
        condor_helper = CondorPipeline(self.args, self.logger, None)
        if not self.args.compile_results_only:
            # Write all features to file
            self.args.features_filename = "%s/%s" % (self.args.output_dir, "all_features.csv")
            self.args.task_idx = 0
            condor_helper.write_features(self.args, self.feature_nodes)
            # Run worker pipeline
            worker.pipeline(self.args, self.logger)
        # Aggregate results
        return condor_helper.compile_results()


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

    @staticmethod
    def get_output_filepath(targs, prefix, suffix="txt"):
        """Helper function to generate output filepath"""
        return "%s/%s_worker_%d.%s" % (targs.output_dir, prefix, targs.task_idx, suffix)

    def write_features(self, targs, task_features):
        """Write features to file"""
        features_filename = self.get_output_filepath(targs, "features")
        targs.features_filename = features_filename
        with open(features_filename, "w") as features_file:
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
            self.logger.error("\nFailed to run cmd: '%s' successfully in the alloted number of attempts %d" % (task[constants.CMD], constants.MAX_ATTEMPTS))
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

                # Evict job still if timeout exceeded - it will be put back in queue and restarted/resumed elsewhere
                elapsed_time = (datetime.now() - task[constants.JOB_START_TIME]).seconds
                if elapsed_time > self.master_args.eviction_timeout:
                    self.logger.info("Job time limit exceeded - evicting and restarting/resuming elsewhere")
                    subprocess.check_call("condor_vacate_job %d" % task[constants.CLUSTER], shell=True)
                    task[constants.JOB_START_TIME] = datetime.now()
                    continue

                # Some jobs get stuck in idle indefinitely (likely due to condor bug) - attempt to run these in master node
                job_status = subprocess.check_output("condor_q -format '%d' JobStatus {0}".format(task[constants.CLUSTER]), shell=True)
                if job_status and int(job_status) == 1 and elapsed_time > self.master_args.idle_timeout:
                    self.logger.warn("Job '%s' has been idle for too long, attempting to run in master node instead" % task[constants.CMD])
                    subprocess.call("condor_rm %d" % task[constants.CLUSTER], shell=True)
                    task[constants.JOB_COMPLETE] = 1
                    unfinished_tasks -= 1
                    failed_tasks.append(task)
                    continue

                # Parse log file to determine task status
                with open(task[constants.LOG_FILENAME]) as log_file:
                    lines = [line.strip() for line in log_file]
                for line in lines:
                    if line.find(constants.ABNORMAL_TERMINATION) >= 0:
                        self.logger.warn("Cmd '%s' failed due to abnormal termination. Re-attempting..." % task[constants.CMD])
                        rerun_tasks.append(task)
                        break
                    elif line.find(constants.NORMAL_TERMINATION_SUCCESS) >= 0:
                        self.logger.info("Cmd '%s' completed successfully" % task[constants.CMD])
                        task[constants.JOB_COMPLETE] = 1
                        unfinished_tasks -= 1
                        break
                    elif line.find(constants.NORMAL_TERMINATION_FAILURE) >= 0:
                        task[constants.NORMAL_FAILURE_COUNT] += 1
                        if task[constants.NORMAL_FAILURE_COUNT] > constants.MAX_NORMAL_FAILURE_COUNT:
                            self.logger.error("Cmd '%s' terminated with invalid return code. Reached maximum number of "
                                              "normal failures %d on condor workers, attempting in master node." %
                                              (task[constants.CMD], constants.MAX_NORMAL_FAILURE_COUNT))
                            task[constants.JOB_COMPLETE] = 1
                            unfinished_tasks -= 1
                            failed_tasks.append(task)
                            break
                        self.logger.warn("Cmd '%s' terminated normally with invalid return code. Re-running assuming condor failure "
                                         "(attempt %d of %d)..." % (task[constants.CMD], task[constants.NORMAL_FAILURE_COUNT] + 1, constants.MAX_NORMAL_FAILURE_COUNT + 1))
                        rerun_tasks.append(task)
                        break
                    elif line.find(constants.JOB_HELD) >= 0:
                        release_tasks.append(task)
                        break

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
                time.sleep(30) # To prevent infinite-idle condor issue
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
        self.logger.info("All workers completed running successfully")


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
        return tasks


    def compile_results(self):
        """Compile condor task results"""
        self.logger.info("Compiling condor task results")
        all_losses = {}
        all_predictions = {}
        targets = None
        for results_filename in os.listdir(self.master_args.output_dir):
            match = re.match("results_worker_([0-9]+).hdf5", results_filename)
            if match:
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
                idx = int(match.group(1))
                if idx == 0:
                    # Only first worker outputs labels since they're common
                    targets = root[constants.TARGETS].value
        assert targets is not None
        return targets, all_losses, all_predictions


    def run(self):
        """Run condor pipeline"""
        self.logger.info("Begin condor pipeline")
        if not self.master_args.compile_results_only:
            tasks = self.create_tasks()
            self.launch_tasks(tasks)
            self.monitor_tasks(tasks)
        targets, losses, predictions = self.compile_results()
        self.logger.info("End condor pipeline")
        return targets, losses, predictions


if __name__ == "__main__":
    main()
