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
import numpy as np
import sklearn

from compute_p_values import compute_p_value
import constants

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
    # TODO: type of perturbation; note: current parallel setup cannot handle shuffling involving multiple features
    parser.add_argument("-features_per_worker", type=int, default=10, help="worker load")
    parser.add_argument("-eviction_timeout", type=int, default=14400, help="time in seconds to allow condor jobs"
                        " to run before evicting and restarting")
    parser.add_argument("-compile_results_only", help="only compile results (assuming they already exist), "
                        "skipping actually launching jobs", action="store_true")

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
    # Evaluate
    losses, predictions = round_vectors(losses, predictions)
    evaluate(args, feature_nodes, targets, losses, predictions)
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
            node = Feature(row[constants.NODE_NAME], category=row[constants.CATEGORY], parent_name=row[constants.PARENT_NAME],
                           description=row[constants.DESCRIPTION], static_indices=list_idx(row[constants.STATIC_INDICES]),
                           temporal_indices=list_idx(row[constants.TEMPORAL_INDICES]))
            assert node.identifier not in nodes, "(category, name) tuple must be unique across all features: %s" % node.identifier
            nodes[node.identifier] = node
    # Construct tree
    for node in nodes:
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
    nodes += Feature(constants.BASELINE, category=constants.BASELINE, description="No perturbation") # Baseline corresponds to no perturbation
    return np.random.shuffle(nodes) # To balance load across workers


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
    worker_pipeline = CondorPipeline(args, logger, feature_nodes) # TODO: non-condor pipeline?
    return worker_pipeline.run()


def round_vectors(losses, predictions):
    """Round to 4 decimals to avoid floating-point errors"""

    def round_vectordict(vectordict):
        """Round dictionary of vectors"""
        return {key: np.around(value, decimals=4) for (key, value) in vectordict.iteritems()}

    losses = round_vectordict(losses)
    if predictions and next(iter(predictions.values())): # predictions has actual data (only for binary classification)
        predictions = round_vectordict(predictions)
    return losses, predictions


def evaluate(args, feature_nodes, targets, losses, predictions):
    """Evaluates and compares different feature erasures"""
    # pylint: disable = too-many-locals
    outfile = open("%s/outputs.csv" % args.output_dir, "wb")
    writer = csv.writer(outfile, delimiter=",")
    writer.writerow([constants.CATEGORY, constants.NODE_NAME, constants.DESCRIPTION, constants.AUROC, constants.MEAN_LOSS, constants.PVALUE_LOSSES])
    baseline_loss = losses[Feature.get_identifier(constants.BASELINE, constants.BASELINE)]
    for node in feature_nodes:
        name = node.name
        category = node.category
        loss = losses[node.identifier]
        mean_loss = np.mean(loss)
        pvalue_loss = compute_p_value(baseline_loss, loss)
        # Compute AUROC depending on whether task is binary classification or not:
        prediction = predictions[node.identifier]
        auroc = sklearn.metrics.roc_auc_score(targets, prediction) if prediction else ""
        writer.writerow([category, name, node.description.encode('utf8'), auroc, mean_loss, pvalue_loss])
    outfile.close()


class Feature(anytree.Node):
    """Class representing feature/feature group"""
    def __init__(self, name, **kwargs):
        super().__init__(name)
        self.category = kwargs.get(constants.CATEGORY, "")
        self.parent_name = kwargs.get(constants.PARENT_NAME, "")
        self.description = kwargs.get(constants.DESCRIPTION, "")
        self.static_indices = kwargs.get(constants.STATIC_INDICES, [])
        self.temporal_indices = kwargs.get(constants.TEMPORAL_INDICES, [])
        self.identifier = self.get_identifier(self.category, name)

    @staticmethod
    def get_identifier(category, name):
        """
        Construct feature identifier for given (category, name) tuple

        Args:
            category: feature category, acts as namespace
            name: feature name

        Returns:
            identifier
        """
        return "%s:%s" % (category, name)


class CondorPipeline():
    """Class managing condor pipeline for distributing load across workers"""

    def __init__(self, master_args, logger, feature_nodes):
        self.master_args = master_args
        self.logger = logger
        self.feature_nodes = feature_nodes

    @staticmethod
    def get_output_filepath(targs, prefix, suffix="txt"):
        """Helper function to generate output filepath"""
        return "%s/%s_worker_%d.%s" % (targs.output_dir, prefix, targs.task_idx, suffix)

    def write_features(self, targs, task_features):
        """Write features to file"""
        features_filename = self.get_output_filepath(targs, "features")
        targs.features_filename = features_filename
        targs.all_features = False
        with open(features_filename, "w") as features_file:
            writer = csv.writer(features_file)
            writer.writerow([constants.CATEGORY, constants.NODE_NAME,
                             constants.STATIC_INDICES, constants.TEMPORAL_INDICES])
            for feature_node in task_features:
                writer.writerow([feature_node.category_name, feature_node.name,
                                 unlist_idx(feature_node.static_indices),
                                 unlist_idx(feature_node.temporal_indices)])

    def write_arguments(self, targs):
        """Write task-specific arguments"""
        args_filename = self.get_output_filepath(targs, "args", suffix="pkl")
        targs.args_filename = args_filename
        with open(args_filename, "w") as args_file:
            pickle.dump(targs, args_file)

    def write_submit_file(self, targs):
        """Write task-specific submit file"""
        template_filename = "%s/condor_template.sub" % os.path.dirname(os.path.abspath(__file__))
        submit_filename = self.get_output_filepath(targs, "condor_task", suffix="sub")
        task = {}
        task[constants.ARGS_FILENAME] = targs.args_filename
        task[constants.LOG_FILENAME] = self.get_output_filepath(targs, "log")
        task[constants.OUTPUT_FILENAME] = self.get_output_filepath(targs, "out")
        task[constants.ERROR_FILENAME] = self.get_output_filepath(targs, "err")
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
                output = subprocess.check_output(task[constants.CMD], shell=True)
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
            unfinished_tasks = 0
            release = False
            time.sleep(120)
            for task in tasks:
                if constants.JOB_COMPLETE in task:
                    continue
                rerun = False
                unfinished_tasks += 1
                with open(task[constants.LOG_FILENAME]) as log_file:
                    lines = [line.strip() for line in log_file]
                for line in lines:
                    if line.find(constants.ABNORMAL_TERMINATION) >= 0:
                        self.logger.warn("Cmd '%s' failed due to abnormal termination. Re-attempting..." % task[constants.CMD])
                        rerun = True
                        break
                    elif line.find(constants.NORMAL_TERMINATION_SUCCESS) >= 0:
                        self.logger.info("Cmd '%s' completed successfully" % task[constants.CMD])
                        task[constants.JOB_COMPLETE] = 1
                        unfinished_tasks -= 1
                        break
                    elif line.find(constants.NORMAL_TERMINATION_FAILURE) >= 0:
                        self.logger.warn("Cmd '%s' terminated with invalid return code. Examining ..." % task[constants.CMD])
                        with open(task[constants.ERROR_FILENAME]) as error_file:
                            for err_line in error_file:
                                if re.search("I/O error|ImportError|AttributeError|IOError|OSError", err_line):
                                    self.logger.warn("Cmd '%s' failed due to likely condor-related error. Re-attempting..." % task[constants.CMD])
                                    rerun = True
                                    break
                            if not rerun:
                                self.logger.error("\nCmd '%s' completed with errors." % task[constants.CMD])
                                task[constants.JOB_COMPLETE] = 1
                                unfinished_tasks -= 1
                        break
                    elif line.find(constants.JOB_HELD) >= 0:
                        for filetype in [constants.LOG_FILENAME, constants.OUTPUT_FILENAME, constants.ERROR_FILENAME]:
                            try:
                                os.remove(task[filetype])
                            except OSError:
                                pass
                        self.logger.info("\nTask was held, releasing. Cmd: '%s'" % task[constants.CMD])
                        release = True
                        break
                if rerun:
                    for filetype in [constants.LOG_FILENAME, constants.OUTPUT_FILENAME, constants.ERROR_FILENAME]:
                        os.remove(task[filetype])
                    launch_success = self.launch_task(task)
                    if not launch_success:
                        task[constants.JOB_COMPLETE] = 1
                        unfinished_tasks -= 1
                # Release job if held
                if release:
                    subprocess.check_call("condor_release %d" % task[constants.CLUSTER], shell=True)
                # Evict job still if timeout exceeded - it will be put back in queue and restarted/resumed elsewhere
                elapsed_time = (datetime.now() - task[constants.JOB_START_TIME]).seconds
                if elapsed_time > self.master_args.eviction_timeout:
                    self.logger.info("Job time limit exceeded - evicting and restarting/resuming elsewhere")
                    subprocess.check_call("condor_vacate_job %d" % task[constants.CLUSTER], shell=True)
                    task[constants.JOB_START_TIME] = datetime.now()

        if os.path.isfile(kill_filename):
            os.remove(kill_filename)


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
            match = re.match("results_worker_([0-9]+).pkl", results_filename)
            if match:
                self.logger.info("Processing %s" % results_filename)
                idx = int(match.group(1))
                with open("%s/%s" % (self.master_args.output_dir, results_filename), "rb") as results_file:
                    losses = pickle.load(results_file)
                    all_losses.update(losses)
                    # Predictions non-empty if task is binary classification
                    predictions = pickle.load(results_file)
                    all_predictions.update(predictions)
                    if idx == 0:
                        targets = pickle.load(results_file) # Only first worker outputs labels since they're common
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


def list_idx(str_indices):
    """Converts tab-separated string of indices to int list"""
    return [int(idx) for idx in "\t".split(str_indices)]


def unlist_idx(int_indices):
    """Converts int list of indices to tab-separated string"""
    return "\t".join([str(idx) for idx in int_indices])


if __name__ == "__main__":
    main()
