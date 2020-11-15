"""Serial and distributed (condor) perturbation pipelines"""

from collections import deque, OrderedDict
import copy
import glob
import math
import os
import pickle
import shutil

import anytree
import cloudpickle
import numpy as np

from anamod.core import constants, worker
from anamod.core.utils import CondorJobWrapper
from anamod.fdr.fdr_algorithms import bh_procedure


class SerialPipeline():
    """Feature importance analysis pipeline"""
    def __init__(self, args, features):
        self.args = copy.copy(args)
        self.features = features
        self.num_jobs = 1

    def write_features(self):
        """Write features to analyze to files"""
        num_features_per_file = math.ceil(len(self.features) / self.num_jobs)
        for idx in range(self.num_jobs):
            job_features = self.features[idx * num_features_per_file: (idx + 1) * num_features_per_file]
            features_filename = constants.INPUT_FEATURES_FILENAME.format(self.args.output_dir, idx)
            with open(features_filename, "wb") as features_file:
                cloudpickle.dump(job_features, features_file, protocol=pickle.DEFAULT_PROTOCOL)

    def compile_results(self, output_dirs):
        """Compile results"""
        self.args.logger.info("Compiling results")
        features = []
        for idx in range(self.num_jobs):
            directory = output_dirs[idx]
            # Load features
            features_filename = constants.OUTPUT_FEATURES_FILENAME.format(directory, idx)
            with open(features_filename, "rb") as features_file:
                features.extend(cloudpickle.load(features_file))

        return features

    def cleanup(self, job_dirs=None):
        """Clean intermediate files after completing pipeline"""
        if not self.args.cleanup:
            return
        self.args.logger.info("Begin intermediate file cleanup")
        # Remove intermediate working directory files
        filetypes = [constants.INPUT_FEATURES_FILENAME.format(self.args.output_dir, "*"),
                     constants.OUTPUT_FEATURES_FILENAME.format(self.args.output_dir, "*"),
                     constants.RESULTS_FILENAME.format(self.args.output_dir, "*")]
        for filetype in filetypes:
            for filename in glob.glob(filetype):
                try:
                    os.remove(filename)
                except OSError as error:
                    self.args.logger.warning(f"Cleanup: unable to remove {filename}: {error}")
        if job_dirs:  # Remove condor job directories
            for job_dir in job_dirs:
                try:
                    shutil.rmtree(job_dir)
                except OSError as error:
                    self.args.logger.warning(f"Cleanup: unable to remove {job_dir}: {error}")
        self.args.logger.info("End intermediate file cleanup")

    def run(self):
        """Run pipeline"""
        self.args.logger.info("Begin running serial pipeline")
        if not self.args.compile_results_only:
            # Write all features to file
            self.write_features()
            # Run worker pipeline
            self.args.worker_idx = 0
            self.args.fdr_control = True
            self.args.features_filename = constants.INPUT_FEATURES_FILENAME.format(self.args.output_dir, 0)
            worker.pipeline(self.args)
        results = self.compile_results([self.args.output_dir])
        self.cleanup()
        return results


class CondorPipeline(SerialPipeline):
    """Class managing condor pipeline for distributing load across workers"""
    def __init__(self, args, features):
        super().__init__(args, features)
        self.num_jobs = math.ceil(len(self.features) / self.args.features_per_worker)

    def setup_jobs(self):
        """Setup and run condor jobs"""
        transfer_args = ["analysis_type", "perturbation", "num_permutations", "permutation_test_statistic", "loss_function",
                         "importance_significance_level", "window_search_algorithm", "window_effect_size_threshold"]
        jobs = [None] * self.num_jobs
        for idx in range(self.num_jobs):
            # Create and launch condor job
            features_filename = constants.INPUT_FEATURES_FILENAME.format(self.args.output_dir, idx)
            input_files = [features_filename, self.args.model_filename, self.args.model_loader_filename, self.args.data_filename]
            job_dir = f"{self.args.output_dir}/outputs_{idx}"
            cmd = f"python3 -m anamod.core.worker -worker_idx {idx}"
            for arg in transfer_args:
                if hasattr(self.args, arg):
                    cmd += f" -{arg} {getattr(self.args, arg)}"
            # Relative file paths for non-shared FS, absolute for shared FS
            for name, path in dict(output_dir=job_dir, features_filename=features_filename, model_filename=self.args.model_filename,
                                   model_loader_filename=self.args.model_loader_filename, data_filename=self.args.data_filename).items():
                cmd += f" -{name} {os.path.abspath(path)}" if self.args.shared_filesystem else f" -{name} {os.path.basename(path)}"
            job = CondorJobWrapper(cmd, input_files, job_dir, shared_filesystem=self.args.shared_filesystem,
                                   memory=f"{self.args.memory_requirement}GB", disk=f"{self.args.disk_requirement}GB",
                                   avoid_bad_hosts=self.args.avoid_bad_hosts, retry_arbitrary_failures=self.args.retry_arbitrary_failures,
                                   cleanup=self.args.cleanup)
            jobs[idx] = job
        return jobs

    def run(self):
        """Run condor pipeline"""
        self.args.logger.info("Begin condor pipeline")
        self.write_features()
        jobs = self.setup_jobs()
        if not self.args.compile_results_only:
            running_jobs = OrderedDict.fromkeys(jobs)
            for idx, job in enumerate(jobs):
                features_filename = constants.OUTPUT_FEATURES_FILENAME.format(job.job_dir, idx)
                if os.path.isfile(features_filename):
                    # Outputs computed previously, do not rerun job
                    # TODO: maybe add option to toggle reusing old results
                    running_jobs.pop(job)
                else:
                    job.run()
            CondorJobWrapper.monitor(list(running_jobs.keys()), cleanup=self.args.cleanup)
        job_dirs = [job.job_dir for job in jobs]
        results = self.compile_results(job_dirs)
        self.fdr_control(results)
        self.cleanup(job_dirs)
        self.args.logger.info("End condor pipeline")
        return results

    def fdr_control(self, output_features):
        """Apply hierarchical FDR control to aggregated feature importance results"""
        # Map feature names
        output_feature_map = {output_feature.name: output_feature for output_feature in output_features}
        # Use stored hierarchy to apply FDR control to flattened output features
        # TODO: maybe reconstruct hierarchy over output features - each worker's hierarchy is separate, even though feature names are same across them
        queue = deque()
        root = self.features[0].root
        root = anytree.Node(constants.DUMMY_ROOT, children=[root]) if self.args.analysis_type == constants.HIERARCHICAL else root
        queue.append(root)
        while queue:
            parent = queue.popleft()
            parent_important = output_feature_map[parent.name].important if parent.parent else True
            if not parent.children:
                continue
            pvalues = np.ones(len(parent.children))
            for idx, child in enumerate(parent.children):
                output_feature = output_feature_map[child.name]
                pvalues[idx] = output_feature.overall_pvalue
            adjusted_pvalues, rejected_hypotheses = bh_procedure(pvalues, self.args.importance_significance_level)
            for idx, child in enumerate(parent.children):
                output_feature = output_feature_map[child.name]
                output_feature.overall_pvalue = adjusted_pvalues[idx]
                output_feature.important = rejected_hypotheses[idx] and parent_important
                queue.append(child)
        root.children[0].parent = None if self.args.analysis_type == constants.HIERARCHICAL else root
        # Re-tag temporal importances
        if self.args.analysis_type == constants.TEMPORAL:
            for output_feature in output_features:
                output_feature.window_important &= output_feature.important
                output_feature.ordering_important &= output_feature.important
                output_feature.window_ordering_important &= output_feature.important
