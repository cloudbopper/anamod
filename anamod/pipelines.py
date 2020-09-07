"""Serial and distributed (condor) perturbation pipelines"""

import copy
import glob
import math
import os
import pickle
import shutil

import cloudpickle

from anamod import constants, worker
from anamod.fdr.fdr_algorithms import bh_procedure
from anamod.utils import CondorJobWrapper


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

        self.fdr_control(features)
        return features

    def fdr_control(self, features):
        """Apply FDR control to features"""
        # TODO: better solution to unify hierarchical/temporal analysis FDR control
        if self.args.analysis_type == constants.HIERARCHICAL:
            return
        pvalues = [feature.overall_pvalue for feature in features]
        adjusted_pvalues, rejected_hypotheses = bh_procedure(pvalues, self.args.importance_significance_level)
        for idx, feature in enumerate(features):
            feature.overall_pvalue = adjusted_pvalues[idx]
            if not rejected_hypotheses[idx]:
                feature.important = False

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
        transfer_args = ["analysis_type", "perturbation", "num_shuffling_trials", "loss_function",
                         "importance_significance_level", "window_search_algorithm", "window_effect_size_threshold"]
        jobs = [None] * self.num_jobs
        for idx in range(self.num_jobs):
            # Create and launch condor job
            features_filename = constants.INPUT_FEATURES_FILENAME.format(self.args.output_dir, idx)
            input_files = [features_filename, self.args.model_filename, self.args.model_loader_filename, self.args.data_filename]
            job_dir = f"{self.args.output_dir}/outputs_{idx}"
            cmd = f"python3 -m anamod.worker -worker_idx {idx}"
            for arg in transfer_args:
                cmd += f" -{arg} {self.args.__getattribute__(arg)}"
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
            for job in jobs:
                job.run()
            CondorJobWrapper.monitor(jobs, cleanup=self.args.cleanup)
        job_dirs = [job.job_dir for job in jobs]
        results = self.compile_results(job_dirs)
        self.cleanup(job_dirs)
        self.args.logger.info("End condor pipeline")
        return results
