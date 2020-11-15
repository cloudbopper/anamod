"""
anamod master pipeline
Given test samples, a trained model and a feature hierarchy,
computes the effect on the model's output loss after perturbing the features/feature groups
in the hierarchy.
"""

import csv
import importlib
import os
import sys
from unittest.mock import patch

import numpy as np

from anamod.core import constants, utils
from anamod.core.pipelines import CondorPipeline, SerialPipeline
from anamod.core.utils import round_value
from anamod.fdr import hierarchical_fdr_control


def main(args):
    """Parse arguments from command-line"""
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.rng = np.random.default_rng(args.seed)
    args.logger = utils.get_logger(__name__, "%s/anamod.log" % args.output_dir)
    validate_args(args)
    return pipeline(args)


def pipeline(args):
    """Master pipeline"""
    # FIXME: some outputs returned via return value (temporal analysis), other via output file (hierarchical analysis)
    # TODO: 'args' is now an object. Change to reflect that and figure out way to print object attributes
    args.logger.info("Begin anamod master pipeline with args: %s" % args)
    # Perturb features
    worker_pipeline = CondorPipeline(args) if args.condor else SerialPipeline(args)
    analyzed_features = worker_pipeline.run()
    if args.analysis_type == constants.HIERARCHICAL:
        hierarchical_fdr(args, analyzed_features)
    # TODO: Analyze pairwise interactions
    args.logger.info("End anamod master pipeline")
    return analyzed_features


def hierarchical_fdr(args, features):
    """Performs hierarchical FDR control on results"""
    # Write FDR control input file
    input_filename = "%s/%s" % (args.output_dir, constants.PVALUES_FILENAME)
    with open(input_filename, "w", newline="") as outfile:
        writer = csv.writer(outfile, delimiter=",")
        writer.writerow([constants.NODE_NAME, constants.PARENT_NAME, constants.DESCRIPTION, constants.EFFECT_SIZE, constants.PVALUE])
        for node in features:
            name = node.name
            parent_name = node.parent.name if node.parent else ""
            writer.writerow([name, parent_name, node.description,
                             round_value(node.overall_effect_size), round_value(node.overall_pvalue)])
    # Run FDR control
    output_dir = "%s/%s" % (args.output_dir, constants.HIERARCHICAL_FDR_DIR)
    cmd = (f"python -m anamod.fdr.hierarchical_fdr_control -output_dir {output_dir} -procedure yekutieli "
           f"-rectangle_leaves 1 -alpha {args.importance_significance_level} {input_filename}")
    args.logger.info("Running cmd: %s" % cmd)
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        hierarchical_fdr_control.main()
    # TODO: update feature importance attributes based on results of hierarchical FDR control
    # Better yet, pass features directly to hierarchical FDR control and update


def validate_args(args):
    """Validate arguments"""
    if args.condor:
        try:
            importlib.import_module("htcondor")
        except ModuleNotFoundError:
            print("htcondor module not found. "
                  "Use 'pip install htcondor' to install htcondor on a compatible platform, or "
                  "disable condor", file=sys.stderr)
            raise
