"""Constant definitions"""

# Hierarchy
NODE_NAME = "name"
PARENT_NAME = "parent_name"
DESCRIPTION = "description"
STATIC_INDICES = "static_indices"
TEMPORAL_INDICES = "temporal_indices"
STATIC = "static"
TEMPORAL = "temporal"

# Condor
SUBMIT_FILENAME = "SUBMIT_FILENAME"
ARGS_FILENAME = "ARGS_FILENAME"
LOG_FILENAME = "LOG_FILENAME"
OUTPUT_FILENAME = "OUTPUT_FILENAME"
ERROR_FILENAME = "ERROR_FILENAME"
CMD = "cmd"
ATTEMPT = "attempt"
MAX_ATTEMPTS = 50
JOB_COMPLETE = "job_complete"
JOB_HELD = "Job was held."
JOB_TERMINATED = "Job terminated."
NORMAL_TERMINATION_SUCCESS = "Normal termination (return value 0)"
NORMAL_TERMINATION_FAILURE = "Normal termination (return value 1)"
ABNORMAL_TERMINATION = "Abnormal termination"
CLUSTER = "cluster"
JOB_START_TIME = "job_start_time"

# Evaluation
EFFECT_SIZE = "effect_size"
MEAN_LOSS = "mean_loss"
PVALUE_LOSSES = "p-value-losses"
PAIRED_TTEST = "paired-t-test"
WILCOXON_TEST = "wilcoxon-test"
PVALUES_FILENAME = "pvalues.csv"

# HDF5
RECORDS = "records"
TARGET = "target"
LOSSES = "losses"
PREDICTIONS = "predictions"
TARGETS = "targets"

# Simulation
RELEVANT = "relevant"
IRRELEVANT = "irrelevant"
MODEL_FILENAME = "model.npy"
GEN_MODEL_CONFIG_FILENAME = "gen_model_config.pkl"
GEN_MODEL_FILENAME = "gen_model.py"

# Hierarchical FDR
HIERARCHICAL_FDR_DIR = "hierarchical_fdr_results"
HIERARCHICAL_FDR_OUTPUTS = "hierarchical_fdr_outputs.csv"
ADJUSTED_PVALUE = "adjusted_p-value"
REJECTED_STATUS = "rejected_status"
# Dependence assumptions
POSITIVE = "positive"
ARBITRARY = "arbitrary"
# Procedures
YEKUTIELI = "yekutieli"
LYNCH_GUO = "lynch_guo"


# Miscellaneous
BASELINE = u"baseline"
SEED = 13997
BINARY_CLASSIFIER = "binary_classifier"
CLASSIFIER = "classifier"
REGRESSION = "regression"
