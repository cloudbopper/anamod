"""Constant definitions"""

# Arguments
HIERARCHICAL = "hierarchical"

# Hierarchy
NODE_NAME = "name"
PARENT_NAME = "parent_name"
DESCRIPTION = "description"
INDICES = "idx"
CLUSTER_FROM_DATA = "cluster_from_data"
RANDOM = "random"

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
VIRTUAL_ENV = "VIRTUAL_ENV"
MEMORY_REQUIREMENT = "MEMORY_REQUIREMENT"
SCRIPT_DIR = "SCRIPT_DIR"
NORMAL_FAILURE_COUNT = "Normal failure count"
MAX_NORMAL_FAILURE_COUNT = 5

# Evaluation
EFFECT_SIZE = "effect_size"
MEAN_LOSS = "mean_loss"
PVALUE_LOSSES = "p-value-losses"
PAIRED_TTEST = "paired-t-test"
WILCOXON_TEST = "wilcoxon-test"
PVALUES_FILENAME = "pvalues.csv"
LESS = "less"
GREATER = "greater"
TWOSIDED = "two-sided"

# Interactions
INTERACTIONS_PVALUES_FILENAME = "interaction_pvalues.csv"
DUMMY_ROOT = "dummy_root"
INTERACTIONS_FDR_DIR = "interaction_fdr_results"

# HDF5
LOSSES = "losses"
PREDICTIONS = "predictions"
RECORD_IDS = "record_ids"
TARGETS = "targets"
STATIC = "static"
TEMPORAL = "temporal"
DATA = "data"

# Simulation
RELEVANT = "relevant"
IRRELEVANT = "irrelevant"
MODEL_FILENAME = "model.cpkl"
ADDITIVE_GAUSSIAN = "additive_gaussian"
EPSILON_IRRELEVANT = "epsilon_irrelevant"
NO_NOISE = "none"
FDR = "FDR"
POWER = "Power"
OUTER_NODES_FDR = "Outer_Nodes_FDR"
OUTER_NODES_POWER = "Outer_Nodes_Power"
BASE_FEATURES_FDR = "Base_Features_FDR"
BASE_FEATURES_POWER = "Base_Features_Power"
INTERACTIONS_FDR = "Interactions_FDR"
INTERACTIONS_POWER = "Interactions_Power"
TEMPORAL_FDR = "Temporal_FDR"
TEMPORAL_POWER = "Temporal_Power"
AVERAGE_WINDOW_FDR = "Average_Window_FDR"
AVERAGE_WINDOW_POWER = "Average_Window_Power"
WINDOW_OVERLAP = "Window_Overlap"
SIMULATION_RESULTS = "simulation_results"
SIMULATION_RESULTS_FILENAME = "simulation_results.json"

# Trial (multiple simulations)
DEFAULT = "default"
INSTANCE_COUNTS = "instance_counts"
NOISE_LEVELS = "noise_levels"
FEATURE_COUNTS = "feature_counts"
SHUFFLING_COUNTS = "shuffling_counts"
SEQUENCE_LENGTHS = "sequence_lengths"
WINDOW_SEQUENCE_DEPENDENCE = "window_sequence_dependence"
MODEL_TYPES = "model_types"
ALL_SIMULATION_RESULTS = "all_simulation_results"
CONFIG_HIERARCHICAL = "config_run_simulations_hierarchical.ini"
CONFIG_TEMPORAL = "config_run_simulations_temporal.ini"

# Hierarchical FDR
HIERARCHICAL_FDR_DIR = "hierarchical_fdr_results"
HIERARCHICAL_FDR_OUTPUTS = "hierarchical_fdr_outputs"
ADJUSTED_PVALUE = "adjusted_p-value"
REJECTED_STATUS = "rejected_status"
# Dependence assumptions
POSITIVE = "positive"
ARBITRARY = "arbitrary"
# Procedures
YEKUTIELI = "yekutieli"
LYNCH_GUO = "lynch_guo"
TREE = "tree"

# Perturbation
ZEROING = "zeroing"
SHUFFLING = "shuffling"
RNG_SEED = "rng_seed"
ACROSS_INSTANCES = "across"
WITHIN_INSTANCE = "within"
PVALUE_THRESHOLD = 0.05

# Miscellaneous
BASELINE = u"baseline"
SEED = 13997
BINARY_CLASSIFIER = "binary_classifier"
CLASSIFIER = "classifier"
REGRESSION = "regression"
