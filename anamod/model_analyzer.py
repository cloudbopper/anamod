"""Python API to analyze temporal models"""
from abc import ABC
import importlib
import os
import sys

import anytree
import h5py
import numpy as np

from anamod import master, constants, model_loader
from anamod.feature import Feature


class ModelAnalyzer(ABC):
    """Analyze properties of learned models."""
    # pylint: disable = too-many-instance-attributes, line-too-long
    __doc__ += (
        f"""

        **Required parameters:**

            model: object
                A model object that provides the required interface (TODO: add/cite interface documentation).

                This may be a simple wrapper around, say, a scikit-learn or Tensorflow model (TODO: example).

            data: 2D/3D numpy array
                The matrix or tensor containing test data.

                For static models, this must be a matrix of instances **x** features.

                For temporal models, this must be a tensor of instances **x** features **x** sequences.

            targets: 1D numpy array
                A vector containing the targets/labels for instances contained in the data.

        **Common optional parameters:**

            output_dir: str, default: '{constants.DEFAULT_OUTPUT_DIR}'
                Directory to write logs, intermediate files, and outputs to.

            num_permutations: int, default: {constants.DEFAULT_NUM_PERMUTATIONS}
                Number of permutations to perform in permutation test.

            permutation_test_statistic: str, choices: {constants.CHOICES_TEST_STATISTICS}, default: {constants.MEAN_LOSS}
                Test statistic to use for computing empirical p-values

            feature_names: list of strings, default: None
                List of names to be used assigned to features.

                If `None`, features will be identified using their indices as names.

                If :attr:`feature_hierarchy` is provided, names from that will be used instead.

            seed: int, default: {constants.SEED}
                Seed for random number generator (used to order features to be analyzed).

            loss_function: str, choices: {constants.CHOICES_LOSS_FUNCTIONS}, default: None
                Loss function to apply to model outputs.
                If no loss function is specified, then quadratic loss is chosen for continuous targets
                and binary cross-entropy is chosen for binary targets.

            compile_results_only: bool, default: False
                Flag to attempt to compile results only (assuming they already exist), skipping actually launching jobs.

        **Hierarchical feature analysis parameters:**

            feature_hierarchy: object, default: None
                Hierarchy over features, defined as an anytree_ node.
                anytree_ allows importing trees from multiple formats (Python dict, JSON)

                If no hierarchy is provided, a flat hierarchy will be auto-generated over base features.

                Supersedes :attr:`feature_names` for source of feature names.

                .. _anytree: https://anytree.readthedocs.io/en/2.8.0/

            analyze_interactions: bool, default: False
                Flag to enable testing of interaction significance. By default,
                only pairwise interactions between leaf features identified as important by hierarchical FDR.
                are tested. To enable testing of all pairwise interactions, also use -analyze_all_pairwise_interactions.

            analyze_all_pairwise_interactions: bool, default: False
                Analyze all pairwise interactions between leaf features,
                instead of just pairwise interactions of leaf features identified by hierarchical FDR.

        **Temporal model analysis parameters:**

            importance_significance_level: float, default: 0.1
                Significance level used to assess feature importance while testing for overall/window/ordering relevance

            window_search_algorithm: str, choices: {constants.CHOICES_WINDOW_SEARCH_ALGORITHM}, default: '{constants.EFFECT_SIZE}'
                Search algorithm to use to search for relevant window (TODO: document)

            window_effect_size_threshold: float, default: 0.01
                Fraction of total feature importance (effect size) permitted outside window while searching for relevant window

        **HTCondor parameters:**
            condor: bool, default: False
                Flag to enable parallelization using HTCondor.
                Requires package htcondor to be installed (TODO: ref).

            shared_filesystem: bool, default: False
                Flag to indicate a shared filesystem, making
                file/software transfer unnecessary for running condor.

            cleanup: bool, default: True
                Remove intermediate condor files upon completion (typically for debugging).
                Enabled by default to reduced space usage and clutter."

            features_per_worker: int, default: 1
                Number of features to test per condor job. Fewer features per job reduces job
                load at the cost of more jobs.

            memory_requirement: int, default: 8
                Memory requirement in GB

            disk_requirement: int, default: 8
                Disk requirement in GB

            model_loader_filename: str, default: None
                Python script that provides functions to load/save model.
                Required for condor since each job runs in its own environment.
                If none is provided, cloudpickle will be used - see model_loader_ for a template (TODO: fix ref)

                .. _model_loader: py:mod:anamod.model_loader

            avoid_bad_hosts: bool, default: False
                Avoid condor hosts that intermittently give issues.
                Enable to reduce likelihood of failures at the cost of increased runtime.
                List of hosts: {constants.CONDOR_AVOID_HOSTS}

            retry_arbitrary_failures: bool, default: False
                Retry failing jobs due to any reason, up to a maximum of {constants.CONDOR_MAX_RETRIES} attempts per job.
                Use with caution - enable if failures stem from condor issues.
        """)

    def __init__(self, model, data, targets, **kwargs):
        self.kwargs = kwargs
        # Common optional parameters
        self.output_dir = self.process_keyword_arg("output_dir", constants.DEFAULT_OUTPUT_DIR)
        self.perturbation = constants.PERMUTATION  # Zeroing deprecated, removed option
        self.num_permutations = self.process_keyword_arg("num_permutations", constants.DEFAULT_NUM_PERMUTATIONS)
        self.permutation_test_statistic = self.process_keyword_arg("permutation_test_statistic", constants.MEAN_LOSS)
        self.feature_names = self.process_keyword_arg("feature_names", None)
        self.seed = self.process_keyword_arg("seed", constants.SEED)
        self.loss_function = self.process_keyword_arg("loss_function", None, constants.CHOICES_LOSS_FUNCTIONS)
        self.set_loss_function(targets)
        self.compile_results_only = self.process_keyword_arg("compile_results_only", False)
        # Hierarchical feature analysis parameters
        self.feature_hierarchy = self.process_keyword_arg("feature_hierarchy", None)
        self.analyze_interactions = self.process_keyword_arg("analyze_interactions", False)
        if self.analyze_interactions:
            raise NotImplementedError("Interaction analysis currently disabled pending updated theoretical analysis")
        # Temporal model analysis parameters
        self.importance_significance_level = self.process_keyword_arg("importance_significance_level", 0.1)
        self.window_search_algorithm = self.process_keyword_arg("window_search_algorithm", constants.EFFECT_SIZE,
                                                                constants.CHOICES_WINDOW_SEARCH_ALGORITHM)
        # TODO: Automatic proportional selection of window effect size threshold w.r.t. sequence length
        self.window_effect_size_threshold = self.process_keyword_arg("window_effect_size_threshold", 0.01)
        # pylint: disable = invalid-name
        self.analyze_all_pairwise_interactions = self.process_keyword_arg("analyze_all_pairwise_interactions", False)
        # HTCondor parameters
        self.condor = self.process_keyword_arg("condor", False)
        self.shared_filesystem = self.process_keyword_arg("shared_filesystem", False)
        self.cleanup = self.process_keyword_arg("cleanup", True)
        self.features_per_worker = self.process_keyword_arg("features_per_worker", 1)
        self.memory_requirement = self.process_keyword_arg("memory_requirement", 8)
        self.disk_requirement = self.process_keyword_arg("disk_requirement", 32)
        self.model_loader_filename = self.process_keyword_arg("model_loader_filename", None)
        self.avoid_bad_hosts = self.process_keyword_arg("avoid_bad_hosts", True)
        self.retry_arbitrary_failures = self.process_keyword_arg("retry_arbitrary_failures", False)
        # Required parameters
        self.model_filename = self.gen_model_file(model)
        self.data_filename = self.gen_data_file(data, targets)
        # TODO: this method of identifying analysis type may fail unexpectedly if user forgets to provide hierarchy with non-temporal data
        self.analysis_type = constants.HIERARCHICAL if self.feature_hierarchy else constants.TEMPORAL
        self.gen_hierarchy(data)

    def process_keyword_arg(self, argname, default_value, choices=None):
        """Process keyword argument along with simple type validation"""
        value = self.kwargs.get(argname, default_value)
        dtype = type(default_value)
        try:
            value = bool(value) if dtype == bool else value
            assert default_value is None or isinstance(value, dtype)
            assert choices is None or value in choices
        except Exception as exc:
            print(f"Usage:\n\n{self.__doc__}", file=sys.stderr)
            error = f"Invalid argument for keyword {argname}: {value}; default: {default_value}, type {dtype}"
            if choices is not None:
                error += f"; choices: {choices}"
            raise ValueError(error) from exc
        return value

    def analyze(self):
        """
        Analyze model and return analyzed features.

        Returns
        -------
        features: list <feature object>
            List of feature objects with importance analysis information. TODO: Describe feature objects
        """
        features = master.main(self)
        return features

    def gen_model_file(self, model):
        """Generate model file"""
        if self.model_loader_filename is None:
            self.model_loader_filename = os.path.abspath(model_loader.__file__)
        model_filename = f"{self.output_dir}/{constants.MODEL_FILENAME}"
        assert os.path.exists(self.model_loader_filename), f"Model loader file {self.model_loader_filename} does not exist"
        dirname, filename = os.path.split(os.path.abspath(self.model_loader_filename))
        sys.path.insert(1, dirname)
        loader = importlib.import_module(os.path.splitext(filename)[0])
        loader.save_model(model, model_filename)
        return model_filename

    def gen_data_file(self, data, targets):
        """Generate data file"""
        data_filename = f"{self.output_dir}/{constants.DATA_FILENAME}"
        root = h5py.File(data_filename, "w")
        num_instances = data.shape[0]
        record_ids = [str(idx).encode("utf8") for idx in range(num_instances)]
        root.create_dataset(constants.RECORD_IDS, data=record_ids)
        root.create_dataset(constants.DATA, data=data)
        root.create_dataset(constants.TARGETS, data=targets)
        root.close()
        return data_filename

    def gen_hierarchy(self, data):
        """
        Create a new feature hierarchy:
        (i) from input hierarchy if available, and
        (ii) from feature set if not
        """
        num_features = data.shape[1]
        if self.feature_hierarchy is None:
            # Create hierarchy if not available
            if self.feature_names is None:
                # Generate feature names if not available
                self.feature_names = [f"{idx}" for idx in range(num_features)]
            root = Feature(constants.DUMMY_ROOT, description=constants.DUMMY_ROOT, perturbable=False)  # Dummy node, shouldn't be perturbed
            for idx, feature_name in enumerate(self.feature_names):
                Feature(feature_name, parent=root, idx=[idx])
            self.feature_hierarchy = root
        else:
            # TODO: Document real hierarchy with examples
            # Input hierarchy needs a list of indices assigned to all base features
            # Create hierarchy over features from input hierarchy
            feature_nodes = {}
            all_idx = set()
            # Parse and validate input hierarchy
            for node in anytree.PostOrderIter(self.feature_hierarchy):
                idx = []
                if node.is_leaf:
                    valid = (hasattr(node, "idx") and
                             isinstance(node.idx, list) and
                             len(node.idx) >= 1 and
                             all([isinstance(node.idx[i], int) for i in range(len(node.idx))]))
                    assert valid, f"Leaf node {node.name} must contain a non-empty list of integer indices under attribute 'idx'"
                    assert not all_idx.intersection(node.idx), f"Leaf node {node.name} has index overlap with other leaf nodes"
                    idx = node.idx
                    all_idx.update(idx)
                else:
                    # Ensure internal nodes have empty initial indices
                    valid = not hasattr(node, "idx") or not node.idx
                    assert valid, f"Internal node {node.name} must have empty initial indices under attribute 'idx'"
                feature_nodes[node.name] = Feature(node.name, description=node.description, idx=idx)
            # Update feature group (internal node) indices and tree connections
            assert min(all_idx) >= 0 and max(all_idx) < num_features, "Feature indices in hierarchy must be in range [0, num_features - 1]"
            feature_node = None
            for node in anytree.PostOrderIter(self.feature_hierarchy):
                feature_node = feature_nodes[node.name]
                parent = node.parent
                if parent:
                    feature_node.parent = feature_nodes[parent.name]
                for child in node.children:
                    feature_node.idx += feature_nodes[child.name].idx
            self.feature_hierarchy = feature_node  # root

    def set_loss_function(self, targets):
        """Set loss function if not provided based on inferred model type"""
        if self.loss_function is not None:
            return
        num_unique_targets = np.unique(targets).shape[0]
        if num_unique_targets == 2:
            self.loss_function = constants.BINARY_CROSS_ENTROPY
        elif num_unique_targets > len(targets) / 10:
            self.loss_function = constants.QUADRATIC_LOSS
        else:
            raise ValueError(f"Unable to infer loss function automatically; number of unique targets: {num_unique_targets}; "
                             f"set loss_function to one of '{constants.CHOICES_LOSS_FUNCTIONS}'")
