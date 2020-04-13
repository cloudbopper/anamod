"""Python API to analyze temporal models"""
from abc import ABC
import csv
import importlib
import os
import sys
import tempfile

import anytree
import h5py

from anamod import master, constants, model_loader


class ModelAnalyzer(ABC):
    """Analyzes properties of learned models"""
    def __init__(self, model, data, targets, **kwargs):
        """
        Model:   must provide required interface
        Data:    feature matrix/tensor
        Targets: vector of targets/labels
        """
        # FIXME: Temporary directory may not be in shared location on shared FS
        self.output_dir = kwargs.pop("output_dir") if "output_dir" in kwargs else tempfile.mkdtemp()
        self.model_filename = self.gen_model_file(model, kwargs.get("model_loader_filename", os.path.abspath(model_loader.__file__)))
        self.data_filename = self.gen_data_file(data, targets)
        # TODO: Add optional feature names
        # TODO: Add -condor 1 -shared_filesystem 0 iff htcondor available
        self.cmd = f"python -m anamod -model_filename {self.model_filename} -data_filename {self.data_filename} -output_dir {self.output_dir}"
        self.add_options(**kwargs)

    def add_options(self, **kwargs):
        """Add options to model analysis"""
        for key, value in kwargs.items():
            self.cmd += f" -{key} {value}"

    def analyze(self):
        """Analyze model"""
        strargs = " ".join(self.cmd.split()[3:])
        features = master.main(strargs)
        return features

    def gen_model_file(self, model, model_loader_filename):
        """Generate model file"""
        model_filename = f"{self.output_dir}/{constants.MODEL_FILENAME}"
        assert os.path.exists(model_loader_filename), f"Model loader file {model_loader_filename} does not exist"
        dirname, filename = os.path.split(os.path.abspath(model_loader_filename))
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


class TemporalModelAnalyzer(ModelAnalyzer):
    """Analyzes properties of temporal model"""
    def __init__(self, model, data, targets, **kwargs):
        kwargs["analysis_type"] = constants.TEMPORAL
        super().__init__(model, data, targets, **kwargs)


class HierarchicalModelAnalyzer(ModelAnalyzer):
    """Analyzes hierarchical feature importance"""
    def __init__(self, model, data, targets, hierarchy, **kwargs):
        """Hierarchy in the form of an anytree root node"""
        kwargs["analysis_type"] = constants.HIERARCHICAL
        super().__init__(model, data, targets, **kwargs)
        self.hierarchy_filename = self.gen_hierarchy_file(hierarchy)
        self.cmd += f" -hierarchy_filename {self.hierarchy_filename}"

    def gen_hierarchy_file(self, hierarchy):
        """
        Generate hierarchy CSV file (hierarchical feature importance analysis only)

        Columns:
                *name*:             feature name, must be unique across features
                *parent_name*:      name of parent if it exists, else '' (root node)
                *description*:      node description
                *idx*:              [only required for leaf nodes] list of tab-separated indices corresponding to the indices
                                    of these features in the data
        """
        hierarchy_filename = f"{self.output_dir}/hierarchy.csv"
        with open(hierarchy_filename, "w", newline="") as hierarchy_file:
            writer = csv.writer(hierarchy_file, delimiter=",")
            writer.writerow([constants.NODE_NAME, constants.PARENT_NAME,
                             constants.DESCRIPTION, constants.INDICES])
            for node in anytree.PreOrderIter(hierarchy):
                idx = node.idx if node.is_leaf else ""
                parent_name = node.parent.name if node.parent else ""
                writer.writerow([node.name, parent_name, node.description, idx])
        return hierarchy_filename
